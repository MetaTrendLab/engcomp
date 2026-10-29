from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rowwise_kernel(a_ptr, b_ptr, c_ptr, sa_ptr, sb_ptr, M, N, K, sam, sbk, sbn, scm,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    npm, npn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    gid = pid // (GM * npn)
    fm = gid * GM
    gs = min(npm - fm, GM)
    pm = fm + (pid % (GM * npn)) % gs
    pn = (pid % (GM * npn)) // gs
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a = a_ptr + rm[:, None] * sam + rk[None, :]
    b = b_ptr + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        acc = tl.dot(tl.load(a, mask=rm[:, None] < M, other=0.0), tl.load(b, mask=rn[None, :] < N, other=0.0), acc)
        a += BK
        b += BK * sbk
    sa = tl.load(sa_ptr + rm, mask=rm < M, other=0.0)
    sb = tl.load(sb_ptr + rn, mask=rn < N, other=0.0)
    acc = acc * sa[:, None] * sb[None, :]
    tl.store(c_ptr + rm[:, None] * scm + rn[None, :], acc.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _gu_kernel(a_ptr, b_ptr, y_ptr, amax_ptr, xs_ptr, ws_ptr, M, N, K, sam, sbk, sbn, sym,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    npm, npn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    gid = pid // (GM * npn)
    fm = gid * GM
    gs = min(npm - fm, GM)
    pm = fm + (pid % (GM * npn)) % gs
    pn = (pid % (GM * npn)) // gs
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a = a_ptr + rm[:, None] * sam + rk[None, :]
    b = b_ptr + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        acc = tl.dot(tl.load(a, mask=rm[:, None] < M, other=0.0), tl.load(b), acc)
        a += BK
        b += BK * sbk
    v = acc.to(tl.bfloat16).to(tl.float32)
    xs = tl.load(xs_ptr + rm, mask=rm < M, other=0.0)
    ws = tl.load(ws_ptr + rn)
    v = v * xs[:, None] * ws[None, :]
    g, u = tl.split(tl.reshape(v, (BM, BN // 2, 2)))
    a_ = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    y = (a_ * u.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    ri = pn * (BN // 2) + tl.arange(0, BN // 2)
    mm = rm < M
    tl.store(y_ptr + rm[:, None] * sym + ri[None, :], y, mask=mm[:, None])
    am = tl.max(tl.abs(y.to(tl.float32)), axis=1)
    tl.atomic_max(amax_ptr + rm, am, mask=mm)


@triton.jit
def _quant_kernel(y_ptr, amax_ptr, q_ptr, s_ptr, I, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    o = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = o < I
    s = tl.maximum(tl.load(amax_ptr + r), 1e-12) / 448.0
    y = tl.load(y_ptr + r * I + o, mask=m, other=0.0).to(tl.float32)
    tl.store(q_ptr + r * I + o, tl.minimum(tl.maximum(y / s, -448.0), 448.0).to(tl.float8e4nv), mask=m)
    if tl.program_id(1) == 0:
        tl.store(s_ptr + r, s)


CFG_ROW = dict(BM=128, BN=128, BK=128, GM=8, num_warps=8, num_stages=3)
CFG_GU = dict(BM=128, BN=128, BK=128, GM=8, num_warps=8, num_stages=3)


def linear_rowwise(x8, xs, w8t, ws, cfg=None):
    c_ = cfg or CFG_ROW
    M, K = x8.shape
    N = w8t.shape[1]
    out = torch.empty(M, N, device=x8.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, c_["BM"]) * triton.cdiv(N, c_["BN"]),)
    _rowwise_kernel[grid](x8, w8t, out, xs, ws, M, N, K, x8.stride(0), w8t.stride(0), w8t.stride(1), out.stride(0), **c_)
    return out


def interleave_gu(w8t, ws, inter):
    idx = torch.stack([torch.arange(inter), torch.arange(inter) + inter], 1).reshape(-1).to(w8t.device)
    w = w8t.t()[idx].contiguous().t()
    return w, ws[:, idx].contiguous()


def gu_swiglu_q(x8, xs, w8i, wsi, cfg=None):
    c_ = cfg or CFG_GU
    M, K = x8.shape
    N = w8i.shape[1]
    inter = N // 2
    y = torch.empty(M, inter, device=x8.device, dtype=torch.bfloat16)
    amax = torch.zeros(M, device=x8.device, dtype=torch.float32)
    grid = (triton.cdiv(M, c_["BM"]) * triton.cdiv(N, c_["BN"]),)
    _gu_kernel[grid](x8, w8i, y, amax, xs, wsi, M, N, K, x8.stride(0), w8i.stride(0), w8i.stride(1), y.stride(0), **c_)
    q = torch.empty(M, inter, device=x8.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(M, 1, device=x8.device, dtype=torch.float32)
    B = 2048
    _quant_kernel[(M, triton.cdiv(inter, B))](y, amax, q, s, inter, BLOCK=B, num_warps=8)
    return q, s


def quant_rows(y, amax):
    M, inter = y.shape
    q = torch.empty(M, inter, device=y.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(M, 1, device=y.device, dtype=torch.float32)
    B = 2048
    _quant_kernel[(M, triton.cdiv(inter, B))](y, amax, q, s, inter, BLOCK=B, num_warps=8)
    return q, s
