from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from triton.runtime.jit import JITFunction

ON = os.environ.get("Q35_CONV1", "0") in ("1", "2")
PS = os.environ.get("Q35_CONV1", "0") == "2"
BD = int(os.environ.get("Q35_CONV1_BD", "64"))
NW = int(os.environ.get("Q35_CONV1_WARPS", "4"))
_K = {}


def _jit(k):
    while not isinstance(k, JITFunction):
        k = k.fn
    return k


def _kernels():
    if not _K:
        import fla.modules.conv.triton.kernels as ck
        _K["fwd"] = _jit(ck.causal_conv1d_fwd_kernel)
        _K["states"] = _jit(ck.causal_conv1d_states_fwd_kernel)
    return _K["fwd"], _K["states"]


def conv(x, weight, cu_seqlens, chunk_indices, initial_state=None, bd=None, num_warps=None):
    fwd, _ = _kernels()
    T, D = x.shape
    W = weight.shape[1]
    y = torch.empty(T, D, device=x.device, dtype=x.dtype)
    bd = bd or BD
    fwd[(triton.cdiv(D, bd), len(chunk_indices), 1)](
        x, y, weight, None, None, cu_seqlens, initial_state, chunk_indices, 1, T, 0, x.stride(0), x.stride(1),
        D=D, W=W, BT=64, BW=triton.next_power_of_2(W), BD=bd, NB=triton.cdiv(T, 1024), ACTIVATION="silu",
        HAS_WEIGHT=True, HAS_BIAS=False, HAS_RESIDUAL=False, USE_INITIAL_STATE=initial_state is not None,
        IS_VARLEN=True, num_warps=num_warps or NW)
    return y


def states(x, W, cu_seqlens):
    _, st = _kernels()
    T, D = x.shape
    N = len(cu_seqlens) - 1
    out = torch.empty(N, D, W, device=x.device, dtype=x.dtype)
    bd = min(triton.next_power_of_2(D), 256)
    st[(triton.cdiv(D, bd), N)](x, None, out, cu_seqlens, T, D, W, 0, x.stride(0), x.stride(1), BD=bd,
                                BW=triton.next_power_of_2(W), USE_INITIAL_STATE=False, IS_VARLEN=True)
    return out


@triton.jit
def conv_ps_kernel(x, y, weight, cu_p, ci_p, cu_s, ci_s, s_par, NTP, TP, stride_x_t, stride_x_d, D: tl.constexpr,
                   W: tl.constexpr, BT: tl.constexpr, BW: tl.constexpr, BD: tl.constexpr, PAR_IDENT: tl.constexpr):
    i_d, i_c = tl.program_id(0), tl.program_id(1)
    if i_c < NTP:
        i_n, i_t = tl.load(ci_p + i_c * 2).to(tl.int32), tl.load(ci_p + i_c * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_p + i_n).to(tl.int64), tl.load(cu_p + i_n + 1).to(tl.int64)
        pb = bos
        pe = bos
    else:
        i_n, i_t = tl.load(ci_s + (i_c - NTP) * 2).to(tl.int32), tl.load(ci_s + (i_c - NTP) * 2 + 1).to(tl.int32)
        bos, eos = TP + tl.load(cu_s + i_n).to(tl.int64), TP + tl.load(cu_s + i_n + 1).to(tl.int64)
        if PAR_IDENT:
            i_p = i_n
        else:
            i_p = tl.load(s_par + i_n).to(tl.int32)
        pb, pe = tl.load(cu_p + i_p).to(tl.int64), tl.load(cu_p + i_p + 1).to(tl.int64)
    T = eos - bos
    p_x = x + bos * stride_x_t

    o_d = i_d * BD + tl.arange(0, BD)
    o_t = i_t * BT + tl.arange(0, BT)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0
    m_y = (o_t < T)[:, None] & m_d[None, :]

    b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0).to(tl.float32)

    b_y = tl.zeros((BT, BD), dtype=tl.float32)
    if (i_c < NTP) | (i_t * BT >= W):
        for i_w in tl.static_range(-W + 1, 1):
            o_x = o_t + i_w
            p_yi = p_x + o_x[:, None] * stride_x_t + o_d[None, :] * stride_x_d
            b_yi = tl.load(p_yi, mask=((o_x >= 0) & (o_x < T))[:, None] & m_d[None, :], other=0.0).to(tl.float32)
            b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi
    else:
        for i_w in tl.static_range(-W + 1, 1):
            o_x = o_t + i_w
            m_x = ((o_x >= 0) & (o_x < T))[:, None] & m_d
            m_c = ((o_x + W >= 0) & (o_x < 0) & (pe + o_x >= pb))[:, None] & m_d
            b_yi = tl.load(p_x + o_x[:, None] * stride_x_t + o_d * stride_x_d, mask=m_x, other=0).to(tl.float32)
            b_yi += tl.load(x + (pe + o_x)[:, None] * stride_x_t + o_d * stride_x_d, mask=m_c, other=0).to(tl.float32)
            b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi

    b_y = b_y * tl.sigmoid(b_y)
    p_y = y + bos * D + o_t[:, None] * D + o_d[None, :]
    tl.store(p_y, tl.cast(b_y, dtype=p_y.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_y)


def conv_ps(x, weight, cu_p, ci_p, cu_s, ci_s, s_par, Tp, bd=None, num_warps=None):
    T, D = x.shape
    W = weight.shape[1]
    y = torch.empty(T, D, device=x.device, dtype=x.dtype)
    bd = bd or BD
    ntp = len(ci_p)
    nts = len(ci_s) if ci_s is not None else 0
    ident = isinstance(s_par, slice)
    assert not ident or s_par == slice(None), s_par
    conv_ps_kernel[(triton.cdiv(D, bd), ntp + nts)](
        x, y, weight, cu_p, ci_p, cu_s if nts else cu_p, ci_s if nts else ci_p, s_par if nts and not ident else cu_p, ntp,
        Tp, x.stride(0), x.stride(1), D=D, W=W, BT=64, BW=triton.next_power_of_2(W), BD=bd, PAR_IDENT=ident,
        num_warps=num_warps or NW)
    return y
