from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice


@triton.jit
def _quant16(v, g, NB: tl.constexpr):
    bamax = tl.max(tl.abs(v), axis=1)
    s8 = tl.minimum(tl.math.div_rn(bamax, 6.0 * g), 448.0).to(tl.float8e4nv)
    sf = tl.maximum(s8.to(tl.float32) * g, 1e-30)
    q = tl.minimum(tl.maximum(tl.math.div_rn(v, sf[:, None]), -6.0), 6.0)
    a = tl.abs(q)
    idx = ((a > 0.25).to(tl.int32) + (a >= 0.75).to(tl.int32) + (a > 1.25).to(tl.int32) + (a >= 1.75).to(tl.int32)
           + (a > 2.5).to(tl.int32) + (a >= 3.5).to(tl.int32) + (a > 5.0).to(tl.int32))
    code = idx | ((q < 0).to(tl.int32) << 3)
    return code, s8.to(tl.uint8, bitcast=True)


@triton.jit
def _store(code, sbyte, out_ptr, sc_ptr, row, blk0, K, CB, NB: tl.constexpr, NVALID: tl.constexpr = 0):
    c3 = tl.reshape(code, [NB, 8, 2])
    lo, hi = tl.split(c3)
    packed = (lo | (hi << 4)).to(tl.uint8)
    blk = blk0 + tl.arange(0, NB)
    ok = blk < (NVALID if NVALID else 1 << 30)
    tl.store(out_ptr + row * (K // 2) + blk[:, None] * 8 + tl.arange(0, 8)[None, :], packed, mask=ok[:, None])
    off = ((row // 128) * CB + blk // 4) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + (blk % 4)
    tl.store(sc_ptr + off, sbyte, mask=ok)


@triton.jit
def _quant_kernel(x_ptr, g_ptr, out_ptr, sc_ptr, K, CB, NB: tl.constexpr):
    row, pk = tl.program_id(0).to(tl.int64), tl.program_id(1)
    g = tl.load(g_ptr)
    blk0 = pk * NB
    offs = (blk0 + tl.arange(0, NB))[:, None] * 16 + tl.arange(0, 16)[None, :]
    v = tl.load(x_ptr + row * K + offs).to(tl.float32)
    code, sb = _quant16(v, g, NB)
    _store(code, sb, out_ptr, sc_ptr, row, blk0, K, CB, NB)


@triton.jit
def _silu_mul_quant_kernel(gu_ptr, g_ptr, out_ptr, sc_ptr, I, CB, NB: tl.constexpr):
    row, pk = tl.program_id(0).to(tl.int64), tl.program_id(1)
    g = tl.load(g_ptr)
    blk0 = pk * NB
    offs = (blk0 + tl.arange(0, NB))[:, None] * 16 + tl.arange(0, 16)[None, :]
    gate = tl.load(gu_ptr + row * (2 * I) + offs).to(tl.float32)
    up = tl.load(gu_ptr + row * (2 * I) + I + offs).to(tl.float32)
    act = (gate * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
    h = (act * up).to(tl.bfloat16).to(tl.float32)
    code, sb = _quant16(h, g, NB)
    _store(code, sb, out_ptr, sc_ptr, row, blk0, I, CB, NB)


@triton.jit
def _add_rms_quant_kernel(h_ptr, d_ptr, w_ptr, g_ptr, ho_ptr, x_ptr, out_ptr, sc_ptr, eps, K, CB, N: tl.constexpr,
                          NBLK: tl.constexpr, NB: tl.constexpr, HAS_X: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    g = tl.load(g_ptr)
    blk = tl.arange(0, NB)
    offs = blk[:, None] * 16 + tl.arange(0, 16)[None, :]
    m = (blk < NBLK)[:, None]
    h = tl.load(h_ptr + row * N + offs, mask=m, other=0.0).to(tl.float32)
    d = tl.load(d_ptr + row * N + offs, mask=m, other=0.0).to(tl.float32)
    hs = h + d
    hn = hs.to(tl.bfloat16)
    tl.store(ho_ptr + row * N + offs, hn, mask=m)
    xf = hn.to(tl.float32)
    rs = libdevice.rsqrt(tl.sum(tl.sum(hs * hs, axis=1), axis=0) / N + eps)
    w = tl.load(w_ptr + offs, mask=m, other=0.0).to(tl.float32)
    x = ((xf * rs) * (1.0 + w)).to(tl.bfloat16)
    if HAS_X:
        tl.store(x_ptr + row * N + offs, x, mask=m)
    code, sb = _quant16(x.to(tl.float32), g, NB)
    _store(code, sb, out_ptr, sc_ptr, row, 0, K, CB, NB, NBLK)


@triton.jit
def _gated_norm_quant_kernel(o_ptr, z_ptr, w_ptr, g_ptr, out_ptr, sc_ptr, z_stride, eps, K, CB, NH: tl.constexpr,
                             NHP: tl.constexpr, HD: tl.constexpr, NB: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    g = tl.load(g_ptr)
    h = tl.arange(0, NHP)[:, None]
    dd = tl.arange(0, HD)[None, :]
    m = h < NH
    o = tl.load(o_ptr + t * NH * HD + h * HD + dd, mask=m, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + t * z_stride + h * HD + dd, mask=m, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + dd).to(tl.float32)
    r = libdevice.rsqrt(tl.sum(o * o, axis=1) / HD + eps)
    y = ((w * (o * r[:, None])) * (z / (1.0 + libdevice.exp(-z)))).to(tl.bfloat16).to(tl.float32)
    code, sb = _quant16(tl.reshape(y, [NB, 16]), g, NB)
    _store(code, sb, out_ptr, sc_ptr, t, 0, K, CB, NB, NH * (HD // 16))


def _nb(K: int) -> int:
    for nb in (32, 16, 8, 4, 2, 1):
        if K % (16 * nb) == 0:
            return nb
    raise ValueError(K)


def _alloc(M: int, K: int, dev):
    CB = -(-(K // 16) // 4)
    sc = torch.zeros(-(-M // 128) * CB * 512, device=dev, dtype=torch.uint8)
    return torch.empty(M, K // 2, device=dev, dtype=torch.uint8), sc, CB


def global_scale(amax: torch.Tensor) -> torch.Tensor:
    return (amax / (448.0 * 6.0)).clamp(min=1e-12).float().reshape(1)


def nvfp4_quant_static(x: torch.Tensor, g: torch.Tensor):
    M, K = x.shape
    out, sc, CB = _alloc(M, K, x.device)
    nb = _nb(K)
    _quant_kernel[(M, K // (16 * nb))](x, g, out, sc, K, CB, NB=nb)
    return out.view(torch.float4_e2m1fn_x2), sc.view(torch.float8_e4m3fn), g


def add_rms_nvfp4_static(h: torch.Tensor, d: torch.Tensor, w: torch.Tensor, eps: float, g: torch.Tensor, need_x: bool):
    M, K = h.shape
    h, d = h.contiguous(), d.contiguous()
    out, sc, CB = _alloc(M, K, h.device)
    ho = torch.empty_like(h)
    x = torch.empty_like(h) if need_x else ho
    nblk = K // 16
    _add_rms_quant_kernel[(M,)](h, d, w, g, ho, x, out, sc, eps, K, CB, N=K, NBLK=nblk, NB=triton.next_power_of_2(nblk),
                                HAS_X=need_x, num_warps=8)
    return ho, x, out.view(torch.float4_e2m1fn_x2), sc.view(torch.float8_e4m3fn)


def gated_norm_nvfp4_static(o: torch.Tensor, z: torch.Tensor, w: torch.Tensor, eps: float, g: torch.Tensor, T: int):
    nh, hd = o.shape[-2], o.shape[-1]
    K = nh * hd
    out, sc, CB = _alloc(T, K, o.device)
    _gated_norm_quant_kernel[(T,)](o, z, w, g, out, sc, z.stride(0), eps, K, CB, NH=nh, NHP=triton.next_power_of_2(nh), HD=hd,
                                   NB=triton.next_power_of_2(nh) * (hd // 16), num_warps=8)
    return out.view(torch.float4_e2m1fn_x2), sc.view(torch.float8_e4m3fn)


def silu_mul_nvfp4_static(gu: torch.Tensor, g: torch.Tensor):
    M, twoI = gu.shape
    I = twoI // 2
    out, sc, CB = _alloc(M, I, gu.device)
    nb = _nb(I)
    _silu_mul_quant_kernel[(M, I // (16 * nb))](gu, g, out, sc, I, CB, NB=nb)
    return out.view(torch.float4_e2m1fn_x2), sc.view(torch.float8_e4m3fn), g


if __name__ == "__main__":
    import time
    from fp4 import quantize_nvfp4
    torch.manual_seed(0)
    for M, K in [(8192, 5120), (777, 5120), (4096, 17408), (130, 6144), (1, 5120)]:
        x = (torch.randn(M, K, device="cuda") * torch.rand(1, K, device="cuda") * 5).bfloat16()
        amax = x.abs().max().float()
        g = global_scale(amax)
        rq, rs, rg = quantize_nvfp4(x, amax)
        tq, ts, _ = nvfp4_quant_static(x, g)
        nbytes = rs.numel()
        print(f"M{M} K{K}: packed equal {(rq.view(torch.uint8) == tq.view(torch.uint8)).float().mean().item():.6f}, "
              f"scales equal(rows<M) {(rs.view(torch.uint8)[:nbytes] == ts.view(torch.uint8)[:nbytes]).float().mean().item():.6f}, "
              f"g {abs(rg.item() - g.item()):.2e}")
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(50):
            nvfp4_quant_static(x, g)
        torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 50
        print(f"   triton quant {dt*1e6:.0f} us, {M*K*2/dt/1e9:.0f} GB/s read")
    M, I = 4096, 17408
    gu = torch.randn(M, 2 * I, device="cuda").bfloat16()
    h = torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]
    g = global_scale(h.abs().max().float())
    rq, rs, _ = quantize_nvfp4(h, h.abs().max().float())
    tq, ts, _ = silu_mul_nvfp4_static(gu, g)
    print("silu_mul packed equal", (rq.view(torch.uint8) == tq.view(torch.uint8)).float().mean().item(),
          "scales equal", (rs.view(torch.uint8) == ts.view(torch.uint8)[:rs.numel()]).float().mean().item())
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(50):
        silu_mul_nvfp4_static(gu, g)
    torch.cuda.synchronize(); print(f"   silu_mul+quant {(time.perf_counter()-t)/50*1e6:.0f} us")
