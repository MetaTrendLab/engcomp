#!/usr/bin/env python3
import sys

import torch
from vllm import _custom_ops as ops

sys.path.insert(0, "/opt/reax/engine")
from f8g6 import F8G6

SHAPES = {"qkv": (16384, 5376), "o": (5376, 8192), "gu": (43008, 5376), "down": (5376, 21504)}
Ms = [int(a) for a in sys.argv[1:]] or [256, 512, 1024, 2048, 3072]


def t_ms(fn, reps=9):
    fn()
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    ts.sort()
    return ts[len(ts) // 2]


g = F8G6()
gen = torch.Generator(device="cuda").manual_seed(3)
for name, (N, K) in SHAPES.items():
    b = (torch.randn(N, K, device="cuda", generator=gen) * 2).to(torch.float8_e4m3fn)
    w_q = b.t()
    w_s = torch.tensor([0.0123], device="cuda")
    sbv = w_s.expand(N).contiguous()
    for M in Ms:
        a = (torch.randn(M, K, device="cuda", generator=gen) * 2).to(torch.float8_e4m3fn)
        x_s = torch.rand(M, 1, device="cuda", generator=gen) * 0.02 + 0.001
        ref = lambda: ops.cutlass_scaled_mm(a, w_q, scale_a=x_s, scale_b=w_s, out_dtype=torch.bfloat16)
        c0 = ref()
        t_ref = t_ms(ref)
        res, same9, same7 = [], [], []
        for cfg in g.fn:
            if not cfg.startswith(("g9", "g7_")) or not g.fits(cfg, N, K):
                continue
            ok = torch.equal(g(a, b, x_s.view(-1), sbv, cfg=cfg, gm=8), c0)
            (same9 if cfg.startswith("g9") else same7).append(ok)
            if cfg.startswith("g9") and ok:
                for gm in (8, 16):
                    res.append((t_ms(lambda: g(a, b, x_s.view(-1), sbv, cfg=cfg, gm=gm)), cfg, gm))
        res.sort()
        fl = 2.0 * M * N * K / 1e9
        best = f"{res[0][1]}/{res[0][2]} {res[0][0] * 1e3:.1f} us {fl / res[0][0]:.0f} TF ({t_ref / res[0][0]:.3f}x)" if res else "-"
        print(f"{name:5s} N {N:>5} K {K:>5} M {M:>5}: cutlass {t_ref * 1e3:7.1f} us {fl / t_ref:4.0f} TF | g9 byte-identical "
              f"{sum(same9)}/{len(same9)}, g7 {sum(same7)}/{len(same7)} | best {best}", flush=True)
