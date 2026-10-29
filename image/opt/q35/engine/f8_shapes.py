#!/usr/bin/env python3
import sys

import torch

sys.path.insert(0, "/opt/reax/engine")
from f8cubin import CFGS, F8Gemm

SHAPES = {"dn_qkvz": (16384, 5120), "dn_out": (5120, 6144), "attn_qkv": (14336, 5120), "attn_o": (5120, 6144),
          "mlp_gu": (34816, 5120), "mlp_down": (5120, 17408)}
LAYERS = {"dn_qkvz": 48, "dn_out": 48, "attn_qkv": 16, "attn_o": 16, "mlp_gu": 64, "mlp_down": 64}
PEAK = 930.0


def t_ms(fn, reps=15):
    for _ in range(2):
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


def main():
    Ms = [int(a) for a in sys.argv[1:]] or [3040]
    g = F8Gemm()
    gen = torch.Generator(device="cuda").manual_seed(5)
    plain = [c for c in CFGS if c.startswith("f8_")]
    for M in Ms:
        tot = {"ref": 0.0, "best": 0.0, "peak": 0.0}
        for name, (N, K) in SHAPES.items():
            a = (torch.randn(M, K, device="cuda", generator=gen) * 2).to(torch.float8_e4m3fn)
            b = (torch.randn(N, K, device="cuda", generator=gen) * 2).to(torch.float8_e4m3fn)
            sa = torch.rand(M, 1, device="cuda", generator=gen) + 0.5
            sb = torch.rand(1, N, device="cuda", generator=gen) + 0.5
            one = torch.ones((), device="cuda")
            fl = 2.0 * M * N * K / 1e9
            row = lambda: torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
            ten = lambda: torch._scaled_mm(a, b.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
            t_row, t_ten = t_ms(row), t_ms(ten)
            if M == Ms[0]:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as pr:
                    row()
                    torch.cuda.synchronize()
                ks = [e.name for e in pr.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                print(f"  {name}: rowwise kernel {ks[0][:110] if ks else '?'}", flush=True)
            c_row, c_ten = row(), ten()
            sa1, sb1 = sa.view(-1).contiguous(), sb.view(-1).contiguous()
            res = []
            for cfg in plain:
                BM, BN, BK = CFGS[cfg][:3]
                if N % BN or K % BK:
                    continue
                for gm in (8, 16):
                    try:
                        t = t_ms(lambda: g(a, b, sa1, sb1, cfg=cfg, gm=gm))
                        same_r = torch.equal(g(a, b, sa1, sb1, cfg=cfg, gm=gm), c_row)
                        same_t = torch.equal(g(a, b, None, None, cfg=cfg, gm=gm), c_ten)
                    except RuntimeError as e:
                        print(f"  {cfg} gm{gm}: {e}", flush=True)
                        continue
                    res.append((t, cfg, gm, same_r, same_t))
            res.sort()
            bt, bc, bgm, br, bte = res[0]
            print(f"M {M:>5} {name:9s} N {N:>5} K {K:>5}: rowwise {t_row * 1e3:7.1f} us {fl / t_row:4.0f} TF | tensor "
                  f"{t_ten * 1e3:7.1f} us {fl / t_ten:4.0f} TF | ours best {bc} gm{bgm} {bt * 1e3:7.1f} us {fl / bt:4.0f} TF "
                  f"(same row/ten {br}/{bte}) | top3 " + ", ".join(f"{c}/{gg} {fl / t:.0f}" for t, c, gg, _, _ in res[:3]),
                  flush=True)
            w = LAYERS[name]
            tot["ref"] += w * t_row
            tot["best"] += w * min(bt, t_row)
            tot["peak"] += w * fl / PEAK
        print(f"M {M}: per-step GEMM time ref-best {tot['ref']:.1f} ms, with ours where faster {tot['best']:.1f} ms "
              f"({tot['ref'] / tot['best']:.3f}x), peak-bound {tot['peak']:.1f} ms ({tot['ref'] / tot['peak']:.2f}x headroom)",
              flush=True)


if __name__ == "__main__":
    main()
