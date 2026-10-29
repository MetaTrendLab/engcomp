#!/usr/bin/env python3
import json
import sys

import torch

sys.path.insert(0, "/opt/reax/engine")
from f8g6 import F8G6

SHAPES = {"dn_qkvz": (16384, 5120), "dn_out": (5120, 6144), "attn_qkv": (14336, 5120), "attn_o": (5120, 6144),
          "mlp_gu": (34816, 5120), "mlp_down": (5120, 17408)}
LAYERS = {"dn_qkvz": 48, "dn_out": 48, "attn_qkv": 16, "attn_o": 16, "mlp_gu": 64, "mlp_down": 64}


def t_ms(fn, reps=11):
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
    args = sys.argv[1:]
    out = None
    if "--json" in args:
        i = args.index("--json")
        out = args[i + 1]
        args = args[:i] + args[i + 2:]
    Ms = [int(a) for a in args] or [3040]
    g = F8G6()
    print(f"{len(g.fn)} kernels loaded", flush=True)
    gen = torch.Generator(device="cuda").manual_seed(5)
    table, bad = {}, []
    for M in Ms:
        tot_ref = tot_best = 0.0
        for name, (N, K) in SHAPES.items():
            a = (torch.randn(M, K, device="cuda", generator=gen) * 2).to(torch.float8_e4m3fn)
            b = (torch.randn(N, K, device="cuda", generator=gen) * 2).to(torch.float8_e4m3fn)
            sa = torch.rand(M, 1, device="cuda", generator=gen) + 0.5
            sb = torch.rand(1, N, device="cuda", generator=gen) + 0.5
            one = torch.ones((), device="cuda")
            row = lambda: torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
            t_row = t_ms(row)
            c_row = row()
            c_ten = torch._scaled_mm(a, b.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
            sa1, sb1 = sa.view(-1).contiguous(), sb.view(-1).contiguous()
            fl = 2.0 * M * N * K / 1e9
            res = []
            for cfg in g.fn:
                if not g.fits(cfg, N, K):
                    continue
                for gm in (8, 16):
                    t = t_ms(lambda: g(a, b, sa1, sb1, cfg=cfg, gm=gm))
                    if gm == 8:
                        ok = torch.equal(g(a, b, sa1, sb1, cfg=cfg, gm=gm), c_row) and torch.equal(
                            g(a, b, None, None, cfg=cfg, gm=gm), c_ten)
                        if not ok:
                            bad.append((name, M, cfg))
                    res.append((t, cfg, gm, ok))
            res = [r for r in res if r[3]]
            res.sort()
            bt, bc, bgm, _ = res[0]
            table.setdefault(f"{N}:{K}", []).append([M, bc, bgm, round(t_row / bt, 4)])
            print(f"M {M:>5} {name:9s}: ref {t_row * 1e3:7.1f} us {fl / t_row:4.0f} TF | best {bc} gm{bgm} {bt * 1e3:7.1f} us "
                  f"{fl / bt:4.0f} TF ({t_row / bt:.3f}x) | " + ", ".join(f"{c}/{gg} {fl / t:.0f}" for t, c, gg, _ in res[1:5]),
                  flush=True)
            tot_ref += LAYERS[name] * t_row
            tot_best += LAYERS[name] * min(bt, t_row)
        print(f"M {M}: per-step GEMM ref {tot_ref:.1f} ms -> {tot_best:.1f} ms ({tot_ref / tot_best:.3f}x)", flush=True)
    print("NOT byte-identical:", bad if bad else "none", flush=True)
    if out:
        json.dump(table, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
