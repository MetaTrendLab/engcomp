#!/usr/bin/env python3
import json
import sys

import torch

sys.path.insert(0, "/opt/reax/engine")
from f8g6 import F8G6

import os

SHAPES = [(5120, 6144), (5120, 17408), (16384, 5120), (14336, 5120)]
TENSOR = os.environ.get("F8G6_TAB_GU") == "1"
if TENSOR:
    SHAPES = [(34816, 5120)]
CANDS = ["g6_224x128x128_w2x4_s2_b1", "g6p_224x128x128_w2x4_s2_b1", "g6_224x128x64_w2x4_s4_b1",
         "g6_192x128x128_w4x2_s2_b1", "g6p_192x128x128_w4x2_s2_b1", "g6_192x128x64_w4x2_s4_b1",
         "g6_160x128x128_w2x4_s2_b1", "g6p_112x128x64_w1x4_s3_b2", "g6_112x128x64_w1x4_s3_b2",
         "g6_96x128x64_w2x2_s3_b2", "g6p_112x256x128_w1x8_s2_b1", "g6_128x128x64_w2x2_s3_b2",
         "g6_128x256x128_w2x4_s2_b1", "g6_256x128x128_w4x2_s2_b1", "g6_96x256x64_w2x4_s4_b1"]
CANDS += [c.replace("g6", g, 1) for g in ("g7", "g8") for c in CANDS]
SK = os.environ.get("F8G6_TAB_SK") == "1"
if SK:
    CANDS += [c.replace("g6", s, 1).replace("p_", "_") for s in ("s7", "s8") for c in CANDS if c.startswith("g6")]
    CANDS = list(dict.fromkeys(CANDS))
MIN_GAIN = 1.01


LAUNCHES = int(os.environ.get("F8G6_TAB_LAUNCHES", "4"))

def t_ms(fn, reps=9):
    fn()
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(LAUNCHES):
            fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) / LAUNCHES)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    out = sys.argv[1]
    M0, M1, S = (int(x) for x in sys.argv[2:5]) if len(sys.argv) > 4 else (64, 4096, 32)
    if SK:
        from f8sk import G6SK
        g = G6SK()
    else:
        g = F8G6()
    cands = [c for c in CANDS if c in g.fn]
    gen = torch.Generator(device="cuda").manual_seed(7)
    table = {}
    for N, K in SHAPES:
        b = (torch.randn(N, K, device="cuda", generator=gen) * 2).to(torch.float8_e4m3fn)
        sb = torch.rand(1, N, device="cuda", generator=gen) + 0.5
        a_full = (torch.randn(M1, K, device="cuda", generator=gen) * 2).to(torch.float8_e4m3fn)
        sa_full = torch.rand(M1, 1, device="cuda", generator=gen) + 0.5
        picks = []
        for M in range(M0, M1 + 1, S):
            a, sa = a_full[:M], sa_full[:M].contiguous()
            if TENSOR:
                one = torch.ones((), device="cuda")
                row = lambda: torch._scaled_mm(a, b.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
                ka, kb = None, None
            else:
                row = lambda: torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
                ka, kb = sa.view(-1), sb.view(-1)
            t_ref = t_ms(row)
            best = (t_ref, None, 0)
            for cfg in cands:
                if not g.fits(cfg, N, K):
                    continue
                for gm in (8, 16):
                    t = t_ms(lambda: g(a, b, ka, kb, cfg=cfg, gm=gm))
                    if t < best[0]:
                        best = (t, cfg, gm)
            sp = t_ref / best[0]
            pick = None
            if best[1] and sp >= MIN_GAIN:
                if torch.equal(g(a, b, ka, kb, cfg=best[1], gm=best[2]), row()):
                    pick = (best[1], best[2])
                else:
                    print(f"  NOT byte-identical: {N}x{K} M {M} {best[1]}", flush=True)
            picks.append((M, pick, sp))
        iv = []
        for M, pick, sp in picks:
            lo = M - S + 1 if M > M0 else 1
            if pick and iv and iv[-1][2:4] == list(pick) and iv[-1][1] == lo - 1:
                iv[-1][1] = M
                iv[-1][4] = round(min(iv[-1][4], sp), 3)
            elif pick:
                iv.append([lo, M, pick[0], pick[1], round(sp, 3)])
        table[f"{-N if TENSOR else N}:{K}"] = iv
        won = [sp for _, p, sp in picks if p]
        print(f"{N}x{K}: ours at {len(won)}/{len(picks)} Ms, speed-up min/median/max "
              f"{min(won, default=1):.3f}/{sorted(won)[len(won) // 2] if won else 1:.3f}/{max(won, default=1):.3f}; "
              f"{len(iv)} intervals", flush=True)
        for M, pick, sp in picks[::8]:
            print(f"   M {M:>5}: {pick} {sp:.3f}", flush=True)
    json.dump(table, open(out, "w"), indent=1)
    print("written", out, flush=True)


if __name__ == "__main__":
    main()
