#!/usr/bin/env python3
import os
import sys

import torch

sys.path.insert(0, "/opt/reax/engine")
import fp8_fused
from f8g6 import F8G6
from pplx_engine import silu_mul_q_scaled_triton

I, K = 17408, 5120


LAUNCHES = int(os.environ.get("F8G6_TAB_LAUNCHES", "4"))

def t_ms(fn, reps=11):
    for _ in range(2):
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
    args = sys.argv[1:]
    table = None
    if args and args[0] == "--table":
        table, M0, M1, S = args[1], int(args[2]), int(args[3]), int(args[4])
        Ms = list(range(M0, M1 + 1, S))
    else:
        Ms = [int(a) for a in args] or [1024, 2048, 3040]
    picks = []
    import os
    if os.environ.get("F8G6_TAB_SK") == "1":
        from f8sk import G6SK
        g = G6SK()
    else:
        g = F8G6()
    cfgs = [c for c in g.fn if c.startswith("g7gu_") or c.startswith("s7gu_")]
    gen = torch.Generator(device="cuda").manual_seed(9)
    w8 = (torch.randn(2 * I, K, device="cuda", generator=gen) * 2).to(torch.float8_e4m3fn)
    ws = (torch.rand(1, 2 * I, device="cuda", generator=gen) * 0.02 + 0.01)
    idx = torch.stack([torch.arange(I), torch.arange(I) + I], 1).reshape(-1).cuda()
    w8i = w8[idx].contiguous().view(torch.uint8)
    wsi = ws.view(-1)[idx].contiguous()
    one = torch.ones((), device="cuda")
    for M in Ms:
        x8 = (torch.randn(M, K, device="cuda", generator=gen) * 2).to(torch.float8_e4m3fn)
        xs = torch.rand(M, 1, device="cuda", generator=gen) * 0.02 + 0.01

        def ref():
            gu = torch._scaled_mm(x8, w8.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
            return silu_mul_q_scaled_triton(gu, xs, ws, I)

        q0, s0 = ref()
        t_ref = t_ms(ref)
        res = []
        for c in cfgs:
            def ours(c=c):
                y, am = g.gu(x8.view(torch.uint8), w8i, xs.view(-1), wsi, cfg=c, gm=16)
                return fp8_fused.quant_rows(y, am)
            q1, s1 = ours()
            same = torch.equal(q0.view(torch.uint8), q1.view(torch.uint8)) and torch.equal(s0, s1)
            res.append((t_ms(ours), c, same))
        res.sort()
        print(f"M {M:>5}: ref (cuBLAS + _smqs) {t_ref * 1e3:7.1f} us | " + ", ".join(
            f"{c} {t * 1e3:.1f} us {t_ref / t:.3f}x {'same' if ok else 'DIFF'}" for t, c, ok in res[:4]), flush=True)
        bad = [c for _, c, ok in res if not ok]
        print(f"   not byte-identical: {bad if bad else 'none'}", flush=True)
        good = [r for r in res if r[2]]
        if not good:
            print(f"   M {M}: no byte-identical config, left out of the table", flush=True)
            continue
        picks.append((M, good[0][1], t_ref / good[0][0]))
    if table:
        import json
        iv = []
        for M, c, sp in picks:
            lo = M - S + 1 if M > M0 else 1
            if iv and iv[-1][2] == c and iv[-1][1] == lo - 1:
                iv[-1][1] = M
                iv[-1][4] = round(min(iv[-1][4], sp), 3)
            else:
                iv.append([lo, M, c, 16, round(sp, 3)])
        iv[-1][1] = 1 << 20
        json.dump({"gu": iv}, open(table, "w"), indent=1)
        sps = sorted(sp for _, _, sp in picks)
        print(f"gu table: {len(iv)} intervals, speed-up min/median/max {sps[0]:.3f}/{sps[len(sps) // 2]:.3f}/{sps[-1]:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
