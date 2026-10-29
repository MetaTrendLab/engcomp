#!/usr/bin/env python3
import sys

import torch

sys.path.insert(0, "/opt/reax/engine")
from f8cubin import F8Gemm
from f8g6 import F8G6

M = 3040
CFGS = ["f8_256x128x64_w4x2_s4", "f8_128x256x128_w2x4_s2", "f8_128x128x64_w2x2_s3_b2"]
g = F8Gemm()
g6 = F8G6()
G6 = {"gu": ["g7_256x128x64_w4x2_s4_b1", "g7_128x256x128_w2x4_s2_b1"], "down": ["g7p_224x128x128_w2x4_s2_b1", "g7_224x128x128_w2x4_s2_b1"]}
gen = torch.Generator(device="cuda").manual_seed(5)
for name, (N, K) in {"gu": (34816, 5120), "down": (5120, 17408)}.items():
    a = (torch.randn(M, K, device="cuda", generator=gen) * 2).to(torch.float8_e4m3fn)
    b = (torch.randn(N, K, device="cuda", generator=gen) * 2).to(torch.float8_e4m3fn)
    sa = torch.rand(M, 1, device="cuda", generator=gen) + 0.5
    sb = torch.rand(1, N, device="cuda", generator=gen) + 0.5
    one = torch.ones((), device="cuda")
    fns = [lambda: torch._scaled_mm(a, b.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16),
           lambda: torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)]
    fns += [lambda c=c: g(a, b, sa.view(-1), sb.view(-1), cfg=c, gm=8) for c in CFGS]
    fns += [lambda c=c: g6(a, b, sa.view(-1), sb.view(-1), cfg=c, gm=16) for c in G6[name]]
    for f in fns:
        f()
        f()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(name)
    for f in fns:
        f()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    print(name, "done", flush=True)
