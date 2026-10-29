import sys

OLD = '''    c3 = ops.shuffle_rows(c3, c_map)

    assert output.dtype == out_dtype
    if not apply_router_weight_on_input:
'''
NEW = '''    if not apply_router_weight_on_input and _reax_unperm(output, c3, c_map, topk_weights, m, num_topk, k):
        return
    c3 = ops.shuffle_rows(c3, c_map)

    assert output.dtype == out_dtype
    if not apply_router_weight_on_input:
'''
ADD = '''

# ---------------------------------------------------------------- engv2: fused unpermute + topk-weighted sum (moe_unperm.py)
import os as _reax_os

import triton as _reax_triton
import triton.language as _reax_tl


@_reax_triton.jit
def _reax_unperm_p(c3, cmap, w, t, j, K, cols, mask):
    """bf16(float(c3[c_map[t * 8 + j], cols]) * float(bf16(w[t, j]))) widened to fp32 (ATen's bf16 mul)."""
    src = _reax_tl.load(cmap + t * 8 + j).to(_reax_tl.int64)
    x = _reax_tl.load(c3 + src * K + cols, mask=mask, other=0.0).to(_reax_tl.float32)
    wj = _reax_tl.load(w + t * 8 + j).to(_reax_tl.bfloat16).to(_reax_tl.float32)
    return (x * wj).to(_reax_tl.bfloat16).to(_reax_tl.float32)


@_reax_triton.jit
def _reax_unperm_kernel(c3, cmap, w, out, K, BLOCK: _reax_tl.constexpr, ORDER: _reax_tl.constexpr):
    t = _reax_tl.program_id(0)
    cols = _reax_tl.program_id(1) * BLOCK + _reax_tl.arange(0, BLOCK)
    mask = cols < K
    p0 = _reax_unperm_p(c3, cmap, w, t, 0, K, cols, mask)
    p1 = _reax_unperm_p(c3, cmap, w, t, 1, K, cols, mask)
    p2 = _reax_unperm_p(c3, cmap, w, t, 2, K, cols, mask)
    p3 = _reax_unperm_p(c3, cmap, w, t, 3, K, cols, mask)
    p4 = _reax_unperm_p(c3, cmap, w, t, 4, K, cols, mask)
    p5 = _reax_unperm_p(c3, cmap, w, t, 5, K, cols, mask)
    p6 = _reax_unperm_p(c3, cmap, w, t, 6, K, cols, mask)
    p7 = _reax_unperm_p(c3, cmap, w, t, 7, K, cols, mask)
    z = _reax_tl.zeros([BLOCK], dtype=_reax_tl.float32)  # ATen starts every accumulator at the identity 0.0
    if ORDER == 0:  # one accumulator, j = 0 .. 7
        acc = (((((((z + p0) + p1) + p2) + p3) + p4) + p5) + p6) + p7
    elif ORDER == 1:  # 4 accumulators (j mod 4), combined ((a0 + a1) + a2) + a3 (ATen thread_reduce, vt0 = 4)
        acc = ((((z + p0) + p4) + ((z + p1) + p5)) + ((z + p2) + p6)) + ((z + p3) + p7)
    elif ORDER == 2:  # 4 accumulators combined (a0 + a1) + (a2 + a3)
        acc = (((z + p0) + p4) + ((z + p1) + p5)) + (((z + p2) + p6) + ((z + p3) + p7))
    elif ORDER == 3:  # block.y = 2: thread y sums j = y, y+2, y+4, y+6 (one accumulator each), then y0 + y1
        acc = ((((z + p0) + p2) + p4) + p6) + ((((z + p1) + p3) + p5) + p7)
    elif ORDER == 4:  # block.y = 4 or 8: (j, j+4) per thread, shared-memory tree (y0 + y2) + (y1 + y3)
        acc = (((z + p0) + p4) + ((z + p2) + p6)) + (((z + p1) + p5) + ((z + p3) + p7))
    else:  # pairwise tree
        acc = (((z + p0) + p1) + ((z + p2) + p3)) + (((z + p4) + p5) + ((z + p6) + p7))
    _reax_tl.store(out + t * K + cols, acc.to(_reax_tl.bfloat16), mask=mask)


_REAX_UNPERM = {"order": None}


def _reax_unperm_launch(output, c3, c_map, topk_weights, m, topk, k, order):
    BLOCK = 512
    _reax_unperm_kernel[(m, _reax_triton.cdiv(k, BLOCK))](c3, c_map, topk_weights, output, k, BLOCK=BLOCK, ORDER=order,
                                                          num_warps=4)


def _reax_unperm_eager(c3, c_map, topk_weights, m, topk, k):
    c3s = ops.shuffle_rows(c3, c_map)
    return (c3s.view(m, topk, k) * topk_weights.view(m, topk, 1).to(c3.dtype)).sum(dim=1)


def _reax_unperm_data(m, topk, k, g):
    """c3 with exponents spread over 2^-24 .. 2^24 (fp32 sums of the products are then rarely exact, so any change of
    summation order shows), a random row permutation and normalised routing weights."""
    e = torch.randint(-24, 25, (m * topk, k), device="cuda", generator=g, dtype=torch.int16).to(torch.bfloat16)
    c3 = torch.randn(m * topk, k, device="cuda", generator=g, dtype=torch.bfloat16) * torch.exp2(e)  # exact in bf16
    del e
    perm = torch.randperm(m * topk, device="cuda", generator=g).to(torch.int32)
    w = torch.rand(m, topk, device="cuda", generator=g).float()
    return c3, perm, w / w.sum(-1, keepdim=True)


_REAX_UNPERM_M = (1, 2, 3, 5, 8, 13, 16, 31, 64, 100, 128, 255, 256, 512, 1000, 1024, 2048, 4096, 8192)


def _reax_unperm_check(topk, k, dtype):
    """First order that reproduces the eager ops bit for bit (sign of zero included) at every m probed."""
    g = torch.Generator(device="cuda").manual_seed(5150)
    for order in (1, 0, 2, 3, 4, 5):
        ok = True
        for m in _REAX_UNPERM_M:
            c3, perm, w = _reax_unperm_data(m, topk, k, g)
            ref = _reax_unperm_eager(c3, perm, w, m, topk, k)
            out = torch.empty(m, k, device="cuda", dtype=dtype)
            _reax_unperm_launch(out, c3, perm, w, m, topk, k, order)
            if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):  # bit patterns: -0.0 != +0.0
                ok = False
                break
        if ok:
            return order
    return -1


def _reax_unperm(output, c3, c_map, topk_weights, m, topk, k):
    if _reax_os.environ.get("REAX_MOE_UNPERM", "1") != "1" or c3.dtype != torch.bfloat16 or topk != 8:
        return False
    if not (c3.is_contiguous() and c_map.is_contiguous() and topk_weights.is_contiguous() and output.is_contiguous()):
        return False
    st = _REAX_UNPERM
    if st["order"] is None:
        if torch.cuda.is_current_stream_capturing():
            return False
        path = _reax_os.environ.get("REAX_MOE_UNPERM_FILE", "")
        if path and _reax_os.path.exists(path):
            st["order"] = int(open(path).read().split()[0])
            how = f"loaded from {path}"
        else:
            st["order"] = _reax_unperm_check(topk, k, c3.dtype)
            how = "checked in-engine"
        print(f"reax moe_unperm: order {st['order']} {how} "
              f"({'bit-exact vs shuffle_rows + mul + sum' if st['order'] >= 0 else 'no order matches: eager path'})", flush=True)
    if st["order"] < 0:
        return False
    _reax_unperm_launch(output, c3[: m * topk], c_map, topk_weights, m, topk, k, st["order"])
    return True
'''

if sys.argv[1] == "test":
    import torch
    from vllm.model_executor.layers.fused_moe.experts import cutlass_moe as cm
    g = torch.Generator(device="cuda").manual_seed(7)
    for m in cm._REAX_UNPERM_M:
        c3, perm, w = cm._reax_unperm_data(m, 8, 2816, g)
        ref = cm._reax_unperm_eager(c3, perm, w, m, 8, 2816)
        bad = []
        for order in range(6):
            out = torch.empty(m, 2816, device="cuda", dtype=torch.bfloat16)
            cm._reax_unperm_launch(out, c3, perm, w, m, 8, 2816, order)
            bad.append(int((out.view(torch.int16) != ref.view(torch.int16)).sum()))
        print(f"m {m:5d}: mismatching elements per order 0..5 {bad} of {m * 2816}", flush=True)
    print("check picks order", cm._reax_unperm_check(8, 2816, torch.bfloat16))
    from triton.testing import do_bench
    for m in (8, 64, 256, 512, 1024, 2048):
        c3, perm, w = cm._reax_unperm_data(m, 8, 2816, g)
        out = torch.empty(m, 2816, device="cuda", dtype=torch.bfloat16)
        te = do_bench(lambda: out.copy_(cm._reax_unperm_eager(c3, perm, w, m, 8, 2816)))
        tf = do_bench(lambda: cm._reax_unperm_launch(out, c3, perm, w, m, 8, 2816, 1))
        print(f"m {m:5d}: eager {te * 1e3:7.1f} us  fused {tf * 1e3:7.1f} us  x{te / tf:.2f}", flush=True)
    sys.exit(0)
if sys.argv[1] == "derive":
    import torch
    from vllm.model_executor.layers.fused_moe.experts import cutlass_moe as cm
    order = cm._reax_unperm_check(8, 2816, torch.bfloat16)
    open(sys.argv[2], "w").write(f"{order}\n")
    print(f"reax moe_unperm: derived order {order}", flush=True)
    sys.exit(0)
path = sys.argv[1]
src = open(path).read()
i = src.index("def run_cutlass_moe_fp4(")
j = src.index(OLD, i)
assert j < src.index("\ndef ", i + 1), "run_cutlass_moe_fp4 tail not found"
src = src[:j] + NEW + src[j + len(OLD):] + ADD
open(path, "w").write(src)
print("moe_unperm: installed")
