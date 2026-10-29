import sys

OLD = '''    ops.cutlass_fp4_moe_mm(
        c3,
'''
ADD = '''

# ---------------------------------------------------------------- engv2: MoE down GEMM through moe_dn.cu (moe_dn.py)
import ctypes as _reax_dn_ctypes
import os as _reax_dn_os

_REAX_DN = {"on": None, "var": 0, "lib": None}


def _reax_dn_lib():
    st = _REAX_DN
    if st["lib"] is None:
        c = _reax_dn_ctypes
        lib = (c.PyDLL if _reax_dn_os.environ.get("REAX_PYDLL", "0") == "1" else c.CDLL)(_reax_dn_os.environ.get("REAX_MOE_DN_LIB", "/opt/engv2/libreax_moedn.so"))
        lib.reax_moedn_mm_v.restype = c.c_int
        lib.reax_moedn_mm_v.argtypes = [c.c_int] + [c.c_void_p] * 9 + [c.c_int] * 3 + [c.c_void_p]
        lib.reax_moedn_variants.restype = c.c_int
        lib.reax_moedn_variant_name.restype = c.c_char_p
        lib.reax_moedn_variant_name.argtypes = [c.c_int]
        st["lib"] = lib
    return st["lib"]


def _reax_dn_variants():
    lib = _reax_dn_lib()
    return [lib.reax_moedn_variant_name(i).decode() for i in range(lib.reax_moedn_variants())]


def _reax_dn_raw(out, a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets, var=None):
    rc = _reax_dn_lib().reax_moedn_mm_v(
        _REAX_DN["var"] if var is None else var,
        out.data_ptr(), a.data_ptr(), b.data_ptr(), a_sf.data_ptr(), b_sf.data_ptr(), alphas.data_ptr(),
        problem_sizes.data_ptr(), expert_offsets.data_ptr(), sf_offsets.data_ptr(), b.shape[0], b.shape[1],
        b.shape[2] * 2, torch.cuda.current_stream().cuda_stream)
    if rc != 0:
        raise RuntimeError(f"reax_moedn_mm: error {rc}")


def _reax_moe_dn(out, a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets):
    """ops.cutlass_fp4_moe_mm for the down projection; moe_dn.cu when the start-up check said so."""
    st = _REAX_DN
    if st["on"] is None and not torch.cuda.is_current_stream_capturing():
        st["on"] = False
        path = _reax_dn_os.environ.get("REAX_MOE_DN_FILE", "")
        if _reax_dn_os.environ.get("REAX_MOE_DN", "1") == "1" and path and _reax_dn_os.path.exists(path):
            flags = open(path).read().split()
            st["on"] = flags[:1] == ["1"]
            st["var"] = int(flags[1]) if len(flags) > 1 else 0
        print(f"reax moe_dn: {'on' if st['on'] else 'off'}, variant {st['var']} (check file {path or '-'})", flush=True)
    # up to 1280 tokens (10240 rows): a c16 step is <= ~1014, an image step ~400; at m 1760 the kernel is +4.8 % on the
    # PRO 6000 (-3 % on the 5090: more rows per expert -> more 64-row chunks, the weight stream no longer dominates)
    if (st["on"] and a.shape[0] <= 10240 and b.dim() == 3 and b.shape[1:] == (2816, 352) and out.dtype == torch.bfloat16
            and a.is_contiguous() and b.is_contiguous() and a_sf.is_contiguous() and b_sf.is_contiguous()
            and out.is_contiguous() and alphas.dtype == torch.float32 and problem_sizes.shape[0] == b.shape[0]):
        return _reax_dn_raw(out, a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets)
    return ops.cutlass_fp4_moe_mm(out, a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets)


# rows per expert of four Blink text MoE calls (REAX_MOE_GU_DUMP, c16 bench load: moe_gu.py REAL + two more; m 965 / 865
# / 919 / 808, largest expert 526 / 730 / 614 / 546 rows): the tuning routings (uniform routing hides the
# hot-expert tail the _s<SP> variants remove)
_REAX_DN_REAL = ([168,0,1,0,28,170,2,1,49,10,78,41,8,98,35,3,7,13,26,3,64,5,56,70,65,49,0,17,62,30,9,91,37,105,57,4,101,311,151,1,119,0,82,0,48,29,33,56,0,9,16,458,2,22,1,20,14,452,17,16,1,69,68,30,20,80,526,10,158,180,29,6,347,0,192,14,26,15,72,93,25,0,10,12,0,18,40,9,456,15,15,0,0,13,0,7,2,0,26,3,13,37,0,26,133,69,0,48,1,136,238,33,94,0,5,220,74,31,111,15,0,21,23,101,3,13,260,8],
        [3,25,13,59,4,26,77,4,133,0,68,2,10,20,9,0,0,3,168,2,16,18,36,16,0,51,215,5,4,13,3,44,14,221,11,4,21,11,159,46,730,0,5,30,152,1,158,22,3,2,4,16,2,3,7,95,405,3,3,177,10,6,1,0,3,5,55,19,3,4,8,79,23,4,1,34,0,1,56,4,4,112,3,0,9,21,190,2,0,46,0,0,60,72,19,28,16,340,1,2,0,55,8,66,2,18,12,9,1,49,29,577,0,47,1,240,0,13,14,291,0,520,21,77,263,0,0,14],
                 [121,132,100,78,473,0,8,34,29,118,3,0,9,63,24,22,14,191,13,33,22,1,5,20,4,3,0,0,53,27,20,48,76,6,56,6,183,26,0,2,88,0,11,11,1,74,0,1,141,614,11,12,3,16,0,2,35,31,32,58,36,13,334,116,269,14,0,2,38,67,9,3,75,61,0,4,1,139,32,41,0,0,14,13,441,11,24,42,37,41,365,46,14,2,8,0,31,0,7,22,25,2,77,15,6,40,0,587,0,21,0,43,2,48,2,456,77,46,54,8,0,182,22,3,14,57,0,29],
                 [146,119,72,58,422,0,7,38,19,125,2,0,6,52,23,20,10,169,15,23,16,3,6,25,3,7,0,0,52,26,17,43,42,7,52,7,151,16,0,2,81,0,1,17,2,51,0,0,115,546,11,17,2,9,0,1,27,28,23,46,37,12,297,107,255,5,0,1,35,60,9,1,80,46,0,2,1,132,30,50,0,0,8,11,351,9,13,39,35,27,306,36,15,4,6,0,33,1,10,19,31,0,88,10,9,29,2,500,0,13,0,23,0,35,4,390,68,32,66,13,0,144,20,8,23,50,0,45])


def _reax_dn_case(m, g, kind, n=2816, k=704, e=128, topk=8, counts=None):
    """Inputs of the down GEMM as run_cutlass_moe_fp4 builds them: m tokens routed top-k over e experts (as moe_tile);
    counts (rows per expert, sum m x topk): the sorted expert list dealt round-robin over the tokens."""
    if counts is not None:
        m = int(sum(counts)) // topk
        ids = torch.repeat_interleave(torch.arange(e, device="cuda"), torch.tensor(counts, device="cuda").long())
        topk_ids = ids.view(topk, m).t().contiguous().int()
    elif kind == 0:
        topk_ids = torch.stack([torch.randperm(e, device="cuda", generator=g)[:topk] for _ in range(m)]).int()
    else:  # skewed: a few hot experts
        w = torch.ones(e, device="cuda")
        w[:8] = 40.0 if kind == 1 else 400.0
        topk_ids = torch.stack([torch.multinomial(w, topk, replacement=False, generator=g) for _ in range(m)]).int()
    eo = torch.empty(e + 1, dtype=torch.int32, device="cuda")
    bo = torch.empty(e + 1, dtype=torch.int32, device="cuda")
    p1 = torch.empty((e, 3), dtype=torch.int32, device="cuda")
    p2 = torch.empty((e, 3), dtype=torch.int32, device="cuda")
    am = torch.empty(m * topk, dtype=torch.int32, device="cuda")
    cm_ = torch.empty(m * topk, dtype=torch.int32, device="cuda")
    ops.get_cutlass_moe_mm_data(topk_ids, eo, p1, p2, am, cm_, e, k, n, bo, is_gated=True)
    a = torch.randint(0, 256, (m * topk, k // 2), dtype=torch.uint8, device="cuda", generator=g)
    b = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, device="cuda", generator=g)
    srows = int(bo[-1].item()) + 128
    # E4M3 scales without NaN (0x7f): positive, wide range
    a_sf = torch.randint(0x18, 0x50, (srows, k // 16), dtype=torch.uint8, device="cuda", generator=g)
    b_sf = torch.randint(0x18, 0x50, (e, n, k // 16), dtype=torch.uint8, device="cuda", generator=g)
    alphas = torch.exp2(torch.randint(-14, -4, (e,), device="cuda", generator=g).float()) * (
        1.0 + torch.rand(e, device="cuda", generator=g))
    return (a, b, a_sf.view(torch.float8_e4m3fn), b_sf.view(torch.float8_e4m3fn), alphas.float(), p2, eo[:-1], bo[:-1])


def _reax_dn_check(small=False, variants=None):
    """Byte-for-byte vs ops.cutlass_fp4_moe_mm (both outputs pre-filled with the same pattern) at many m and routings,
    for every variant; returns the exact ones."""
    g = torch.Generator(device="cuda").manual_seed(4242)
    ms = (1, 3, 16, 200) if small else (1, 2, 3, 7, 8, 16, 31, 64, 100, 128, 200, 333, 512, 887, 1000, 1024, 1500, 2048)
    names = _reax_dn_variants()
    exact = set(range(len(names)) if variants is None else variants)
    for m in ms:
        for kind in (0, 1, 2):
            a, b, a_sf, b_sf, alphas, ps, eo, bo = _reax_dn_case(m, g, kind)
            ref = torch.full((m * 8, 2816), -7.0, dtype=torch.bfloat16, device="cuda")
            ops.cutlass_fp4_moe_mm(ref, a, b, a_sf, b_sf, alphas, ps, eo, bo)
            for v in sorted(exact):
                out = torch.full_like(ref, -7.0)
                _reax_dn_raw(out, a, b, a_sf, b_sf, alphas, ps, eo, bo, var=v)
                torch.cuda.synchronize()
                bad = out.view(torch.int16) != ref.view(torch.int16)
                if bool(bad.any()):
                    exact.discard(v)
                    print(f"reax moe_dn: MISMATCH {names[v]} m {m} kind {kind}: {int(bad.sum())} of {out.numel()} "
                          f"values, first at {tuple(int(x) for x in bad.nonzero()[0])}", flush=True)
    return sorted(exact)


def _reax_dn_time(mine, m=887, var=None, rep=300):
    from triton.testing import do_bench
    g = torch.Generator(device="cuda").manual_seed(7)
    a, b, a_sf, b_sf, alphas, ps, eo, bo = _reax_dn_case(m, g, 0)
    out = torch.empty((m * 8, 2816), dtype=torch.bfloat16, device="cuda")
    if mine:
        return do_bench(lambda: _reax_dn_raw(out, a, b, a_sf, b_sf, alphas, ps, eo, bo, var=var), warmup=25, rep=rep)
    return do_bench(lambda: ops.cutlass_fp4_moe_mm(out, a, b, a_sf, b_sf, alphas, ps, eo, bo), warmup=25, rep=rep)


_REAX_DN_TUNE_M = ("965 real", "865 real", "919 real", "808 real")  # _REAX_DN_REAL


def _reax_dn_rot(fn, ws, reps=15):
    """median ms per call over rounds of 2 x len(ws) back-to-back calls, one weight set per call in rotation (as the
    engine runs the layers: no L2 flush, whose dirty lines slow vLLM's op ~40 % against its in-engine time)"""
    k = 2 * len(ws)
    for i in range(k):
        fn(ws[i % len(ws)])
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for i in range(k):
            fn(ws[i % len(ws)])
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) / k)
    ts.sort()
    return ts[len(ts) // 2]


def _reax_dn_tune(exact, verbose=False):
    """Time every exact variant and vLLM's op on the _REAX_DN_REAL routings (4 weight sets in rotation); returns (best
    variant by summed time, its times, vLLM's times)."""
    names = _reax_dn_variants()
    g = torch.Generator(device="cuda").manual_seed(7)
    cases = [_reax_dn_case(0, g, 0, counts=c) for c in _REAX_DN_REAL]
    ws = [(c[1], c[3], c[4]) for c in cases]
    ws = ws[:4]
    while len(ws) < 4:  # more weight sets (b, b_sf, alphas) of the same shapes
        ws.append((torch.randint(0, 256, ws[0][0].shape, dtype=torch.uint8, device="cuda", generator=g),
                   torch.randint(0x18, 0x50, ws[0][1].shape, dtype=torch.uint8, device="cuda",
                                 generator=g).view(torch.float8_e4m3fn), ws[0][2]))
    outs = [torch.empty((c[0].shape[0], 2816), dtype=torch.bfloat16, device="cuda") for c in cases]
    t_ref = [_reax_dn_rot(lambda w, c=c, o=o: ops.cutlass_fp4_moe_mm(o, c[0], w[0], c[2], w[1], w[2], c[5], c[6], c[7]),
                          ws) for c, o in zip(cases, outs)]
    best, best_t = None, None
    for v in exact:
        tv = [_reax_dn_rot(lambda w, c=c, o=o: _reax_dn_raw(o, c[0], w[0], c[2], w[1], w[2], c[5], c[6], c[7], var=v),
                           ws) for c, o in zip(cases, outs)]
        if verbose:
            print(f"  {names[v]:16s} " + "  ".join(f"m {m} {t * 1e3:6.1f} us ({100 * (t / r - 1):+5.1f} %)"
                                                 for m, t, r in zip(_REAX_DN_TUNE_M, tv, t_ref)), flush=True)
        if best_t is None or sum(tv) < sum(best_t):
            best, best_t = v, tv
    return best, best_t, t_ref
'''

if len(sys.argv) > 1 and sys.argv[1] in ("test", "derive"):
    import os
    import torch
    from vllm import _custom_ops as ops
    ns = {"torch": torch, "ops": ops}
    exec(compile(ADD, "moe_dn.py:ADD", "exec"), ns)
    if sys.argv[1] == "derive":
        on, var = 0, 0
        try:
            only = [x for x in os.environ.get("REAX_MOE_DN_ONLY", "").split(",") if x]
            names = ns["_reax_dn_variants"]()
            cand = [names.index(x) for x in only if x in names] if only else list(range(len(names)))
            if os.environ.get("REAX_MOE_DN_SPLIT", "1") == "0":
                cand = [v for v in cand if "_s" not in names[v]]
            exact = ns["_reax_dn_check"](variants=cand)
            if exact:
                var, tv, tr = ns["_reax_dn_tune"](exact)
                on = int(all(t < r / 1.02 for t, r in zip(tv, tr)))
                print(f"reax moe_dn: {len(exact)} variants byte-identical, best {ns['_reax_dn_variants']()[var]}: "
                      + ", ".join(f"m {m} {t * 1e3:.1f} us vs vLLM {r * 1e3:.1f}"
                                  for m, t, r in zip(ns["_REAX_DN_TUNE_M"], tv, tr)), flush=True)
        except Exception as e:
            print(f"reax moe_dn: check failed ({e!r})", flush=True)
        open(sys.argv[2], "w").write(f"{on} {var}\n")
        print("reax moe_dn:", f"on, variant {var}" if on else "vLLM's op", flush=True)
        sys.exit(0)
    names = ns["_reax_dn_variants"]()
    exact = ns["_reax_dn_check"](small=len(sys.argv) > 2)
    print(f"moe_dn: {len(exact)} of {len(names)} variants byte-identical at every case", flush=True)
    var, tv, tr = ns["_reax_dn_tune"](exact, verbose=True)
    print(f"best {names[var]}", flush=True)
    for m in (110, 440, 887, 1000, 1760):
        t_ref, t0, tb = ns["_reax_dn_time"](False, m), ns["_reax_dn_time"](True, m, 0), ns["_reax_dn_time"](True, m, var)
        mb = (128 * 2816 * (352 + 44) + m * 8 * (352 + 2816 * 2)) / 1e6
        print(f"m {m:5d}: vLLM {t_ref * 1e3:6.1f} us ({mb / t_ref / 1e3:5.2f} TB/s) | v0 {t0 * 1e3:6.1f} us | "
              f"{names[var]} {tb * 1e3:6.1f} us ({mb / tb / 1e3:5.2f} TB/s), {100 * (tb / t_ref - 1):+.1f} %", flush=True)
    sys.exit(0)
path = sys.argv[1]
src = open(path).read()
assert src.count(OLD) == 1, "down grouped GEMM call site not found"
src = src.replace(OLD, OLD.replace("ops.cutlass_fp4_moe_mm(", "_reax_moe_dn(")) + ADD
open(path, "w").write(src)
print("moe_dn: installed")
