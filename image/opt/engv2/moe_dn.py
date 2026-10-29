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


def _reax_dn_case(m, g, kind, n=2816, k=704, e=128, topk=8):
    """Inputs of the down GEMM as run_cutlass_moe_fp4 builds them: m tokens routed top-k over e experts (as moe_tile)."""
    if kind == 0:
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


_REAX_DN_TUNE_M = (440, 887, 1000)  # a step's token counts (image steps ~440, c16 text steps 645-1014)


def _reax_dn_tune(exact, verbose=False):
    """Time every exact variant at the tuning m (best of 2); returns (best variant, its times, vLLM's times)."""
    names = _reax_dn_variants()
    t_ref = [min(_reax_dn_time(False, m, rep=100) for _ in range(2)) for m in _REAX_DN_TUNE_M]
    best, best_t = None, None
    for v in exact:
        tv = [min(_reax_dn_time(True, m, v, rep=100) for _ in range(2)) for m in _REAX_DN_TUNE_M]
        if verbose:
            print(f"  {names[v]:14s} " + "  ".join(f"m {m} {t * 1e3:6.1f} us ({100 * (t / r - 1):+5.1f} %)"
                                                 for m, t, r in zip(_REAX_DN_TUNE_M, tv, t_ref)), flush=True)
        if best_t is None or sum(tv) < sum(best_t):
            best, best_t = v, tv
    return best, best_t, t_ref
'''

if len(sys.argv) > 1 and sys.argv[1] in ("test", "derive"):
    import torch
    from vllm import _custom_ops as ops
    ns = {"torch": torch, "ops": ops}
    exec(compile(ADD, "moe_dn.py:ADD", "exec"), ns)
    if sys.argv[1] == "derive":
        on, var = 0, 0
        try:
            exact = ns["_reax_dn_check"]()
            if exact:
                var, tv, tr = ns["_reax_dn_tune"](exact)
                k = ns["_REAX_DN_TUNE_M"].index(887)
                on = int(tv[k] < tr[k] / 1.02)
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
