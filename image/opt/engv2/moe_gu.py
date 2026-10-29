import ctypes as C
import os
import sys

OLD = '''    ops.cutlass_fp4_moe_mm(
        c1,
        rep_a_fp4,
        w1_fp4,
        rep_a_blockscale,
        w1_blockscale,
        w1_alphas,
        problem_sizes1,
        expert_offsets[:-1],
        blockscale_offsets[:-1],
    )
    del rep_a_fp4, rep_a_blockscale
    if activation == MoEActivation.SILU and (
'''
NEW = '''    _gu = _gg if _gg is not None else _reax_moe_gu(activation, activation_config, rep_a_fp4, w1_fp4, rep_a_blockscale,
                                                   w1_blockscale, w1_alphas, a2_gscale, problem_sizes1, expert_offsets,
                                                   blockscale_offsets, num_topk)
    if _gu is None:
        ops.cutlass_fp4_moe_mm(
            c1,
            rep_a_fp4,
            w1_fp4,
            rep_a_blockscale,
            w1_blockscale,
            w1_alphas,
            problem_sizes1,
            expert_offsets[:-1],
            blockscale_offsets[:-1],
        )
    del rep_a_fp4, rep_a_blockscale
    if _gu is not None:
        int_fp4, int_blockscale = _gu
    elif activation == MoEActivation.SILU and (
'''
GQ_OLD = '''    _gq = _reax_gquant(a, a_map, a1_gscale, expert_offsets, blockscale_offsets, num_topk, c_map, topk_ids)
    if _gq is not None:
        rep_a_fp4, rep_a_blockscale = _gq
    else:
'''
GQ_NEW = '''    # engv2 moe_gu.py: tquant + the gate|up GEMM reading A rows through a_map (REAX_MOE_GU_GATHER=1), or None
    _gg = _reax_moe_gu_g(activation, activation_config, a, a_map, a1_gscale, w1_fp4, w1_blockscale, w1_alphas,
                         a2_gscale, problem_sizes1, expert_offsets, blockscale_offsets, num_topk)
    _gq = None if _gg is not None else _reax_gquant(a, a_map, a1_gscale, expert_offsets, blockscale_offsets, num_topk,
                                                    c_map, topk_ids)
    if _gg is not None:
        rep_a_fp4 = rep_a_blockscale = None
    elif _gq is not None:
        rep_a_fp4, rep_a_blockscale = _gq
    else:
'''
ADD = '''

# ---------------------------------------------------------------- engv2: gate|up GEMM + gelu*up + NVFP4 quant (moe_gu.py)
import ctypes as _reax_gu_ctypes
import os as _reax_gu_os

_REAX_GU = {"cfg": None, "g": False, "big": None, "lib": None, "dump": []}


def _reax_gu_lib():
    st = _REAX_GU
    if st["lib"] is None:
        c = _reax_gu_ctypes
        lib = (c.PyDLL if _reax_gu_os.environ.get("REAX_PYDLL", "0") == "1" else c.CDLL)(
            _reax_gu_os.environ.get("REAX_MOE_GU_LIB", "/opt/engv2/libreax_moegu.so"))
        lib.reax_moegu_launch.restype = c.c_int
        lib.reax_moegu_launch.argtypes = [c.c_int, c.c_int] + [c.c_void_p] * 11 + [c.c_int, c.c_int, c.c_float,
                                                                                    c.c_void_p]
        if hasattr(lib, "reax_moegu_launch_g"):
            lib.reax_moegu_launch_g.restype = c.c_int
            lib.reax_moegu_launch_g.argtypes = [c.c_int, c.c_int] + [c.c_void_p] * 12 + [c.c_int, c.c_int, c.c_float,
                                                                                        c.c_void_p]
        st["lib"] = lib
    return st["lib"]


def _reax_gu_cfg():
    st = _REAX_GU
    st["cfg"] = (-1, 0)
    path = _reax_gu_os.environ.get("REAX_MOE_GU_FILE", "")
    if _reax_gu_os.environ.get("REAX_MOE_GU", "0") == "1" and path and _reax_gu_os.path.exists(path):
        f = open(path).read().split()
        if len(f) >= 2 and int(f[0]) >= 0:
            st["cfg"] = (int(f[0]), int(f[1]))
            st["g"] = (_reax_gu_os.environ.get("REAX_MOE_GU_GATHER", "0") == "1" and len(f) > 2 and f[2] == "1"
                       and hasattr(_reax_gu_lib(), "reax_moegu_launch_g"))
            if not st["g"] and len(f) > 3 and int(f[3]) >= 0:  # gather won the derive but is off in this instance
                st["cfg"] = (int(f[3]), int(f[1]))  # (entry_dual REAX_MOE_GU_GATHER_IMAGE=0): best expanded variant
            # REAX_MOE_GU_BIG=<rows> (default off): the derive's pick for large steps (full waves), gather path only
            big = int(_reax_gu_os.environ.get("REAX_MOE_GU_BIG", "0") or 0)
            if st["g"] and big > 0 and _reax_gu_os.path.exists(path + ".big"):
                st["big"] = (int(open(path + ".big").read().split()[0]), big)
    print(f"reax moe_gu: {'variant %d numerics %d' % st['cfg'] if st['cfg'][0] >= 0 else 'off'}, gather "
          f"{'on' if st['g'] else 'off'}"
          + (f", variant {st['big'][0]} from {st['big'][1]} rows" if st["big"] else "")
          + f" (check file {path or '-'})", flush=True)


def _reax_moe_gu_g(activation, activation_config, x, a_map, a1_gscale, w1, w1_sf, w1_alphas, a2_gscale, problem_sizes1,
                   expert_offsets, blockscale_offsets, topk):
    """The gate|up result of _reax_moe_gu from the bf16 hidden states x [m, 2816]: tquant (one FP4 row per token, plain
    scales) + moe_gu reading expert rows through a_map; None unless REAX_MOE_GU_GATHER is on and every expert has the
    same input global scale (then the token's bytes are the same for all its experts)."""
    st = _REAX_GU
    if st["cfg"] is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        _reax_gu_cfg()
    v, num = st["cfg"]
    if not st["g"] or v < 0 or activation != MoEActivation.GELU_TANH:
        return None
    if activation_config is not None and activation_config.clamp_limit is not None:
        return None
    if (x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != 2816 or not x.is_contiguous()
            or a_map.dtype != torch.int32 or w1.dim() != 3 or tuple(w1.shape[1:]) != (1408, 1408)
            or not w1.is_contiguous() or not w1_sf.is_contiguous() or a1_gscale.dtype != torch.float32
            or not a1_gscale.is_contiguous() or a2_gscale.dtype != torch.float32 or w1_alphas.dtype != torch.float32):
        return None
    uni = getattr(a1_gscale, "_reax_uni", None)  # one check per scale tensor (a layer's parameter)
    if uni is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        uni = bool((a1_gscale == a1_gscale[:1]).all())
        a1_gscale._reax_uni = uni
    if not uni:
        return None
    m = x.shape[0]
    m_numtopk = a_map.shape[0]
    if m_numtopk != m * topk or m_numtopk > envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE * topk:
        return None
    if st["big"] is not None and m_numtopk >= st["big"][1]:  # static per CUDA graph size: fixed at capture
        v = st["big"][0]
    gql = _reax_gq_lib()
    stream = torch.cuda.current_stream().cuda_stream
    aq = torch.empty(m, 1408, device=x.device, dtype=torch.uint8)
    asf = torch.empty(m, 176, device=x.device, dtype=torch.uint8)
    rc = gql.reax_tquant_launch(m, 2816, x.data_ptr(), a1_gscale.data_ptr(), aq.data_ptr(), asf.data_ptr(),
                                _REAX_GQ["sms"], stream)
    if rc != 0:
        raise RuntimeError(f"reax_tquant_launch: CUDA error {rc}")
    out = torch.empty(m_numtopk, 352, device=x.device, dtype=torch.uint8)
    sf = torch.empty(envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE * topk, 11, dtype=torch.int32, device=x.device)
    rc = _reax_gu_lib().reax_moegu_launch_g(
        v, num, out.data_ptr(), sf.data_ptr(), aq.data_ptr(), w1.data_ptr(), asf.data_ptr(), w1_sf.data_ptr(),
        w1_alphas.data_ptr(), a2_gscale.data_ptr(), problem_sizes1.data_ptr(), expert_offsets.data_ptr(),
        blockscale_offsets.data_ptr(), a_map.data_ptr(), w1.shape[0], m_numtopk, 0.0, stream)
    if rc != 0:
        raise RuntimeError(f"reax_moegu_launch_g: error {rc}")
    return out, sf.view(torch.float8_e4m3fn)


def _reax_moe_gu(activation, activation_config, a, w1, a_sf, w1_sf, w1_alphas, a2_gscale, problem_sizes1,
                 expert_offsets, blockscale_offsets, topk):
    """(int_fp4, int_blockscale) as cutlass_fp4_moe_mm + gelu_tanh_and_mul + scaled_fp4_experts_quant leave them (same
    allocation as moe_gact), or None (vLLM's / moe_gact's path)."""
    st = _REAX_GU
    if st["cfg"] is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        _reax_gu_cfg()
    dump = _reax_gu_os.environ.get("REAX_MOE_GU_DUMP", "")
    if dump and not torch.cuda.is_current_stream_capturing():  # rows per expert of each call (eager runs only)
        st["dump"].append(problem_sizes1[:, 0].cpu())
        if len(st["dump"]) % 240 == 0:
            try:
                _reax_gu_os.makedirs(_reax_gu_os.path.dirname(dump) or ".", exist_ok=True)
                torch.save(torch.stack(st["dump"]), dump)
            except OSError as e:
                print(f"reax moe_gu: dump failed ({e!r})", flush=True)
    v, num = st["cfg"]
    if v < 0 or activation != MoEActivation.GELU_TANH:
        return None
    if activation_config is not None and activation_config.clamp_limit is not None:
        return None
    if (w1.dim() != 3 or tuple(w1.shape[1:]) != (1408, 1408) or a.dim() != 2 or a.shape[1] != 1408
            or not a.is_contiguous() or not w1.is_contiguous() or not a_sf.is_contiguous()
            or not w1_sf.is_contiguous() or a2_gscale.dtype != torch.float32 or w1_alphas.dtype != torch.float32):
        return None
    m_numtopk = a.shape[0]
    if m_numtopk > envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE * topk:
        return None
    out = torch.empty(m_numtopk, 352, device=a.device, dtype=torch.uint8)
    sf = torch.empty(envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE * topk, 11, dtype=torch.int32, device=a.device)
    rc = _reax_gu_lib().reax_moegu_launch(
        v, num, out.data_ptr(), sf.data_ptr(), a.data_ptr(), w1.data_ptr(), a_sf.data_ptr(), w1_sf.data_ptr(),
        w1_alphas.data_ptr(), a2_gscale.data_ptr(), problem_sizes1.data_ptr(), expert_offsets.data_ptr(),
        blockscale_offsets.data_ptr(), w1.shape[0], m_numtopk, 0.0, torch.cuda.current_stream().cuda_stream)
    if rc != 0:
        raise RuntimeError(f"reax_moegu_launch: error {rc}")
    return out, sf.view(torch.float8_e4m3fn)
'''

if len(sys.argv) > 1 and sys.argv[1] not in ("test", "gtest", "derive"):
    path = sys.argv[1]
    src = open(path).read()
    assert src.count(OLD) == 1, "gate|up GEMM + activation branch not found"
    assert src.count(GQ_OLD) == 1, "moe_gquant.py's call not found (install moe_gquant.py first)"
    src = src.replace(OLD, NEW).replace(GQ_OLD, GQ_NEW) + ADD
    if "import vllm.envs as envs" not in src and "from vllm import envs" not in src:
        src += "\nimport vllm.envs as envs\n"
    open(path, "w").write(src)
    print("moe_gu: installed")
    sys.exit(0)

import torch
from vllm import _custom_ops as ops
import vllm.envs as envs

E, TOPK, NG, K = 128, 8, 704, 2816


def _lib():
    lib = C.CDLL(os.environ.get("REAX_MOE_GU_LIB", "/opt/engv2/libreax_moegu.so"))
    lib.reax_moegu_launch.restype = C.c_int
    lib.reax_moegu_launch.argtypes = [C.c_int, C.c_int] + [C.c_void_p] * 11 + [C.c_int, C.c_int, C.c_float, C.c_void_p]
    lib.reax_moegu_variants.restype = C.c_int
    lib.reax_moegu_variant_name.restype = C.c_char_p
    lib.reax_moegu_variant_name.argtypes = [C.c_int]
    if hasattr(lib, "reax_moegu_launch_g"):
        lib.reax_moegu_launch_g.restype = C.c_int
        lib.reax_moegu_launch_g.argtypes = [C.c_int, C.c_int] + [C.c_void_p] * 12 + [C.c_int, C.c_int, C.c_float,
                                                                                    C.c_void_p]
    return lib


def _gq_lib():
    lib = C.CDLL(os.environ.get("REAX_MOE_GQUANT_LIB", "/opt/engv2/libreax_gquant.so"))
    lib.reax_tquant_launch.restype = C.c_int
    lib.reax_tquant_launch.argtypes = [C.c_int] * 2 + [C.c_void_p] * 4 + [C.c_int, C.c_void_p]
    lib.reax_gquant1_launch.restype = C.c_int
    lib.reax_gquant1_launch.argtypes = [C.c_int] * 3 + [C.c_void_p] * 8 + [C.c_int, C.c_int, C.c_void_p]
    return lib


def _gact_lib():
    lib = C.CDLL(os.environ.get("REAX_MOE_GACT_LIB", "/opt/engv2/libreax_gact.so"))
    lib.reax_gact_launch.restype = C.c_int
    lib.reax_gact_launch.argtypes = [C.c_int, C.c_int, C.c_int] + [C.c_void_p] * 6 + [C.c_int, C.c_int, C.c_float,
                                                                                    C.c_void_p]
    lib.reax_gact_configs.restype = C.c_int
    return lib


def case(m, g, kind, counts=None):
    if counts is not None:
        m = int(counts.sum()) // TOPK
        ids = torch.repeat_interleave(torch.arange(E, device="cuda"), counts.to("cuda").long())
        topk_ids = ids.view(TOPK, m).t().contiguous().int()
    elif kind == 0:
        topk_ids = torch.stack([torch.randperm(E, device="cuda", generator=g)[:TOPK] for _ in range(m)]).int()
    else:
        w = torch.ones(E, device="cuda")
        w[:8] = 40.0 if kind == 1 else 400.0
        topk_ids = torch.stack([torch.multinomial(w, TOPK, replacement=False, generator=g) for _ in range(m)]).int()
    eo = torch.empty(E + 1, dtype=torch.int32, device="cuda")
    bo = torch.empty(E + 1, dtype=torch.int32, device="cuda")
    p1 = torch.empty((E, 3), dtype=torch.int32, device="cuda")
    p2 = torch.empty((E, 3), dtype=torch.int32, device="cuda")
    am = torch.empty(m * TOPK, dtype=torch.int32, device="cuda")
    cm = torch.empty(m * TOPK, dtype=torch.int32, device="cuda")
    ops.get_cutlass_moe_mm_data(topk_ids, eo, p1, p2, am, cm, E, NG, K, bo, is_gated=True)
    a = torch.randint(0, 256, (m * TOPK, K // 2), dtype=torch.uint8, device="cuda", generator=g)
    b = torch.randint(0, 256, (E, 2 * NG, K // 2), dtype=torch.uint8, device="cuda", generator=g)
    srows = int(bo[-1].item()) + 128
    a_sf = torch.randint(0x18, 0x48, (srows, K // 16), dtype=torch.uint8, device="cuda", generator=g)
    b_sf = torch.randint(0x18, 0x48, (E, 2 * NG, K // 16), dtype=torch.uint8, device="cuda", generator=g)
    alphas = torch.exp2(torch.randint(-26, -16, (E,), device="cuda", generator=g).float()) * (
        1.0 + torch.rand(E, device="cuda", generator=g))
    gscale = torch.exp2(torch.randint(-6, 13, (E,), device="cuda", generator=g).float()) * (
        1.0 + torch.rand(E, device="cuda", generator=g))
    return dict(a=a, b=b, a_sf=a_sf.view(torch.float8_e4m3fn), b_sf=b_sf.view(torch.float8_e4m3fn),
                alphas=alphas.float(), gscale=gscale.float(), p1=p1, eo=eo, bo=bo, m=m, am=am, cm=cm, ids=topk_ids)


def gather_inputs(c, g, xk):
    m = c["m"]
    if xk == 0:
        x = (torch.randn(m, K, device="cuda", generator=g) * 0.5).bfloat16()
    else:
        ex = torch.randint(-133, 31, (m, K), device="cuda", generator=g, dtype=torch.int32).float()
        x = (torch.randn(m, K, device="cuda", generator=g) * torch.exp2(ex)).bfloat16()
    g1 = (torch.exp2(torch.randint(-6, 13, (1,), device="cuda", generator=g).float())
          * (1.0 + torch.rand(1, device="cuda", generator=g))).expand(E).contiguous()
    rows = envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE * TOPK
    ro = torch.empty(m * TOPK, K // 2, device="cuda", dtype=torch.uint8)
    rs = torch.zeros(rows, (K // 16 + 3) // 4, dtype=torch.int32, device="cuda")
    torch.ops._C.scaled_fp4_experts_quant(ro, rs, ops.shuffle_rows(x, c["am"]), g1, c["eo"], c["bo"])
    c["a"], c["a_sf"], c["x"], c["g1"] = ro, rs.view(torch.uint8).view(torch.float8_e4m3fn), x, g1
    return c


def run_g(lib, gl, v, numerics, c, o, s, aq, asf):
    m, stream = c["m"], torch.cuda.current_stream().cuda_stream
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    rc = gl.reax_tquant_launch(m, K, c["x"].data_ptr(), c["g1"].data_ptr(), aq.data_ptr(), asf.data_ptr(), sms, stream)
    rc = rc or lib.reax_moegu_launch_g(v, numerics, o.data_ptr(), s.data_ptr(), aq.data_ptr(), c["b"].data_ptr(),
                                       asf.data_ptr(), c["b_sf"].data_ptr(), c["alphas"].data_ptr(),
                                       c["gscale"].data_ptr(), c["p1"].data_ptr(), c["eo"].data_ptr(),
                                       c["bo"].data_ptr(), c["am"].data_ptr(), E, m * TOPK, 0.0, stream)
    if rc:
        raise RuntimeError(f"tquant / reax_moegu_launch_g: error {rc}")


def run_q1(lib, gl, v, numerics, c, o, s, qo, qs):
    m, stream = c["m"], torch.cuda.current_stream().cuda_stream
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    rc = gl.reax_gquant1_launch(m, TOPK, K, c["x"].data_ptr(), c["cm"].data_ptr(), c["ids"].data_ptr(),
                                c["g1"].data_ptr(), qo.data_ptr(), qs.data_ptr(), c["eo"].data_ptr(), c["bo"].data_ptr(),
                                E, sms, stream)
    if rc:
        raise RuntimeError(f"reax_gquant1_launch: error {rc}")
    rc = lib.reax_moegu_launch(v, numerics, o.data_ptr(), s.data_ptr(), qo.data_ptr(), c["b"].data_ptr(),
                               qs.data_ptr(), c["b_sf"].data_ptr(), c["alphas"].data_ptr(), c["gscale"].data_ptr(),
                               c["p1"].data_ptr(), c["eo"].data_ptr(), c["bo"].data_ptr(), E, m * TOPK, 0.0, stream)
    if rc:
        raise RuntimeError(f"reax_moegu_launch: error {rc}")


def check_gather(lib, gl, cand, small=False):
    g = torch.Generator(device="cuda").manual_seed(4244)
    ms = (1, 16, 200, 887) if small else (1, 2, 3, 7, 8, 16, 31, 64, 100, 128, 200, 333, 512, 887, 1000, 1024, 1500)
    ok = set(cand)
    for m in ms:
        for kind in (0, 1, 2):
            for xk in (0, 1):
                c = gather_inputs(case(m, g, kind), g, xk)
                ro, rs = out_bufs(m)
                ref(c, ro, rs)
                aq = torch.empty(m, K // 2, device="cuda", dtype=torch.uint8)
                asf = torch.empty(m, K // 16, device="cuda", dtype=torch.uint8)
                for v, n in sorted(ok):
                    o, s = out_bufs(m)
                    run_g(lib, gl, v, n, c, o, s, aq, asf)
                    torch.cuda.synchronize()
                    if not (torch.equal(o, ro) and torch.equal(s, rs)):
                        ok.discard((v, n))
                        print(f"moe_gu gather: variant {v} numerics {n} MISMATCH m {m} kind {kind} x {xk}: fp4 bytes "
                              f"{int((o != ro).sum())}, scale words {int((s != rs).sum())}", flush=True)
    return sorted(ok)


def gtest():
    lib, gl = _lib(), _gq_lib()
    names = [lib.reax_moegu_variant_name(i).decode() for i in range(lib.reax_moegu_variants())]
    cand = [(names.index(x), n) for x in DERIVE for n in (0, 1)]
    ok = check_gather(lib, gl, cand, small=len(sys.argv) > 2)
    print(f"moe_gu gather: exact {[(names[v], n) for v, n in ok]} of {len(cand)}", flush=True)
    g = torch.Generator(device="cuda").manual_seed(7)
    for m, kind in ((440, 0), (887, 0), (0, 10), (0, 11)):
        cs = [gather_inputs(c, g, 0) for c in rot_cases(m, g, kind)]
        for c in cs[1:]:
            for k in ("x", "g1", "am", "cm", "ids"):
                c[k] = cs[0][k]
        m = cs[0]["m"]
        o, s = out_bufs(m)
        aq = torch.empty(m, K // 2, device="cuda", dtype=torch.uint8)
        asf = torch.empty(m, K // 16, device="cuda", dtype=torch.uint8)
        qo = torch.empty(m * TOPK, K // 2, device="cuda", dtype=torch.uint8)
        qs = torch.empty(envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE * TOPK, 44, device="cuda", dtype=torch.int32)
        row = []
        for v in sorted({v for v, _ in ok}):
            t1 = rot_time(lambda c, v=v: run_q1(lib, gl, v, 0, c, o, s, qo, qs), cs)
            tg = rot_time(lambda c, v=v: run_g(lib, gl, v, 0, c, o, s, aq, asf), cs)
            row.append(f"{names[v]} gquant1+gu {t1 * 1e3:.1f} / gather {tg * 1e3:.1f} us ({100 * (tg / t1 - 1):+.1f} %)")
        print(f"m {m:5d} routing {kind}: " + " | ".join(row), flush=True)


def out_bufs(m):
    o = torch.zeros(m * TOPK, NG // 2, dtype=torch.uint8, device="cuda")
    s = torch.zeros(envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE * TOPK, (NG // 16 + 3) // 4, dtype=torch.int32,
                    device="cuda")
    return o, s


def ref(c, o, s):
    m = c["m"]
    c1 = torch.empty(m * TOPK, 2 * NG, dtype=torch.bfloat16, device="cuda")
    ops.cutlass_fp4_moe_mm(c1, c["a"], c["b"], c["a_sf"], c["b_sf"], c["alphas"], c["p1"], c["eo"][:-1], c["bo"][:-1])
    c2 = torch.empty(m * TOPK, NG, dtype=torch.bfloat16, device="cuda")
    torch.ops._C.gelu_tanh_and_mul(c2, c1)
    torch.ops._C.scaled_fp4_experts_quant(o, s, c2, c["gscale"], c["eo"], c["bo"])
    return c1


def run(lib, v, numerics, c, o, s):
    rc = lib.reax_moegu_launch(v, numerics, o.data_ptr(), s.data_ptr(), c["a"].data_ptr(), c["b"].data_ptr(),
                               c["a_sf"].data_ptr(), c["b_sf"].data_ptr(), c["alphas"].data_ptr(),
                               c["gscale"].data_ptr(), c["p1"].data_ptr(), c["eo"].data_ptr(), c["bo"].data_ptr(), E,
                               c["a"].shape[0], 0.0, torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f"reax_moegu_launch: error {rc}")


def two_ops(gl, gv, sms, c, o, s, c1):
    ops.cutlass_fp4_moe_mm(c1, c["a"], c["b"], c["a_sf"], c["b_sf"], c["alphas"], c["p1"], c["eo"][:-1], c["bo"][:-1])
    rc = gl.reax_gact_launch(gv, c1.shape[0], NG, c1.data_ptr(), c["gscale"].data_ptr(), o.data_ptr(), s.data_ptr(),
                             c["eo"].data_ptr(), c["bo"].data_ptr(), E, sms, 0.0,
                             torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f"reax_gact_launch: error {rc}")


def check(lib, small=False, only=None, nums=(0, 1)):
    names = [lib.reax_moegu_variant_name(i).decode() for i in range(lib.reax_moegu_variants())]
    g = torch.Generator(device="cuda").manual_seed(4243)
    ms = (1, 16, 200, 887) if small else (1, 2, 3, 7, 8, 16, 31, 64, 100, 128, 200, 333, 512, 887, 1000, 1024, 1500)
    exact = {(v, n) for v in range(len(names)) for n in nums if only is None or names[v] in only}
    for m in ms:
        for kind in (0, 1, 2):
            c = case(m, g, kind)
            ro, rs = out_bufs(m)
            ref(c, ro, rs)
            for v, n in sorted(exact):
                o, s = out_bufs(m)
                run(lib, v, n, c, o, s)
                torch.cuda.synchronize()
                if not (torch.equal(o, ro) and torch.equal(s, rs)):
                    exact.discard((v, n))
                    print(f"moe_gu: {names[v]} numerics {n} MISMATCH m {m} kind {kind}: fp4 bytes {int((o != ro).sum())}"
                          f" of {o.numel()}, scale words {int((s != rs).sum())}", flush=True)
    return names, sorted(exact)


def main():
    from triton.testing import do_bench
    lib, gl = _lib(), _gact_lib()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    names, exact = check(lib, small=len(sys.argv) > 2)
    print(f"moe_gu: exact (variant, numerics): {[(names[v], n) for v, n in exact]}", flush=True)
    nums = sorted({n for _, n in exact})
    if not exact:
        return
    num = nums[0]
    g = torch.Generator(device="cuda").manual_seed(7)
    for m, kind in ((110, 0), (440, 0), (887, 0), (1000, 0), (1760, 0), (887, 1), (887, 2), (0, 10), (0, 11)):
        c = case(m, g, kind, counts=torch.tensor(REAL[kind - 10], dtype=torch.int32) if kind >= 10 else None)
        m = c["m"]
        o, s = out_bufs(m)
        c1 = torch.empty(m * TOPK, 2 * NG, dtype=torch.bfloat16, device="cuda")
        t_ref = min(min(do_bench(lambda gv=gv: two_ops(gl, gv, sms, c, o, s, c1), warmup=25, rep=100) for _ in range(2))
                    for gv in range(num, 2 * gl.reax_gact_configs(), 2))
        t_mm = min(do_bench(lambda: ops.cutlass_fp4_moe_mm(c1, c["a"], c["b"], c["a_sf"], c["b_sf"], c["alphas"],
                                                            c["p1"], c["eo"][:-1], c["bo"][:-1]), warmup=25, rep=100)
                   for _ in range(2))
        res = sorted((min(do_bench(lambda v=v: run(lib, v, num, c, o, s), warmup=25, rep=100) for _ in range(2)), v)
                     for v, n in exact if n == num)
        mb = (E * 2 * NG * (K // 2 + K // 16) + m * TOPK * (K // 2 + NG // 2)) / 1e6
        print(f"m {m:5d} routing {kind}: cutlass gate|up {t_mm * 1e3:6.1f} us + gact = {t_ref * 1e3:6.1f} us | moe_gu best "
              f"{names[res[0][1]]} {res[0][0] * 1e3:6.1f} us ({mb / res[0][0] / 1e3:4.2f} TB/s) -> "
              f"{100 * (res[0][0] / t_ref - 1):+.1f} % | " + " ".join(f"{names[v]} {t * 1e3:.1f}" for t, v in res),
              flush=True)


DERIVE = ("r64_m2_s3_k128", "r128_m2_s2_k128", "r64_m1_s3_k128", "r128_m2_s4", "r64_m2_s4")

REAL = ([168,0,1,0,28,170,2,1,49,10,78,41,8,98,35,3,7,13,26,3,64,5,56,70,65,49,0,17,62,30,9,91,37,105,57,4,101,311,151,1,119,0,82,0,48,29,33,56,0,9,16,458,2,22,1,20,14,452,17,16,1,69,68,30,20,80,526,10,158,180,29,6,347,0,192,14,26,15,72,93,25,0,10,12,0,18,40,9,456,15,15,0,0,13,0,7,2,0,26,3,13,37,0,26,133,69,0,48,1,136,238,33,94,0,5,220,74,31,111,15,0,21,23,101,3,13,260,8],
        [3,25,13,59,4,26,77,4,133,0,68,2,10,20,9,0,0,3,168,2,16,18,36,16,0,51,215,5,4,13,3,44,14,221,11,4,21,11,159,46,730,0,5,30,152,1,158,22,3,2,4,16,2,3,7,95,405,3,3,177,10,6,1,0,3,5,55,19,3,4,8,79,23,4,1,34,0,1,56,4,4,112,3,0,9,21,190,2,0,46,0,0,60,72,19,28,16,340,1,2,0,55,8,66,2,18,12,9,1,49,29,577,0,47,1,240,0,13,14,291,0,520,21,77,263,0,0,14])


def rot_cases(m, g, kind, n=4):
    counts = torch.tensor(REAL[kind - 10], dtype=torch.int32) if kind >= 10 else None
    cs = [case(m, g, kind, counts=counts) for _ in range(n)]
    m = cs[0]["m"]
    for c in cs[1:]:
        for k in ("a", "a_sf", "p1", "eo", "bo", "m"):
            c[k] = cs[0][k]
    return cs


def rot_time(fn, cs, reps=15):
    k = 2 * len(cs)
    for i in range(k):
        fn(cs[i % len(cs)])
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for i in range(k):
            fn(cs[i % len(cs)])
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000 / k)
    ts.sort()
    return ts[len(ts) // 2] / 1000


def derive(path):
    res = "-1"
    if os.path.exists(path + ".big"):
        os.remove(path + ".big")
    try:
        lib, gl = _lib(), _gact_lib()
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        gf = os.environ.get("REAX_MOE_GACT_FILE", "")
        gv = int(open(gf).read().split()[0]) if gf and os.path.exists(gf) else 0
        num = gv & 1 if gv >= 0 else 0
        only = tuple(x for x in os.environ.get("REAX_MOE_GU_ONLY", "").split(",") if x) or DERIVE
        gain = float(os.environ.get("REAX_MOE_GU_GAIN", "0.02"))
        names, exact = check(lib, only=only, nums=(num,))
        cand = [v for v, n in exact if n == num]
        if cand:
            g = torch.Generator(device="cuda").manual_seed(7)
            ts, tr = {}, []
            for m, kind in ((440, 0), (887, 0), (0, 10), (0, 11)):
                cs = rot_cases(m, g, kind)
                m = cs[0]["m"]
                o, s = out_bufs(m)
                c1 = torch.empty(m * TOPK, 2 * NG, dtype=torch.bfloat16, device="cuda")
                tr.append(rot_time(lambda c: two_ops(gl, max(gv, 0), sms, c, o, s, c1), cs))
                for v in cand:
                    ts.setdefault(v, []).append(rot_time(lambda c, v=v: run(lib, v, num, c, o, s), cs))
                del cs
            best = min(cand, key=lambda v: max(t / r for t, r in zip(ts[v], tr)))
            print("reax moe_gu: " + ", ".join(f"m {m} {t * 1e3:.1f} us vs {r * 1e3:.1f}"
                                              for m, t, r in zip(("440", "887", "965 real", "865 real"), ts[best], tr))
                  + f" ({names[best]}, numerics {num}, {len(exact)} exact)", flush=True)
            if gain < 0 or all(ts[best][i] < tr[i] / (1 + gain) for i in (1, 2, 3)):
                res = f"{best} {num}"
                if os.environ.get("REAX_MOE_GU_GATHER", "0") == "1":
                    gq = _gq_lib()
                    gcand = check_gather(lib, gq, [(v, num) for v in cand])
                    t1, tg = {v: [] for v in cand}, {v: [] for v, _ in gcand}
                    for m, kind in ((887, 0), (0, 10), (0, 11)):
                        cs = [gather_inputs(c, g, 0) for c in rot_cases(m, g, kind)]
                        for c in cs[1:]:
                            for k in ("x", "g1", "am", "cm", "ids"):
                                c[k] = cs[0][k]
                        m = cs[0]["m"]
                        o, s = out_bufs(m)
                        aq = torch.empty(m, K // 2, device="cuda", dtype=torch.uint8)
                        asf = torch.empty(m, K // 16, device="cuda", dtype=torch.uint8)
                        qo = torch.empty(m * TOPK, K // 2, device="cuda", dtype=torch.uint8)
                        qs = torch.empty(envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE * TOPK, 44, device="cuda",
                                         dtype=torch.int32)
                        for v in cand:
                            t1[v].append(rot_time(lambda c, v=v: run_q1(lib, gq, v, num, c, o, s, qo, qs), cs))
                        for v in tg:
                            tg[v].append(rot_time(lambda c, v=v: run_g(lib, gq, v, num, c, o, s, aq, asf), cs))
                        del cs
                    mean = lambda ts: sum(ts) / len(ts)
                    b1 = min(t1, key=lambda v: mean(t1[v]))
                    bg = min(tg, key=lambda v: mean(tg[v])) if tg else None
                    on = bg is not None and mean(tg[bg]) < mean(t1[b1]) / 1.01
                    res = f"{bg} {num} 1 {b1}" if on else res + " 0"
                    if on and int(os.environ.get("REAX_MOE_GU_BIG", "0") or 0) > 0 and len(tg) > 1:
                        tb = {v: [] for v in tg}
                        for m, kind in ((1760, 0), (1760, 1)):
                            cs = [gather_inputs(c, g, 0) for c in rot_cases(m, g, kind)]
                            for c in cs[1:]:
                                for k in ("x", "g1", "am", "cm", "ids"):
                                    c[k] = cs[0][k]
                            m = cs[0]["m"]
                            o, s = out_bufs(m)
                            aq = torch.empty(m, K // 2, device="cuda", dtype=torch.uint8)
                            asf = torch.empty(m, K // 16, device="cuda", dtype=torch.uint8)
                            for v in tb:
                                tb[v].append(rot_time(lambda c, v=v: run_g(lib, gq, v, num, c, o, s, aq, asf), cs))
                            del cs
                        bb = min(tb, key=lambda v: mean(tb[v]))
                        pick = bb if bb != bg and mean(tb[bb]) < mean(tb[bg]) / 1.01 else None
                        if pick is not None:
                            open(path + ".big", "w").write(f"{pick}\n")
                        print(f"reax moe_gu big: m 1760 best {names[bb]} {mean(tb[bb]) * 1e3:.1f} us vs "
                              f"{names[bg]} {mean(tb[bg]) * 1e3:.1f} -> {names[pick] if pick is not None else 'none'}",
                              flush=True)
                    print(f"reax moe_gu gather: {len(gcand)}/{len(cand)} exact; best expanded {names[b1]} "
                          f"{mean(t1[b1]) * 1e3:.1f} us" + (f", best gather {names[bg]} {mean(tg[bg]) * 1e3:.1f} us"
                                                            if bg is not None else "")
                          + f" (gquant1 / tquant + GEMM, mean of m 887 + 2 REAL) -> {'on' if on else 'off'}",
                          flush=True)
    except Exception as e:
        print(f"reax moe_gu: check failed ({e!r})", flush=True)
    open(path, "w").write(res + "\n")
    print("reax moe_gu:", res, flush=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "test":
    main()
elif __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "gtest":
    gtest()
elif __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "derive":
    derive(sys.argv[2])
