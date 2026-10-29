import sys

OLD = '''    else:
        apply_moe_activation(
            activation,
            c2,
            c1,
            activation_config=activation_config,
        )
        int_fp4, int_blockscale = ops.scaled_fp4_experts_quant(
            c2, a2_gscale, expert_offsets, blockscale_offsets, num_topk
        )
'''
NEW = '''    else:
        _ga = _reax_gact(activation, activation_config, c1, a2_gscale, expert_offsets, blockscale_offsets, num_topk)
        if _ga is not None:
            int_fp4, int_blockscale = _ga
        else:
            apply_moe_activation(
                activation,
                c2,
                c1,
                activation_config=activation_config,
            )
            int_fp4, int_blockscale = ops.scaled_fp4_experts_quant(
                c2, a2_gscale, expert_offsets, blockscale_offsets, num_topk
            )
'''
ADD = '''

# ---------------------------------------------------------------- engv2: gelu_tanh_and_mul + NVFP4 experts quant (moe_gact.py)
import ctypes as _reax_ga_ctypes
import os as _reax_ga_os

_REAX_GA = {"variant": None, "lib": None, "sms": 0}


def _reax_ga_lib():
    st = _REAX_GA
    if st["lib"] is None:
        c = _reax_ga_ctypes
        lib = (c.PyDLL if _reax_ga_os.environ.get("REAX_PYDLL", "0") == "1" else c.CDLL)(_reax_ga_os.environ.get("REAX_MOE_GACT_LIB", "/opt/engv2/libreax_gact.so"))
        lib.reax_gact_launch.restype = c.c_int
        lib.reax_gact_launch.argtypes = [c.c_int, c.c_int, c.c_int] + [c.c_void_p] * 6 + [c.c_int, c.c_int, c.c_float,
                                                                                          c.c_void_p]
        lib.reax_gact_configs.restype = c.c_int
        st["lib"] = lib
        st["sms"] = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return st["lib"]


def _reax_gact_raw(variant, c1, a2_gscale, expert_offsets, blockscale_offsets, out, sf):
    rc = _reax_ga_lib().reax_gact_launch(
        variant, c1.shape[0], c1.shape[1] // 2, c1.data_ptr(), a2_gscale.data_ptr(), out.data_ptr(), sf.data_ptr(),
        expert_offsets.data_ptr(), blockscale_offsets.data_ptr(), a2_gscale.shape[0], _REAX_GA["sms"], 0.0,
        torch.cuda.current_stream().cuda_stream)
    if rc != 0:
        raise RuntimeError(f"reax_gact_launch: CUDA error {rc}")


def _reax_gact(activation, activation_config, c1, a2_gscale, expert_offsets, blockscale_offsets, topk):
    """Same outputs (and allocation) as gelu_tanh_and_mul + ops.scaled_fp4_experts_quant, or None."""
    st = _REAX_GA
    if st["variant"] is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        st["variant"] = -1
        if _reax_ga_os.environ.get("REAX_MOE_GACT", "1") == "1":
            path = _reax_ga_os.environ.get("REAX_MOE_GACT_FILE", "")
            try:
                if path and _reax_ga_os.path.exists(path):
                    st["variant"] = int(open(path).read().split()[0])
                    how = f"check result loaded from {path}"
                else:
                    st["variant"] = _reax_gact_pick(small=True)
                    how = "checked in-engine"
            except Exception as e:  # noqa: BLE001 - keep the eager ops
                how = f"disabled ({e!r})"
            print(f"reax moe_gact: variant {st['variant']} ({how})", flush=True)
    if st["variant"] < 0 or activation != MoEActivation.GELU_TANH:
        return None
    if activation_config is not None and activation_config.clamp_limit is not None:
        return None
    if (c1.dtype != torch.bfloat16 or not c1.is_contiguous() or c1.dim() != 2 or c1.shape[1] % 32 != 0
            or a2_gscale.dtype != torch.float32):
        return None
    m_numtopk, k = c1.shape[0], c1.shape[1] // 2
    if m_numtopk > envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE * topk:
        return None
    padded_k = (k // 16 + 3) // 4
    out = torch.empty(m_numtopk, k // 2, device=c1.device, dtype=torch.uint8)
    sf = torch.empty(envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE * topk, padded_k, dtype=torch.int32, device=c1.device)
    _reax_gact_raw(st["variant"], c1, a2_gscale, expert_offsets, blockscale_offsets, out, sf)
    return out, sf.view(torch.float8_e4m3fn)


def _reax_ga_case(m, k, e, topk, g, kind):
    """c1 = [gate | up] rows of the gate|up GEMM output (m * topk rows, 2k wide), routing, per-expert global scales."""
    rows = m * topk
    if kind == 0:  # model-like
        c1 = (torch.randn(rows, 2 * k, device="cuda", generator=g) * 1.5).bfloat16()
    elif kind == 1:  # exponents 2^-133 .. 2^20: bf16 subnormals, zeros, gelu saturation, huge products
        ex = torch.randint(-133, 21, (rows, 2 * k), device="cuda", generator=g, dtype=torch.int32).float()
        c1 = (torch.randn(rows, 2 * k, device="cuda", generator=g) * torch.exp2(ex)).bfloat16()
    else:  # small integers and halves, signed zeros in up, constant and all-zero 16-blocks
        c1 = (torch.randint(-12, 13, (rows, 2 * k), device="cuda", generator=g).float() * 0.5).bfloat16()
        up = c1[:, k:]
        z = torch.rand(rows, k, device="cuda", generator=g) < 0.15
        up[z] = torch.where(torch.rand(rows, k, device="cuda", generator=g) < 0.5, 0.0, -0.0).bfloat16()[z]
        gate = c1[:, :k].reshape(rows, k // 16, 16)
        c = torch.rand(rows, k // 16, device="cuda", generator=g) < 0.05
        gate[c] = gate[c][:, :1].expand(-1, 16).clone()
    logits = torch.randn(m, e, device="cuda", generator=g)
    if kind == 2:
        logits[:, : e // 8] += 3.0  # crowded experts: > 128 rows in some
    topk_ids = logits.topk(topk, dim=1).indices.int()
    gscale = torch.exp2(torch.randint(-6, 13, (e,), device="cuda", generator=g).float()) * (
        1.0 + torch.rand(e, device="cuda", generator=g))
    return c1, topk_ids, gscale.float()


def _reax_gact_check(variant, small=False):
    """Byte-for-byte: packed FP4 output and the whole (pre-zeroed) E4M3 scale buffer vs gelu_tanh_and_mul + vLLM quant."""
    g = torch.Generator(device="cuda").manual_seed(6161)
    k, e, topk = 704, 128, 8
    ms = (1, 3, 8, 31, 64, 200) if small else (1, 2, 3, 7, 8, 16, 31, 64, 100, 128, 200, 333, 512, 887, 1000, 1024, 2048)
    n = 0
    for m in ms:
        for kind in (0, 1, 2):
            c1, topk_ids, gscale = _reax_ga_case(m, k, e, topk, g, kind)
            expert_offsets = torch.empty(e + 1, dtype=torch.int32, device="cuda")
            blockscale_offsets = torch.empty(e + 1, dtype=torch.int32, device="cuda")
            ps1 = torch.empty((e, 3), dtype=torch.int32, device="cuda")
            ps2 = torch.empty((e, 3), dtype=torch.int32, device="cuda")
            a_map = torch.empty(topk_ids.numel(), dtype=torch.int32, device="cuda")
            c_map = torch.empty(topk_ids.numel(), dtype=torch.int32, device="cuda")
            ops.get_cutlass_moe_mm_data(topk_ids, expert_offsets, ps1, ps2, a_map, c_map, e, k, 2816, blockscale_offsets,
                                        is_gated=True)
            padded_k = (k // 16 + 3) // 4
            rows = envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE * topk
            c2 = torch.empty(m * topk, k, device="cuda", dtype=torch.bfloat16)
            torch.ops._C.gelu_tanh_and_mul(c2, c1)
            ref_o = torch.empty(m * topk, k // 2, device="cuda", dtype=torch.uint8)
            ref_s = torch.zeros(rows, padded_k, dtype=torch.int32, device="cuda")
            torch.ops._C.scaled_fp4_experts_quant(ref_o, ref_s, c2, gscale, expert_offsets, blockscale_offsets)
            out_o = torch.empty_like(ref_o)
            out_s = torch.zeros_like(ref_s)
            _reax_gact_raw(variant, c1, gscale, expert_offsets, blockscale_offsets, out_o, out_s)
            torch.cuda.synchronize()
            if not (torch.equal(out_o, ref_o) and torch.equal(out_s, ref_s)):
                print(f"reax moe_gact: variant {variant} MISMATCH at m {m} kind {kind}: fp4 bytes "
                      f"{int((out_o != ref_o).sum())}, scale words {int((out_s != ref_s).sum())}", flush=True)
                return False
            n += 1
    print(f"reax moe_gact: variant {variant} byte-identical in {n} cases (m up to {ms[-1]}, k {k}, {e} experts, "
          f"top-{topk})", flush=True)
    return True


def _reax_gact_bufs(m, g):
    k, e, topk = 704, 128, 8
    c1, topk_ids, gscale = _reax_ga_case(m, k, e, topk, g, 0)
    eo = torch.empty(e + 1, dtype=torch.int32, device="cuda")
    bo = torch.empty(e + 1, dtype=torch.int32, device="cuda")
    p1 = torch.empty((e, 3), dtype=torch.int32, device="cuda")
    p2 = torch.empty((e, 3), dtype=torch.int32, device="cuda")
    am = torch.empty(m * topk, dtype=torch.int32, device="cuda")
    cmap = torch.empty(m * topk, dtype=torch.int32, device="cuda")
    ops.get_cutlass_moe_mm_data(topk_ids, eo, p1, p2, am, cmap, e, k, 2816, bo, is_gated=True)
    out = torch.empty(m * topk, k // 2, device="cuda", dtype=torch.uint8)
    sf = torch.zeros(envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE * topk, (k // 16 + 3) // 4, dtype=torch.int32, device="cuda")
    return c1, gscale, eo, bo, out, sf


def _reax_gact_tune(numerics, verbose=False):
    """Byte-check every launch shape with this numerics (small set), time the exact ones (best of 2) at a step's m."""
    from triton.testing import do_bench
    g = torch.Generator(device="cuda").manual_seed(17)
    bufs = [_reax_gact_bufs(m, g) for m in (440, 887, 1000)]
    best, best_t = numerics, None
    for c in range(_reax_ga_lib().reax_gact_configs()):
        v = numerics + 2 * c
        if c and not _reax_gact_check(v, small=True):
            continue
        tv = [min(do_bench(lambda: _reax_gact_raw(v, *b), warmup=25, rep=100) for _ in range(2)) for b in bufs]
        if verbose:
            print(f"  shape {c}: " + "  ".join(f"m {m} {t * 1e3:5.1f} us" for m, t in zip((440, 887, 1000), tv)),
                  flush=True)
        if best_t is None or sum(tv) < sum(best_t):
            best, best_t = v, tv
    return best


def _reax_gact_pick(small=False, tune=False):
    for v in (0, 1):
        if _reax_gact_check(v, small):
            return _reax_gact_tune(v) if tune else v
    return -1
'''

if len(sys.argv) > 1 and sys.argv[1] in ("test", "derive"):
    import torch
    from vllm.model_executor.layers.fused_moe.experts import cutlass_moe as cm
    if sys.argv[1] == "derive":
        v = -1
        try:
            v = cm._reax_gact_pick(tune=True)
        except Exception as e:
            print(f"reax moe_gact: check failed ({e!r})", flush=True)
        open(sys.argv[2], "w").write(f"{v}\n")
        sys.exit(0)
    for v in (0, 1):
        print("variant", v, "exact", cm._reax_gact_check(v))
    best = cm._reax_gact_tune(0, verbose=True)
    print("fastest launch shape", best >> 1, flush=True)
    from triton.testing import do_bench
    from vllm import _custom_ops as ops
    g = torch.Generator(device="cuda").manual_seed(1)
    for m in (8, 64, 256, 512, 887, 1024, 2048):
        c1, topk_ids, gscale = cm._reax_ga_case(m, 704, 128, 8, g, 0)
        eo = torch.empty(129, dtype=torch.int32, device="cuda")
        bo = torch.empty(129, dtype=torch.int32, device="cuda")
        p1 = torch.empty((128, 3), dtype=torch.int32, device="cuda")
        p2 = torch.empty((128, 3), dtype=torch.int32, device="cuda")
        am = torch.empty(m * 8, dtype=torch.int32, device="cuda")
        cmap = torch.empty(m * 8, dtype=torch.int32, device="cuda")
        ops.get_cutlass_moe_mm_data(topk_ids, eo, p1, p2, am, cmap, 128, 704, 2816, bo, is_gated=True)
        c2 = torch.empty(m * 8, 704, device="cuda", dtype=torch.bfloat16)

        def two():
            torch.ops._C.gelu_tanh_and_mul(c2, c1)
            return ops.scaled_fp4_experts_quant(c2, gscale, eo, bo, 8)

        cm._REAX_GA["variant"] = 0
        te = do_bench(two)
        tf = do_bench(lambda: cm._reax_gact(cm.MoEActivation.GELU_TANH, None, c1, gscale, eo, bo, 8))
        print(f"m {m:5d}: gelu_tanh_and_mul + quant {te * 1e3:7.1f} us  fused {tf * 1e3:7.1f} us  x{te / tf:.2f}",
              flush=True)
    sys.exit(0)
path = sys.argv[1]
src = open(path).read()
assert src.count(OLD) == 1, "activation + quant branch not found"
src = src.replace(OLD, NEW) + ADD
if "import vllm.envs as envs" not in src and "from vllm import envs" not in src:
    src += "\nimport vllm.envs as envs\n"
open(path, "w").write(src)
print("moe_gact: installed")
