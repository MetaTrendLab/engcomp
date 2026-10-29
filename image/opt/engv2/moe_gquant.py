import sys

OLD = '''    a = ops.shuffle_rows(a, a_map)
    rep_a_fp4, rep_a_blockscale = ops.scaled_fp4_experts_quant(
        a,
        a1_gscale,
        expert_offsets,
        blockscale_offsets,
        num_topk,
    )
'''
NEW = '''    _gq = _reax_gquant(a, a_map, a1_gscale, expert_offsets, blockscale_offsets, num_topk)
    if _gq is not None:
        rep_a_fp4, rep_a_blockscale = _gq
    else:
        a = ops.shuffle_rows(a, a_map)
        rep_a_fp4, rep_a_blockscale = ops.scaled_fp4_experts_quant(
            a,
            a1_gscale,
            expert_offsets,
            blockscale_offsets,
            num_topk,
        )
'''
ADD = '''

# ---------------------------------------------------------------- engv2: NVFP4 experts quant reading rows through a_map (moe_gquant.py)
import ctypes as _reax_ctypes
import os as _reax_gq_os

_REAX_GQ = {"on": None, "lib": None, "sms": 0}


def _reax_gq_lib():
    st = _REAX_GQ
    if st["lib"] is None:
        lib = (_reax_ctypes.PyDLL if _reax_gq_os.environ.get("REAX_PYDLL", "0") == "1" else _reax_ctypes.CDLL)(_reax_gq_os.environ.get("REAX_MOE_GQUANT_LIB", "/opt/engv2/libreax_gquant.so"))
        lib.reax_gquant_launch.restype = _reax_ctypes.c_int
        lib.reax_gquant_launch.argtypes = [_reax_ctypes.c_int, _reax_ctypes.c_int] + [_reax_ctypes.c_void_p] * 7 + [
            _reax_ctypes.c_int, _reax_ctypes.c_int, _reax_ctypes.c_void_p]
        st["lib"] = lib
        st["sms"] = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return st["lib"]


def _reax_gquant_raw(a, a_map, a1_gscale, expert_offsets, blockscale_offsets, out, sf):
    rc = _reax_gq_lib().reax_gquant_launch(
        a_map.shape[0], a.shape[1], a.data_ptr(), a_map.data_ptr(), a1_gscale.data_ptr(), out.data_ptr(), sf.data_ptr(),
        expert_offsets.data_ptr(), blockscale_offsets.data_ptr(), a1_gscale.shape[0], _REAX_GQ["sms"],
        torch.cuda.current_stream().cuda_stream)
    if rc != 0:
        raise RuntimeError(f"reax_gquant_launch: CUDA error {rc}")


def _reax_gquant(a, a_map, a1_gscale, expert_offsets, blockscale_offsets, topk):
    """Same outputs (and allocation) as ops.scaled_fp4_experts_quant(ops.shuffle_rows(a, a_map), ...), or None."""
    st = _REAX_GQ
    if st["on"] is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        st["on"] = False
        if _reax_gq_os.environ.get("REAX_MOE_GQUANT", "1") == "1":
            path = _reax_gq_os.environ.get("REAX_MOE_GQUANT_FILE", "")
            try:
                if path and _reax_gq_os.path.exists(path):
                    st["on"] = open(path).read().split()[0] == "1"
                    how = f"check result loaded from {path}"
                else:
                    st["on"] = _reax_gquant_check(small=True)
                    how = "checked in-engine"
            except Exception as e:  # noqa: BLE001 - keep the eager ops
                how = f"disabled ({e!r})"
            print(f"reax moe_gquant: {'on' if st['on'] else 'off'} ({how})", flush=True)
    if not st["on"]:
        return None
    if (a.dtype != torch.bfloat16 or not a.is_contiguous() or a_map.dtype != torch.int32 or a.shape[1] % 16 != 0
            or a1_gscale.dtype != torch.float32):
        return None
    m_numtopk, k = a_map.shape[0], a.shape[1]
    if m_numtopk > envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE * topk:
        return None
    padded_k = (k // 16 + 3) // 4
    out = torch.empty(m_numtopk, k // 2, device=a.device, dtype=torch.uint8)
    sf = torch.empty(envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE * topk, padded_k, dtype=torch.int32, device=a.device)
    _reax_gquant_raw(a, a_map, a1_gscale, expert_offsets, blockscale_offsets, out, sf)
    return out, sf.view(torch.float8_e4m3fn)


def _reax_gq_case(m, k, e, topk, g, kind):
    """Hidden states, routing and per-expert global scales; `kind` picks the value distribution."""
    if kind == 0:  # model-like
        a = (torch.randn(m, k, device="cuda", generator=g) * 0.5).bfloat16()
    elif kind == 1:  # exponents over 2^-133 .. 2^30: bf16 subnormals, zeros (underflow), huge blocks
        ex = torch.randint(-133, 31, (m, k), device="cuda", generator=g, dtype=torch.int32).float()
        a = (torch.randn(m, k, device="cuda", generator=g) * torch.exp2(ex)).bfloat16()
    else:  # small integers and halves: exact e2m1 ties, all-zero and constant 16-blocks
        a = (torch.randint(-12, 13, (m, k), device="cuda", generator=g).float() * 0.5).bfloat16()
        blk = a.view(m, k // 16, 16)
        z = torch.rand(m, k // 16, device="cuda", generator=g) < 0.1
        blk[z] = 0
        c = torch.rand(m, k // 16, device="cuda", generator=g) < 0.05
        blk[c] = blk[c][:, :1].expand(-1, 16).clone()
    logits = torch.randn(m, e, device="cuda", generator=g)
    if kind == 2:
        logits[:, : e // 8] += 3.0  # crowded experts: > 128 rows in some (several SF row tiles)
    topk_ids = logits.topk(topk, dim=-1).indices.to(torch.int32)
    gscale = torch.exp2(torch.randint(-6, 13, (e,), device="cuda", generator=g).float()) * (
        1.0 + torch.rand(e, device="cuda", generator=g))
    return a, topk_ids, gscale.float()


def _reax_gquant_check(small=False):
    """Byte-for-byte: packed FP4 output and the whole (pre-zeroed) E4M3 scale buffer vs shuffle_rows + vLLM quant."""
    g = torch.Generator(device="cuda").manual_seed(5151)
    k, e, topk = 2816, 128, 8
    ms = (1, 3, 8, 31, 64, 200) if small else (1, 2, 3, 7, 8, 16, 31, 64, 100, 128, 200, 333, 512, 1000, 1024, 2048)
    n = 0
    for m in ms:
        for kind in (0, 1, 2):
            a, topk_ids, gscale = _reax_gq_case(m, k, e, topk, g, kind)
            expert_offsets = torch.empty(e + 1, dtype=torch.int32, device="cuda")
            blockscale_offsets = torch.empty(e + 1, dtype=torch.int32, device="cuda")
            ps1 = torch.empty((e, 3), dtype=torch.int32, device="cuda")
            ps2 = torch.empty((e, 3), dtype=torch.int32, device="cuda")
            a_map = torch.empty(topk_ids.numel(), dtype=torch.int32, device="cuda")
            c_map = torch.empty(topk_ids.numel(), dtype=torch.int32, device="cuda")
            ops.get_cutlass_moe_mm_data(topk_ids, expert_offsets, ps1, ps2, a_map, c_map, e, 704, k, blockscale_offsets,
                                        is_gated=True)
            padded_k = (k // 16 + 3) // 4
            rows = envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE * topk
            ref_o = torch.empty(m * topk, k // 2, device="cuda", dtype=torch.uint8)
            ref_s = torch.zeros(rows, padded_k, dtype=torch.int32, device="cuda")
            torch.ops._C.scaled_fp4_experts_quant(ref_o, ref_s, ops.shuffle_rows(a, a_map), gscale, expert_offsets,
                                                  blockscale_offsets)
            out_o = torch.empty_like(ref_o)
            out_s = torch.zeros_like(ref_s)
            _reax_gquant_raw(a, a_map, gscale, expert_offsets, blockscale_offsets, out_o, out_s)
            torch.cuda.synchronize()
            if not (torch.equal(out_o, ref_o) and torch.equal(out_s, ref_s)):
                print(f"reax moe_gquant: MISMATCH at m {m} kind {kind}: fp4 bytes "
                      f"{int((out_o != ref_o).sum())}, scale words {int((out_s != ref_s).sum())}", flush=True)
                return False
            n += 1
    print(f"reax moe_gquant: byte-identical in {n} cases (m up to {ms[-1]}, k {k}, {e} experts, top-{topk})", flush=True)
    return True
'''

if len(sys.argv) > 1 and sys.argv[1] in ("test", "derive"):
    import torch
    from vllm.model_executor.layers.fused_moe.experts import cutlass_moe as cm
    if sys.argv[1] == "derive":
        ok = False
        try:
            ok = cm._reax_gquant_check()
        except Exception as e:
            print(f"reax moe_gquant: check failed ({e!r})", flush=True)
        open(sys.argv[2], "w").write("1\n" if ok else "0\n")
        sys.exit(0)
    print("check", cm._reax_gquant_check())
    from triton.testing import do_bench
    from vllm import _custom_ops as ops
    g = torch.Generator(device="cuda").manual_seed(1)
    cm._REAX_GQ["on"] = True
    for m in (8, 64, 256, 512, 1024, 2048):
        a, topk_ids, gscale = cm._reax_gq_case(m, 2816, 128, 8, g, 0)
        eo = torch.empty(129, dtype=torch.int32, device="cuda")
        bo = torch.empty(129, dtype=torch.int32, device="cuda")
        p1 = torch.empty((128, 3), dtype=torch.int32, device="cuda")
        p2 = torch.empty((128, 3), dtype=torch.int32, device="cuda")
        am = torch.empty(m * 8, dtype=torch.int32, device="cuda")
        cmap = torch.empty(m * 8, dtype=torch.int32, device="cuda")
        ops.get_cutlass_moe_mm_data(topk_ids, eo, p1, p2, am, cmap, 128, 704, 2816, bo, is_gated=True)
        te = do_bench(lambda: ops.scaled_fp4_experts_quant(ops.shuffle_rows(a, am), gscale, eo, bo, 8))
        tf = do_bench(lambda: cm._reax_gquant(a, am, gscale, eo, bo, 8))
        print(f"m {m:5d}: shuffle_rows + quant {te * 1e3:7.1f} us  gather-quant {tf * 1e3:7.1f} us  x{te / tf:.2f}",
              flush=True)
    sys.exit(0)
path = sys.argv[1]
src = open(path).read()
i = src.index("def run_cutlass_moe_fp4(")
j = src.index(OLD, i)
assert j < src.index("\ndef ", i + 1), "run_cutlass_moe_fp4 head not found"
src = src[:j] + NEW + src[j + len(OLD):] + ADD
if "import vllm.envs as envs" not in src and "from vllm import envs" not in src:
    src += "\nimport vllm.envs as envs\n"
open(path, "w").write(src)
print("moe_gquant: installed")
