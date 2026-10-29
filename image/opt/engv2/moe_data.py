import sys

OLD = '''    # Note that problem sizes are based on logical number of elements.
    ops.get_cutlass_moe_mm_data(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        e,
        n,
        k,
        blockscale_offsets,
        is_gated=is_gated,
    )
'''
NEW = '''    # Note that problem sizes are based on logical number of elements.
    if not _reax_moe_data(topk_ids, expert_offsets, problem_sizes1, problem_sizes2, a_map, c_map, e, n, k,
                          blockscale_offsets, is_gated):
        ops.get_cutlass_moe_mm_data(
            topk_ids,
            expert_offsets,
            problem_sizes1,
            problem_sizes2,
            a_map,
            c_map,
            e,
            n,
            k,
            blockscale_offsets,
            is_gated=is_gated,
        )
'''
ADD = '''

# ---------------------------------------------------------------- engv2: MoE dispatch metadata in one launch (moe_data.py)
import ctypes as _reax_md_ctypes
import os as _reax_md_os

_REAX_MD = {"on": None, "lib": None}


def _reax_md_lib():
    st = _REAX_MD
    if st["lib"] is None:
        dll = _reax_md_ctypes.PyDLL if _reax_md_os.environ.get("REAX_PYDLL", "0") == "1" else _reax_md_ctypes.CDLL
        lib = dll(_reax_md_os.environ.get("REAX_MOE_DATA_LIB", "/opt/engv2/libreax_moedata.so"))
        lib.reax_moe_data_launch.restype = _reax_md_ctypes.c_int
        lib.reax_moe_data_launch.argtypes = [_reax_md_ctypes.c_int] * 3 + [_reax_md_ctypes.c_void_p] * 7 + [
            _reax_md_ctypes.c_int] * 3 + [_reax_md_ctypes.c_void_p]
        st["lib"] = lib
    return st["lib"]


def _reax_moe_data_raw(topk_ids, eo, p1, p2, am, cm, e, n, k, bo, gated):
    rc = _reax_md_lib().reax_moe_data_launch(
        topk_ids.numel(), topk_ids.shape[-1], e, topk_ids.data_ptr(), eo.data_ptr(), p1.data_ptr(), p2.data_ptr(),
        am.data_ptr(), cm.data_ptr(), bo.data_ptr(), n, k, 1 if gated else 0, torch.cuda.current_stream().cuda_stream)
    if rc != 0:
        raise RuntimeError(f"reax_moe_data_launch: error {rc}")


def _reax_md_bufs(t, e):
    i32 = dict(dtype=torch.int32, device="cuda")
    return (torch.empty(e + 1, **i32), torch.empty((e, 3), **i32), torch.empty((e, 3), **i32),
            torch.empty(t, **i32), torch.empty(t, **i32), torch.empty(e + 1, **i32))


def _reax_md_consistent(topk_ids, ref, new):
    """Offsets / problem sizes equal; maps a valid dispatch for these topk ids. Returns '' or what differs."""
    eo, p1, p2, am, cm, bo = new
    for name, a, b in (("expert_offsets", ref[0], eo), ("problem_sizes1", ref[1], p1), ("problem_sizes2", ref[2], p2),
                       ("blockscale_offsets", ref[5], bo)):
        if not torch.equal(a, b):
            return name
    flat = topk_ids.reshape(-1).long()
    t, topk = flat.numel(), topk_ids.shape[-1]
    c = cm.long()
    if not torch.equal(torch.sort(c).values, torch.arange(t, device=c.device)):
        return "c_map not a permutation"
    lo, hi = eo[:-1].long()[flat], eo[1:].long()[flat]
    if not bool(((c >= lo) & (c < hi)).all()):
        return "c_map outside its expert"
    if not torch.equal(am.long()[c], torch.arange(t, device=c.device) // topk):
        return "a_map[c_map[i]] != i // topk"
    return ""


def _reax_md_routings(g, e=128, topk=8, small=False):
    ms = (1, 3, 8, 64, 887) if small else (1, 2, 3, 7, 8, 16, 31, 64, 100, 128, 200, 333, 512, 887, 1000, 1024, 2048, 4096)
    for m in ms:
        for kind in range(4):
            if kind == 0:  # uniform
                ids = torch.randn(m, e, device="cuda", generator=g).topk(topk, dim=-1).indices
            elif kind == 1:  # a few hot experts (real routing has experts with 450-730 rows)
                lg = torch.randn(m, e, device="cuda", generator=g)
                lg[:, :8] += 2.5
                ids = lg.topk(topk, dim=-1).indices
            elif kind == 2:  # vLLM's dummy runs: every row on the same experts
                ids = torch.arange(topk, device="cuda").expand(m, topk)
            else:  # unused experts
                lg = torch.randn(m, e, device="cuda", generator=g)
                lg[:, e // 2:] -= 50.0
                ids = lg.topk(topk, dim=-1).indices
            yield m, kind, ids.to(torch.int32).contiguous()


def _reax_moe_data_check(small=False):
    g = torch.Generator(device="cuda").manual_seed(2810)
    e, n, k = 128, 704, 2816
    for m, kind, ids in _reax_md_routings(g, e, small=small):
        for gated in (True, False):
            ref = _reax_md_bufs(ids.numel(), e)
            ops.get_cutlass_moe_mm_data(ids, ref[0], ref[1], ref[2], ref[3], ref[4], e, n, k, ref[5], is_gated=gated)
            new = _reax_md_bufs(ids.numel(), e)
            _reax_moe_data_raw(ids, *new[:5], e, n, k, new[5], gated)
            bad = _reax_md_consistent(ids, ref, new)
            if bad:
                print(f"reax moe_data: m {m} routing {kind} gated {gated}: {bad}", flush=True)
                return False
    return True


def _reax_moe_data(topk_ids, expert_offsets, problem_sizes1, problem_sizes2, a_map, c_map, e, n, k, blockscale_offsets,
                   is_gated):
    """vLLM's get_cutlass_moe_mm_data outputs (maps up to the in-expert order) in one launch; False = not handled."""
    st = _REAX_MD
    if st["on"] is None:
        if torch.cuda.is_current_stream_capturing():
            return False
        st["on"] = False
        if _reax_md_os.environ.get("REAX_MOE_DATA", "0") == "1":
            try:
                st["on"] = _reax_moe_data_check(small=True)
                how = "checked in-engine"
            except Exception as ex:  # noqa: BLE001 - keep vLLM's op
                how = f"disabled ({ex!r})"
            print(f"reax moe_data: {'on' if st['on'] else 'off'} ({how})", flush=True)
    if (not st["on"] or e != 128 or blockscale_offsets is None or topk_ids.dtype != torch.int32
            or not topk_ids.is_contiguous()):
        return False
    _reax_moe_data_raw(topk_ids, expert_offsets, problem_sizes1, problem_sizes2, a_map, c_map, e, n, k,
                       blockscale_offsets, is_gated)
    return True
'''


def patch(path):
    src = open(path).read()
    assert src.count(OLD) == 1, f"cutlass_moe.py: get_cutlass_moe_mm_data anchor found {src.count(OLD)} times"
    open(path, "w").write(src.replace(OLD, NEW) + ADD)
    print("moe_data: patched")


def test():
    import torch
    from triton.testing import do_bench_cudagraph
    from vllm import _custom_ops as ops
    ns = {"torch": torch, "ops": ops}
    exec(ADD, ns)
    ok = ns["_reax_moe_data_check"](small=False)
    print(f"moe_data: consistency vs vLLM's op on 18 m x 4 routings x gated/plain: {'PASS' if ok else 'FAIL'}")
    g = torch.Generator(device="cuda").manual_seed(1)
    for m in (440, 887, 1000, 2048):
        lg = torch.randn(m, 128, device="cuda", generator=g)
        lg[:, :8] += 2.5
        ids = lg.topk(8, dim=-1).indices.to(torch.int32).contiguous()
        a = ns["_reax_md_bufs"](ids.numel(), 128)
        b = ns["_reax_md_bufs"](ids.numel(), 128)
        t0 = do_bench_cudagraph(lambda: ops.get_cutlass_moe_mm_data(ids, a[0], a[1], a[2], a[3], a[4], 128, 704, 2816, a[5],
                                                                    is_gated=True), rep=200)
        t1 = do_bench_cudagraph(lambda: ns["_reax_moe_data_raw"](ids, *b[:5], 128, 704, 2816, b[5], True), rep=200)
        print(f"m {m:5d}: vLLM {t0 * 1e3:5.1f} us -> one launch {t1 * 1e3:5.1f} us")


if __name__ == "__main__":
    if sys.argv[1] == "test":
        test()
    else:
        patch(sys.argv[1])
