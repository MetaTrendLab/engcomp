import sys

OLD = '''    if _reax_td_ok(a, b, bias):
        return _reax_td(a, b)
    NUM_SMS = num_compute_units(a.device.index)'''
NEW = '''    if _reax_td_ok(a, b, bias):
        return _reax_td(a, b)
    if _REAX_VIS_ON and bias is None and not _REAX_BYPASS[0] and (b.shape[1], b.shape[0]) in _REAX_VIS_SHAPES:
        _vis = _reax_vis(a, b)
        if _vis is not None:
            return _vis
    NUM_SMS = num_compute_units(a.device.index)'''
ADD = '''

# ---------------------------------------------------------------- engv2 v4: vision tower dense GEMMs through pinned cuBLASLt
_REAX_VIS_ON = os.environ.get("REAX_LT_VIS", "1") == "1"
_REAX_VIS_SHAPES = {(1152, 1152), (4304, 1152), (1152, 4304), (1152, 768), (2816, 1152)}
_REAX_VIS = {"algos": None, "mm": None}
_REAX_VIS_PROBE = list(range(1, 257)) + [383, 511, 512, 513, 640, 1000, 1023, 1024, 1025, 2047, 2048, 2394, 2520, 3001,
                                         4099, 4788, 5040, 7560, 9576, 10080, 14363, 19152]
_REAX_VIS_EDGES = (3000, 6000, 12000, 1 << 30)
_REAX_VIS_TIME_M = (2394, 4788, 9576, 19152)


def _reax_vis_run(a, w, p):
    """p = an exact LT algorithm (tensor) or "bf16g:<cfg>"."""
    if isinstance(p, str):
        return _reax_bf16g()(a, w, p[6:])
    return _REAX_VIS["mm"](a, w, p)


def _reax_vis_derive():
    import hashlib
    import reax_lt
    _REAX_VIS["mm"] = reax_lt.mm
    g = torch.Generator(device="cuda").manual_seed(4323)
    out, log = {}, []
    for (N, K) in sorted(_REAX_VIS_SHAPES):
        w = (torch.randn(N, K, device="cuda", generator=g) * 0.02).bfloat16()
        X = torch.randn(max(_REAX_VIS_PROBE), K, device="cuda", generator=g).bfloat16()
        alive, seen = [], set()
        for hM in (130, 1024, 2520, 10080):
            for algo, _cfg in reax_lt.algos(hM, N, K, 64):
                key = algo.numpy().tobytes()
                if key not in seen:
                    seen.add(key)
                    alive.append(algo)
        n_cand = len(alive)
        bfs = []
        if os.environ.get("REAX_LT_VIS_BF16G", "1") == "1":
            try:
                bg = _reax_bf16g()
                bfs = ["bf16g:" + c for c in bg.fn if bg.supports(c, N, K)]
            except Exception as e:  # noqa: BLE001 - no cubin / driver: LT only
                log.append(f"bf16g-unavailable({type(e).__name__})")
        n_bfs = len(bfs)
        for M in _REAX_VIS_PROBE:
            _REAX_BYPASS[0] = True
            try:
                ref = matmul_persistent(X[:M], w.t())
            finally:
                _REAX_BYPASS[0] = False
            xm = X[:M].contiguous()
            keep = []
            for algo in alive:
                try:
                    if torch.equal(reax_lt.mm(xm, w, algo), ref):
                        keep.append(algo)
                except RuntimeError:
                    pass
            alive = keep
            bfs = [c for c in bfs if torch.equal(_reax_vis_run(xm, w, c), ref)]
            if not alive and not bfs:
                break
        cands = alive + bfs
        if cands:
            picks, gain = [], []
            for M in _REAX_VIS_TIME_M:
                xm = X[:M].contiguous()
                _REAX_BYPASS[0] = True
                try:
                    t_ptr = _reax_time(lambda: matmul_persistent(xm, w.t()))
                finally:
                    _REAX_BYPASS[0] = False
                ts = [_reax_time(lambda: _reax_vis_run(xm, w, c)) for c in cands]
                t_lt = min(ts[:len(alive)]) if alive else float("inf")
                # the pointer kernel itself competes too (None = fall through to matmul_persistent)
                picks.append(cands[ts.index(min(ts))] if min(ts) < t_ptr else None)
                gain.append(f"{t_ptr / min(ts + [t_ptr]):.2f}({t_lt / min(ts + [t_ptr]):.2f})")
            out[(N, K)] = picks
            tags = "".join("p" if p is None else "b" if isinstance(p, str) else "L" for p in picks)
            lts = sorted({hashlib.sha1(p.numpy().tobytes()).hexdigest()[:10] for p in picks
                          if p is not None and not isinstance(p, str)})
            log.append(f"{N}x{K}:{len(alive)}/{n_cand}LT,{len(bfs)}/{n_bfs}bf16g:{tags}:{','.join(lts)}:"
                       f"{'/'.join(gain)}x")
        else:
            log.append(f"{N}x{K}:0/{n_cand}")
        del w, X
    print(f"reax_lt vision: {' '.join(log)}", flush=True)
    return out


def _reax_vis_algos():
    st = _REAX_VIS
    if st["algos"] is not None:
        return st["algos"]
    st["algos"] = {}
    try:
        import reax_lt
        st["mm"] = reax_lt.mm
        path = os.environ.get("REAX_LT_VIS_FILE", "")
        if path and os.path.exists(path):
            st["algos"] = {tuple(k): v for k, v in torch.load(path).items()}
            print(f"reax_lt vision: {len(st['algos'])} pins loaded from {path}", flush=True)
        else:
            st["algos"] = _reax_vis_derive()
    except Exception as e:  # noqa: BLE001 - keep the batch-invariant pointer kernel
        print(f"reax_lt vision: disabled ({e!r})", flush=True)
        st["algos"] = {}
    return st["algos"]


def _reax_vis(a, b):
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16 or not a.is_cuda or not a.is_contiguous():
        return None
    w = b.t()
    if not w.is_contiguous() or torch.cuda.is_current_stream_capturing():
        return None
    picks = _reax_vis_algos().get((w.shape[0], w.shape[1]))
    if picks is None:
        return None
    M = a.shape[0]
    p = next((p for e, p in zip(_REAX_VIS_EDGES, picks) if M < e), picks[-1])
    return None if p is None else _reax_vis_run(a, w, p)
'''

if sys.argv[1] == "derive":
    import torch
    from vllm.model_executor.determinism import batch_invariant as bi
    torch.save(bi._reax_vis_derive(), sys.argv[2])
    sys.exit(0)
path = sys.argv[1]
src = open(path).read()
assert src.count(OLD) == 1, "matmul_persistent routing hunk not found"
src = src.replace(OLD, NEW) + ADD
open(path, "w").write(src)
print("lt_vis: installed")
