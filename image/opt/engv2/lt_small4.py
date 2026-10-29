import sys

OLD = '''    if a.shape[0] >= 1024:
        return _reax_td_eager(a, w.t())
    _REAX_BYPASS[0] = True'''
NEW = '''    if a.shape[0] >= 1024:
        p = _reax_large_pick(w.shape[0], w.shape[1], a.shape[0])
        if isinstance(p, str):
            return _reax_bf16g()(a, w, p[6:])
        if p:
            return _reax_td_triton(a, w)
        return _reax_td_eager(a, w.t())
    sm = _reax_lt_small().get((w.shape[0], w.shape[1])) if a.is_contiguous() and w.is_contiguous() else None
    if sm is not None:
        return _reax_pick_mm(a, w, sm[(a.shape[0] - 1) >> 5])
    _REAX_BYPASS[0] = True'''
OLD2 = '''    if _reax_td_ok(a, b, bias):
        return _reax_td(a, b)
'''
NEW2 = OLD2 + '''    if (_REAX_HEAD_ON and bias is None and not _REAX_BYPASS[0] and a.shape[0] <= 64
            and (b.shape[1], b.shape[0]) == _REAX_HEAD_SHAPE):
        _head = _reax_lt_head(a, b)
        if _head is not None:
            return _head
'''
ADD = '''

# ---------------------------------------------------------------- engv2 b7lt3: small-M cuBLASLt pins per M bucket + large-M choice
_REAX_SMALL = {"algos": None, "large": {}, "head": None}
_REAX_HEAD_ON = os.environ.get("REAX_LT_HEAD", "1") == "1"
_REAX_HEAD_SHAPE = (262144, 2816)  # Blink lm_head (tied embedding): (vocab, hidden)
_REAX_LARGE_EDGES = (1152, 1536, 2560, 1 << 30)


_REAX_FLUSH = []
# Triton TMA tiles (BLOCK M, N, K, warps, stages) that compete with the exact LT algorithms in the small-M buckets: a
# tl.dot over K is a chain of m16n8k16 steps in K order whatever the tile, so all are bit-identical to the pointer kernel
# (PRO 6000 dense_tune: 72/72 configs; still checked at every M here); 2816x8192 (o_proj of the full-attention layers) at
# M 880: TD 150.5 vs the LT pin 185.3 us on the PRO 6000
_REAX_TD_TILES = ((128, 128, 64, 8, 3), (64, 128, 64, 4, 4), (128, 64, 64, 4, 4), (64, 64, 64, 4, 4))


_REAX_BF16G = [None]


def _reax_bf16g():
    """bf16g (tools/engv2/blink/bf16g, PRO 6000 session): exact bf16 TN GEMM (sequential m16n8k16 K chain from zero, the
    pointer kernel's arithmetic) with plain and wave-tail hybrid tilings, launched through the driver API."""
    if _REAX_BF16G[0] is None:
        import reax_bf16g
        _REAX_BF16G[0] = reax_bf16g.BF16Gemm()
    return _REAX_BF16G[0]


def _reax_bf16g_cfgs(N, K):
    if os.environ.get("REAX_LT_SMALL_BF16G", "1") != "1":
        return []
    try:
        g = _reax_bf16g()
    except Exception as e:  # noqa: BLE001 - no cubin / driver: LT and TD only
        print(f"reax_lt: bf16g unavailable ({e!r})", flush=True)
        return []
    return ["bf16g:" + c for c in g.fn if g.supports(c, N, K)]


def _reax_pick_mm(a, w, p):
    """p = an exact LT algorithm (tensor), a Triton TMA tile (tuple) or "bf16g:<cfg>"."""
    if isinstance(p, str):
        return _reax_bf16g()(a, w, p[6:])
    if isinstance(p, tuple):
        return _reax_td_tile(a, w, *p)
    return _REAX_LT_STATE["mm"](a, w, p)


def _reax_td_tile(a, w, bm, bn, bk, warps, stages):
    if not _REAX_TD_ALLOC[0]:
        triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
        _REAX_TD_ALLOC[0] = True
    M, K = a.shape
    N = w.shape[0]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    nsms = num_compute_units(a.device.index)
    grid = (min(nsms, triton.cdiv(M, bm) * triton.cdiv(N, bn)),)
    matmul_kernel_descriptor_persistent[grid](a, w, c, None, M, N, K, BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
                                              GROUP_SIZE_M=8, NUM_SMS=nsms, HAS_BIAS=False, num_warps=warps,
                                              num_stages=stages)
    return c


def _reax_time(fn, reps=12):
    """Median of single calls, each after a 256 MB write that evicts L2 (as triton's do_bench)."""
    if not _REAX_FLUSH:
        _REAX_FLUSH.append(torch.empty(256 << 20, dtype=torch.int8, device="cuda"))
    fn()
    ts = []
    for _ in range(reps):
        _REAX_FLUSH[0].zero_()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    ts.sort()
    return ts[len(ts) // 2]


def _reax_lt_small_derive():
    """{"small": (N, K) -> [algo per 32-row bucket of M], "large": (N, K) -> [use TD per large bucket]}."""
    import hashlib
    import reax_lt
    h = lambda a: hashlib.sha1(a.numpy().tobytes()).hexdigest()[:10]  # noqa: E731
    g = torch.Generator(device="cuda").manual_seed(4322)
    small, large, log = {}, {}, []
    lt_large = _reax_lt_algos()
    for (N, K) in _REAX_TD_CFG:
        w = (torch.randn(N, K, device="cuda", generator=g) * 0.02).bfloat16()
        X = torch.randn(1023, K, device="cuda", generator=g).bfloat16()
        alive, seen = [], set()
        for hM in (130, 512, 900, 16):
            for algo, _cfg in reax_lt.algos(hM, N, K, 64):
                if h(algo) not in seen:
                    seen.add(h(algo))
                    alive.append(algo)
        n_cand = len(alive)
        tiles = [] if os.environ.get("REAX_LT_SMALL_TD", "1") != "1" else list(_REAX_TD_TILES)
        n_tiles = len(tiles)
        bfs = _reax_bf16g_cfgs(N, K)
        n_bfs = len(bfs)
        for M in range(1, 1024):
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
            tiles = [c for c in tiles if torch.equal(_reax_td_tile(xm, w, *c), ref)]
            bfs = [c for c in bfs if torch.equal(_reax_pick_mm(xm, w, c), ref)]
            if not alive:
                break
        if alive:
            pick = []
            for b in range(32):
                xm = X[:min(32 * (b + 1), 1023)].contiguous()
                cands = alive + tiles + bfs
                t = [_reax_time(lambda: _reax_pick_mm(xm, w, a)) for a in cands]
                pick.append(cands[t.index(min(t))])
            small[(N, K)] = pick
            n_td = sum(isinstance(p, tuple) for p in pick)
            n_bf = sum(isinstance(p, str) for p in pick)
            log.append(f"{N}x{K}:{len(alive)}/{n_cand}exact:"
                       + ",".join(sorted({h(a) for a in pick if not isinstance(a, (tuple, str))}))
                       + f":td{len(tiles)}/{n_tiles}exact,{n_td}/32buckets:bf16g{len(bfs)}/{n_bfs}exact,{n_bf}/32buckets")
        if (N, K) in lt_large:
            xl = torch.randn(4099, K, device="cuda", generator=g).bfloat16()
            # bf16g at large M: bit-identical to the TMA kernel (= the pointer kernel's bits, b3) at odd and even M
            bfl = _reax_bf16g_cfgs(N, K)
            for M in (1024, 1100, 1537, 2049, 2560, 4099):
                xm = xl[:M].contiguous()
                ref = _reax_td_triton(xm, w)
                bfl = [c for c in bfl if torch.equal(_reax_pick_mm(xm, w, c), ref)]
            use = []
            for M in (1152, 1536, 2560, 4096):
                xm = xl[:M].contiguous()
                cands = [True, False] + bfl
                t = [_reax_time(lambda: _reax_td_triton(xm, w)) if c is True
                     else _reax_time(lambda: _REAX_LT_STATE["mm"](xm, w, lt_large[(N, K)])) if c is False
                     else _reax_time(lambda: _reax_pick_mm(xm, w, c)) for c in cands]
                use.append(cands[t.index(min(t))])
            large[(N, K)] = use
            log.append("large:" + "".join("b" if isinstance(u, str) else "1" if u else "0" for u in use))
            del xl
        del w, X
    head = None
    if _REAX_HEAD_ON:
        N, K = _REAX_HEAD_SHAPE
        w = (torch.randn(N, K, device="cuda", generator=g) * 0.02).bfloat16()
        X = torch.randn(64, K, device="cuda", generator=g).bfloat16()
        alive, seen = [], set()
        for hM in (8, 16, 32, 1):
            for algo, _cfg in reax_lt.algos(hM, N, K, 64):
                if h(algo) not in seen:
                    seen.add(h(algo))
                    alive.append(algo)
        n_cand = len(alive)
        t_ptr = 0.0
        for M in range(1, 65):
            _REAX_BYPASS[0] = True
            try:
                ref = matmul_persistent(X[:M], w.t())
                if M == 16:
                    t_ptr = _reax_time(lambda: matmul_persistent(X[:16], w.t()))
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
            if not alive:
                break
        if alive:
            xm = X[:16].contiguous()
            t = [_reax_time(lambda: reax_lt.mm(xm, w, a)) for a in alive]
            head = alive[t.index(min(t))]
            log.append(f"head{N}x{K}:{len(alive)}/{n_cand}exact:{h(head)}:{t_ptr / min(t):.2f}x")
        else:
            log.append(f"head{N}x{K}:0/{n_cand}exact")
        del w, X
    print(f"reax_lt small-M (bit-exact at every M 1..1023, fastest per 32-row bucket, L2 flushed): {' '.join(log)}", flush=True)
    return {"small": small, "large": large, "head": head}


def _reax_lt_small():
    st = _REAX_SMALL
    if st["algos"] is not None:
        return st["algos"]
    if torch.cuda.is_current_stream_capturing():  # never derive inside a CUDA graph capture (eager warm-up runs first)
        return {}
    st["algos"] = {}
    if os.environ.get("REAX_LT_SMALL", "1") != "1":
        return st["algos"]
    try:
        import reax_lt
        _reax_lt_algos()
        _REAX_LT_STATE.setdefault("mm", reax_lt.mm)
        path = os.environ.get("REAX_LT_SMALL_FILE", "")
        if path and os.path.exists(path):
            d = torch.load(path)
            print(f"reax_lt small-M: tables for {len(d['small'])} shapes loaded from {path}", flush=True)
        else:
            d = _reax_lt_small_derive()
        st["algos"] = {tuple(k): v for k, v in d["small"].items()}
        st["large"] = {tuple(k): v for k, v in d["large"].items()}
        st["head"] = d.get("head")
    except Exception as e:  # noqa: BLE001 - keep the batch-invariant pointer kernel
        print(f"reax_lt small-M: disabled ({e!r})", flush=True)
        st["algos"] = {}
    return st["algos"]


def _reax_lt_head(a, b):
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16 or not a.is_contiguous() or a.shape[0] == 0:
        return None
    w = b.t()
    if not w.is_contiguous() or torch.cuda.is_current_stream_capturing():
        return None
    _reax_lt_small()
    algo = _REAX_SMALL["head"]
    if algo is None:
        return None
    return _REAX_LT_STATE["mm"](a, w, algo)


def _reax_large_pick(N, K, M):
    """True = the Triton TMA kernel, False = the pinned LT algorithm, "bf16g:<cfg>" = bf16g."""
    if os.environ.get("REAX_LARGE_PICK", "1") != "1":
        return False
    _reax_lt_small()
    use = _REAX_SMALL["large"].get((N, K))
    if use is None:
        return False
    for edge, u in zip(_REAX_LARGE_EDGES, use):
        if M < edge:
            return u
    return use[-1]
'''

if sys.argv[1] == "derive":
    import torch
    from vllm.model_executor.determinism import batch_invariant as bi
    torch.save(bi._reax_lt_small_derive(), sys.argv[2])
    sys.exit(0)
path = sys.argv[1]
src = open(path).read()
assert src.count(OLD) == 1, "b3 dense_mm hunk not found"
assert src.count(OLD2) == 1, "matmul_persistent routing hunk not found"
src = src.replace(OLD, NEW).replace(OLD2, NEW2) + ADD
open(path, "w").write(src)
print("lt_small4: installed")
