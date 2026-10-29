import sys

OLD = '''def _reax_dense_mm(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
'''
NEW = OLD + '''    if _REAX_NEAR_ON and a.is_contiguous() and w.is_contiguous():
        _nr = _reax_near(w.shape[0], w.shape[1])
        if _nr is not None:
            return _REAX_NEAR_MM[0](a, w, _nr)
'''
ADD = '''

# ---------------------------------------------------------------- engv2 Blink: near-exact dense GEMMs (near_lt.py)
_REAX_NEAR_ON = os.environ.get("REAX_LT_NEAR", "0") == "1"
_REAX_NEAR = {}
_REAX_NEAR_MM = [None]


def _reax_near(N, K):
    if (N, K) not in _REAX_NEAR:
        _REAX_NEAR[(N, K)] = None  # the pick below times the exact table through this op
        if (N, K) in _REAX_TD_CFG:
            _REAX_NEAR[(N, K)] = _reax_near_pick(N, K)
    return _REAX_NEAR[(N, K)]


def _reax_near_pick(N, K):
    import hashlib
    import reax_lt
    _REAX_NEAR_MM[0] = reax_lt.mm
    g = torch.Generator(device="cuda").manual_seed(4324)
    w = (torch.randn(N, K, device="cuda", generator=g) * 0.02).bfloat16()
    X = torch.randn(4096, K, device="cuda", generator=g).bfloat16()
    cands, seen = [], set()
    for hM in (130, 512, 900, 2048):
        for algo, _cfg in reax_lt.algos(hM, N, K, 64):
            key = algo.numpy().tobytes()
            if key not in seen:
                seen.add(key)
                cands.append(algo)
    xs = {m: X[:m].contiguous() for m in (1, 7, 33, 100, 333, 645, 900, 1014, 1500, 2600)}
    ok = []
    for algo in cands:
        try:
            full = reax_lt.mm(X, w, algo)
            if all(torch.equal(reax_lt.mm(x, w, algo), full[:m]) for m, x in xs.items()):
                ok.append(algo)
        except RuntimeError:
            pass
    if not ok:
        print(f"reax near: {N}x{K}: no row-invariant algorithm of {len(cands)}, exact table kept", flush=True)
        return None
    t = [sum(_reax_time(lambda: reax_lt.mm(xs[m], w, a)) for m in (645, 900, 1014)) for a in ok]
    best = ok[t.index(min(t))]
    _REAX_BYPASS[0] = True
    try:
        ref = matmul_persistent(xs[900], w.t())
    finally:
        _REAX_BYPASS[0] = False
    t_ex = sum(_reax_time(lambda: _reax_dense_mm(xs[m], w)) for m in (645, 900, 1014))
    same = torch.equal(reax_lt.mm(xs[900], w, best), ref)
    print(f"reax near: {N}x{K}: {len(ok)}/{len(cands)} row-invariant, pick {hashlib.sha1(best.numpy().tobytes()).hexdigest()[:10]}"
          f" ({'= pointer bits' if same else 'differs from the pointer kernel'}), {min(t) * 1e3 / 3:.1f} us avg at"
          f" M 645/900/1014 vs exact table {t_ex * 1e3 / 3:.1f}", flush=True)
    return best
'''

path = sys.argv[1]
src = open(path).read()
assert src.count(OLD) == 1, "reax::dense_mm body not found"
open(path, "w").write(src.replace(OLD, NEW) + ADD)
print("near_lt: installed")
