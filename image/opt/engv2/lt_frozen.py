import collections
import hashlib
import os
import subprocess
import sys

import torch


def _key(p):
    if p is None or isinstance(p, bool):
        return ("x", p)
    if isinstance(p, (str, tuple)):
        return ("s", p)
    return ("L", p.numpy().tobytes())


def _tag(p):
    if p is None:
        return "p"
    if isinstance(p, bool):
        return "1" if p else "0"
    return "b" if isinstance(p, str) else "t" if isinstance(p, tuple) else "L"


def _h(p):
    return hashlib.sha1(p.numpy().tobytes()).hexdigest()[:10]


def _sections(kind, d):
    if kind == "vis":
        return [("vis", d)]
    return [("small", d["small"]), ("large", d["large"]), ("head", {"head": [d.get("head")]})]


def _build(kind, secs):
    if kind == "vis":
        return secs["vis"]
    return {"small": secs["small"], "large": secs["large"], "head": secs["head"].get("head", [None])[0]}


def show(kind, paths):
    for path in paths:
        d = torch.load(path)
        parts = []
        for name, tab in _sections(kind, d):
            for shape, picks in sorted(tab.items(), key=str):
                lts = sorted({_h(p) for p in picks if _tag(p) == "L"})
                label = shape if shape == "head" else f"{shape[0]}x{shape[1]}"
                parts.append(f"{name}:{label}:{''.join(_tag(p) for p in picks)}" + (f":{','.join(lts)}" if lts else ""))
        print(f"{os.path.basename(path)}: {' '.join(parts)}", flush=True)


def merge(kind, out, paths):
    tabs = [dict(_sections(kind, torch.load(p))) for p in paths]
    secs = {}
    for name in tabs[0]:
        merged = {}
        shapes = collections.Counter(s for t in tabs for s in t[name])
        for shape, n in shapes.items():
            if 2 * n <= len(tabs):
                continue
            rows = [t[name][shape] for t in tabs if shape in t[name]]
            picks = []
            for b in range(len(rows[0])):
                votes, obj = collections.Counter(), {}
                for r in rows:
                    k = _key(r[b])
                    votes[k] += 1
                    obj.setdefault(k, r[b])
                picks.append(obj[votes.most_common(1)[0][0]])
            merged[shape] = picks
        secs[name] = merged
    torch.save(_build(kind, secs), out)
    print(f"lt_frozen {kind}: majority of {len(paths)} derives -> {out}", flush=True)


def _bypass_ref(bi, x, w):
    bi._REAX_BYPASS[0] = True
    try:
        return bi.matmul_persistent(x, w.t())
    finally:
        bi._REAX_BYPASS[0] = False


def _uniq(picks, skip=lambda p: p is None):
    seen, out = set(), []
    for p in picks:
        if not skip(p) and _key(p) not in seen:
            seen.add(_key(p))
            out.append(p)
    return out


def check(kind, frozen, out):
    import reax_lt
    from vllm.model_executor.determinism import batch_invariant as bi
    d = torch.load(frozen)
    g = torch.Generator(device="cuda").manual_seed(4401)
    bad = []
    if kind == "vis":
        bi._REAX_VIS["mm"] = reax_lt.mm
        for (N, K), picks in d.items():
            w = (torch.randn(N, K, device="cuda", generator=g) * 0.02).bfloat16()
            X = torch.randn(max(bi._REAX_VIS_PROBE), K, device="cuda", generator=g).bfloat16()
            cands = _uniq(picks)
            for M in bi._REAX_VIS_PROBE:
                xm = X[:M].contiguous()
                ref = _bypass_ref(bi, xm, w)
                for p in list(cands):
                    try:
                        ok = torch.equal(bi._reax_vis_run(xm, w, p), ref)
                    except RuntimeError:
                        ok = False
                    if not ok:
                        bad.append(f"vis {N}x{K} {_tag(p)} M {M}")
                        cands.remove(p)
            del w, X
    else:
        lt_large = bi._reax_lt_algos()
        bi._REAX_LT_STATE.setdefault("mm", reax_lt.mm)
        for (N, K), picks in d["small"].items():
            w = (torch.randn(N, K, device="cuda", generator=g) * 0.02).bfloat16()
            X = torch.randn(1023, K, device="cuda", generator=g).bfloat16()
            cands = _uniq(picks)
            for M in range(1, 1024):
                xm = X[:M].contiguous()
                ref = _bypass_ref(bi, xm, w)
                for p in list(cands):
                    try:
                        ok = torch.equal(bi._reax_pick_mm(xm, w, p), ref)
                    except RuntimeError:
                        ok = False
                    if not ok:
                        bad.append(f"small {N}x{K} {_tag(p)} M {M}")
                        cands.remove(p)
            del w, X
        for (N, K), use in d["large"].items():
            if False in use and (N, K) not in lt_large:
                bad.append(f"large {N}x{K}: LT pin missing on this card")
            bfl = _uniq(use, skip=lambda p: not isinstance(p, str))
            if bfl:
                w = (torch.randn(N, K, device="cuda", generator=g) * 0.02).bfloat16()
                xl = torch.randn(4099, K, device="cuda", generator=g).bfloat16()
                for M in (1024, 1100, 1537, 2049, 2560, 4099):
                    xm = xl[:M].contiguous()
                    ref = bi._reax_td_triton(xm, w)
                    for p in list(bfl):
                        if not torch.equal(bi._reax_pick_mm(xm, w, p), ref):
                            bad.append(f"large {N}x{K} bf16g M {M}")
                            bfl.remove(p)
                del w, xl
        head = d.get("head")
        if head is not None:
            N, K = bi._REAX_HEAD_SHAPE
            w = (torch.randn(N, K, device="cuda", generator=g) * 0.02).bfloat16()
            X = torch.randn(64, K, device="cuda", generator=g).bfloat16()
            for M in range(1, 65):
                xm = X[:M].contiguous()
                try:
                    ok = torch.equal(reax_lt.mm(xm, w, head), _bypass_ref(bi, xm, w))
                except RuntimeError:
                    ok = False
                if not ok:
                    bad.append(f"head M {M}")
                    break
            del w, X
    if bad:
        print(f"lt_frozen {kind}: {frozen} NOT exact on this card ({len(bad)}: {', '.join(bad[:4])}) -> live derive",
              flush=True)
        sys.exit(1)
    torch.save(d, out)
    print(f"lt_frozen {kind}: {frozen} exact on this card -> {out}", flush=True)


def derive_n(kind, outdir, n):
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lt_small4.py" if kind == "small" else "lt_vis.py")
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for i in range(n):
        p = os.path.join(outdir, f"{kind}-{i}.pt")
        subprocess.run([sys.executable, script, "derive", p], check=True)
        paths.append(p)
    merge(kind, os.path.join(outdir, f"{kind}-frozen.lt"), paths)
    show(kind, paths + [os.path.join(outdir, f"{kind}-frozen.lt")])


if __name__ == "__main__":
    kind, mode = sys.argv[1], sys.argv[2]
    assert kind in ("small", "vis"), kind
    if mode == "check":
        check(kind, sys.argv[3], sys.argv[4])
    elif mode == "merge":
        merge(kind, sys.argv[3], sys.argv[4:])
    elif mode == "show":
        show(kind, sys.argv[3:])
    elif mode == "derive-n":
        derive_n(kind, sys.argv[3], int(sys.argv[4]))
    else:
        sys.exit(f"unknown mode {mode}")
