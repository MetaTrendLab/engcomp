from __future__ import annotations

import os
import sys

import torch

ON = os.environ.get("Q35_NOSYNC", "0") == "1"
_HOST = {}
_MEMO = {}
HELPERS = ("prepare_position_ids", "prepare_sequence_ids", "prepare_token_indices", "prepare_chunk_indices",
           "get_max_num_splits")
_installed = False


def _h2d(t, dev):
    return t.pin_memory().to(dev, non_blocking=True)


def T_(dev):
    import numpy as np
    _HOST.clear()
    _MEMO.clear()

    def t(a, dt=torch.int64):
        h = torch.from_numpy(np.ascontiguousarray(a, dtype=np.int64))
        d = _h2d(h, dev)
        if dt != torch.int64:
            return d.to(dt)
        _HOST[id(d)] = (d, h)
        return d
    return t


def h2d(t, dev):
    return _h2d(t, dev)


def _wrap(name, orig):
    def f(cu_seqlens, *a, **kw):
        e = _HOST.get(id(cu_seqlens))
        if e is None or e[0] is not cu_seqlens or any(isinstance(x, torch.Tensor) for x in a):
            return orig(cu_seqlens, *a, **kw)
        key = (name, id(cu_seqlens), a, tuple(sorted((k, v) for k, v in kw.items() if k != "cu_seqlens_cpu")))
        r = _MEMO.get(key)
        if r is None:
            kw2 = {k: v for k, v in kw.items() if k != "cu_seqlens_cpu"}
            kw2["cu_seqlens_cpu"] = e[1]
            r = orig(e[1], *a, **kw2)
            if isinstance(r, torch.Tensor):
                r = _h2d(r, cu_seqlens.device)
            _MEMO[key] = r
        return r
    f.__wrapped_q35__ = orig
    return f


def install():
    global _installed
    if _installed or not ON:
        return
    import fla.modules.conv.causal_conv1d
    import fla.ops.gated_delta_rule
    import fla.ops.utils.index as ix
    origs = {n: getattr(ix, n) for n in HELPERS}
    news = {n: _wrap(n, o) for n, o in origs.items()}
    hits = 0
    for mname, mod in list(sys.modules.items()):
        if not (mname == "fla" or mname.startswith("fla.")) or mod is None:
            continue
        for n, o in origs.items():
            if getattr(mod, n, None) is o:
                setattr(mod, n, news[n])
                hits += 1
    _installed = True
    print(f"q35 nosync: FLA index helpers {list(HELPERS)} from host copies ({hits} bindings)", flush=True)
