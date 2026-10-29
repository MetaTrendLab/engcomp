from __future__ import annotations

import json
import os

SPEC = os.environ.get("Q35_FLA_PIN", "")
KERNELS = {
    "l2norm_fwd": ("fla.modules.l2norm", "l2norm_fwd_kernel"),
    "cumsum": ("fla.ops.utils.cumsum", "chunk_local_cumsum_scalar_kernel"),
    "kkt_solve": ("fla.ops.gated_delta_rule.chunk_fwd", "chunk_gated_delta_rule_fwd_kkt_solve_kernel"),
    "w_u": ("fla.ops.gated_delta_rule.wy_fast", "recompute_w_u_fwd_kernel"),
    "fwd_h": ("fla.ops.common.chunk_delta_h", "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"),
    "fwd_o": ("fla.ops.common.chunk_o", "chunk_fwd_kernel_o"),
}


def parse_cfg(s: str):
    import triton

    kw, nw, ns = {}, 4, 3
    for part in s.split(","):
        part = part.strip()
        if part.startswith("w") and part[1:].isdigit():
            nw = int(part[1:])
        elif part.startswith("s") and part[1:].isdigit():
            ns = int(part[1:])
        elif part:
            k, v = part.split(":")
            kw[k] = int(v)
    return triton.Config(kw, num_warps=nw, num_stages=ns)


def install(spec: str = SPEC) -> dict:
    if not spec:
        return {}
    import importlib

    pins = json.load(open(spec)) if os.path.isfile(spec) else json.loads(spec)
    done = {}
    for name, cfg in pins.items():
        mod, attr = KERNELS[name]
        at = getattr(importlib.import_module(mod), attr)
        while not hasattr(at, "configs"):
            at = at.fn
        c = parse_cfg(cfg)
        if not any(o.kwargs == c.kwargs and o.num_warps == c.num_warps and o.num_stages == c.num_stages for o in at.configs):
            raise ValueError(f"Q35_FLA_PIN {name}={cfg}: not one of fla's own configs")
        at.configs = [c]
        at.cache.clear()
        done[name] = cfg
    print(f"q35_flapin: {done}", flush=True)
    return done


PINNED = install()
