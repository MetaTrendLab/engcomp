
import bisect
import importlib.abc
import importlib.util
import json
import os
import sys

_DIR = os.environ.get("REAX_JEFFAN_F8_DIR", "/opt/engv2/jf8")
_TABLE_PATH = os.environ.get("REAX_JEFFAN_F8_TABLE", os.path.join(_DIR, "jeffan_f8_table_pro6000.json"))


def _load_table():
    t = {}
    for nk, iv in json.load(open(_TABLE_PATH)).items():
        n, k = (int(v) for v in nk.split(":"))
        iv = sorted((int(a), int(b), c, int(gm)) for a, b, c, gm, *_ in iv)
        t[(n, k)] = ([b for _, b, _, _ in iv], [(a, (c, gm)) for a, _, c, gm in iv])
    return t


def _install(ops_mod):
    import torch

    table = _load_table()
    orig = ops_mod.cutlass_scaled_mm
    state = {"g": None, "sb": {}}

    def pick(n, k, m):
        t = table.get((n, k))
        if t is None:
            return None
        i = bisect.bisect_left(t[0], m)
        if i < len(t[1]) and t[1][i][0] <= m:
            return t[1][i][1]
        return None

    @torch.library.custom_op("reax::f8mm", mutates_args=())
    def f8mm(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
        m, k = a.shape
        n = b.shape[1]
        if state["g"] is None:
            if torch.cuda.is_current_stream_capturing():
                return orig(a, b, scale_a, scale_b, torch.bfloat16)
            sys.path.insert(0, _DIR)
            if os.environ.get("REAX_JEFFAN_F8SK") == "1":
                from f8sk import G6SK
                state["g"] = G6SK(os.path.join(_DIR, "f8g6.cubin"), os.path.join(_DIR, "f8sk.cubin"))
            else:
                from f8g6 import F8G6
                state["g"] = F8G6(os.path.join(_DIR, "f8g6.cubin"))
        p = pick(n, k, m) if a.is_contiguous() and scale_a.is_contiguous() else None
        if p is None:
            return orig(a, b, scale_a, scale_b, torch.bfloat16)
        key = (scale_b.data_ptr(), n)
        sbv = state["sb"].get(key)
        if sbv is None:
            sbv = state["sb"][key] = scale_b.reshape(1).expand(n).contiguous()
        out = torch.empty(m, n, device=a.device, dtype=torch.bfloat16)
        return state["g"](a, b.t(), scale_a.view(-1), sbv, cfg=p[0], gm=p[1], out=out)

    @f8mm.register_fake
    def _(a, b, scale_a, scale_b):
        return a.new_empty((a.shape[0], b.shape[1]), dtype=torch.bfloat16)

    def cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias=None):
        if (bias is None and out_dtype == torch.bfloat16 and a.dtype == torch.float8_e4m3fn and b.dtype == torch.float8_e4m3fn
                and b.dim() == 2 and b.stride(0) == 1 and scale_b.numel() == 1 and scale_a.dim() == 2
                and scale_a.shape[-1] == 1 and (b.shape[1], b.shape[0]) in table):
            return torch.ops.reax.f8mm(a, b, scale_a, scale_b)
        return orig(a, b, scale_a, scale_b, out_dtype, bias)

    ops_mod.cutlass_scaled_mm = cutlass_scaled_mm
    print(f"engv2 jeffan_f8: cutlass_scaled_mm -> f8g6 g9 for {sorted(table)} ({_TABLE_PATH})", flush=True)


class _Finder(importlib.abc.MetaPathFinder):

    def find_spec(self, name, path, target=None):
        if name != "vllm._custom_ops":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        exec_module = loader.exec_module

        def patched(module):
            exec_module(module)
            try:
                _install(module)
            except Exception as e:
                print(f"engv2 jeffan_f8: not installed ({e!r})", flush=True)

        loader.exec_module = patched
        return spec


if os.environ.get("REAX_JEFFAN_F8") == "1" and not any(isinstance(f, _Finder) for f in sys.meta_path):
    sys.meta_path.insert(0, _Finder())
