import json
import os
import sys

TARGET = "torch._inductor.runtime.triton_heuristics"
_PINS = []
_CALLS = {}


def _cfg(c):
    return {**c.kwargs, "num_warps": c.num_warps, "num_stages": c.num_stages}


def _pin(key, n):
    if not _PINS:
        p = os.environ.get("REAX_HIDDEN_PINS")
        _PINS.append(json.load(open(p)) if p else {})
    pin = _PINS[0].get(key)
    if isinstance(pin, list):
        pin = pin[min(n, len(pin) - 1)] if pin else None
    return pin


def _log(self, timings, pinned):
    path = os.environ.get("REAX_AUTOTUNE_LOG")
    if path:
        best = min(timings, key=timings.get)
        with open(path, "a") as f:
            f.write(json.dumps({"kernel": os.path.basename(self.filename or ""), "name": self.fn.__name__,
                                "call": getattr(self, "_engv2_call", None), "pinned": pinned,
                                "chosen": _cfg(best.config),
                                "configs": [[_cfg(launcher.config), getattr(launcher, "_engv2_real", t)]
                                            for launcher, t in timings.items()]}) + "\n")


def _patch(m):
    import copy

    ca = m.CachingAutotuner
    orig_precompile, orig_one, orig_bench = ca.precompile, ca.bench, ca.benchmark_all_configs

    def precompile(self, *args, **kwargs):
        orig_precompile(self, *args, **kwargs)
        if kwargs.get("warm_cache_only", args[0] if args else False) or not self.compile_results:
            return
        key = os.path.basename(self.filename or "")
        n = _CALLS.get(key, 0)
        _CALLS[key] = n + 1
        self._engv2_call, self._engv2_pin = n, _pin(key, n)
        pin = self._engv2_pin
        if pin is None:
            return
        if not any(_cfg(r.config) == pin for r in self.compile_results):
            self._ensure_kernel_loaded()
            c = copy.deepcopy(self.compile_results[0].config)
            c.kwargs = {k: v for k, v in pin.items() if k not in ("num_warps", "num_stages")}
            c.num_warps, c.num_stages = pin["num_warps"], pin["num_stages"]
            self.compile_results.append(self._precompile_config(c))
            self._make_launchers()

    def bench(self, launcher, *args, **kwargs):
        t = orig_one(self, launcher, *args, **kwargs)
        pin = getattr(self, "_engv2_pin", None)
        if pin is not None and _cfg(launcher.config) == pin:
            launcher._engv2_real = t
            return 0.0
        return t

    def benchmark_all_configs(self, *args, **kwargs):
        timings = orig_bench(self, *args, **kwargs)
        _log(self, timings, any(hasattr(launcher, "_engv2_real") for launcher in timings))
        return timings

    def logged(fn):
        def wrapper(self, *args, **kwargs):
            try:
                return fn(self, *args, **kwargs)
            except BaseException as e:
                import traceback

                print(f"engv2_autotune: {type(e).__name__} in {fn.__name__} of {self.filename}:\n"
                      + traceback.format_exc(), file=sys.stderr, flush=True)
                raise

        return wrapper

    ca.precompile, ca.bench, ca.benchmark_all_configs = logged(precompile), bench, logged(benchmark_all_configs)
    if os.environ.get("REAX_AUTOTUNE_DEBUG"):
        ca.run, ca.autotune_to_one_config = logged(ca.run), logged(ca.autotune_to_one_config)


class _Finder:

    def find_spec(self, name, path=None, target=None):
        if name != TARGET:
            return None
        for f in sys.meta_path:
            if f is self or not hasattr(f, "find_spec"):
                continue
            spec = f.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        exec_module = spec.loader.exec_module

        def run(module):
            exec_module(module)
            _patch(module)

        spec.loader.exec_module = run
        return spec


def _kernel_path(root, basename, _index={}):
    if root not in _index:
        _index[root] = {f: os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs if f.endswith(".py")}
    return _index[root].get(basename)


def _source_sig(root, basename):
    p = _kernel_path(root, basename)
    if p is None:
        return None
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from autotune_sample import source_sig

    return source_sig(p)


def chains(records):
    out = {}
    for r in records:
        ch = out.setdefault(r["kernel"], {})
        ch[r["call"] if r.get("call") is not None else len(ch)] = r["chosen"]
    return {k: [v.get(i) for i in range(max(v) + 1)] for k, v in out.items()}


def hidden_only(ch, io):
    out = {}
    for k, v in ch.items():
        p = _kernel_path(io, k)
        if p is not None and not any(f.endswith(".best_config") for f in os.listdir(os.path.dirname(p))):
            out[k] = v
    return out


def map_chains(ch, ref_io, cand_io):
    ch = hidden_only(ch, ref_io)
    sigs = {k: _source_sig(ref_io, k) for k in ch}
    by_sig = {}
    for k, sg in sigs.items():
        if sg is not None:
            assert by_sig.setdefault(sg, ch[k]) == ch[k], f"kernels with the same source chose differently: {k}"
    out, used = {}, set()
    for d, _, fs in os.walk(cand_io):
        for f in fs:
            sg = _source_sig(cand_io, f) if f.endswith(".py") else None
            if sg in by_sig:
                out[f] = by_sig[sg]
                used.add(sg)
    print(f"mapped {len(used)} of {len(ch)} logged kernels onto {len(out)} kernels of {cand_io}; unmatched: "
          f"{sorted(k[:12] for k, sg in sigs.items() if sg not in used)}")
    return out


def pins(a):
    out = chains(map(json.loads, open(a.log)))
    out = map_chains(out, a.io, a.cand_io) if a.cand_io else hidden_only(out, a.io)
    for k, v in sorted(out.items()):
        print(f"  {k[:12]} " + " -> ".join(str(c and c.get('R0_BLOCK', c.get('XBLOCK'))) for c in v))
    with open(a.o, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)


def install(site):
    import importlib.util
    import shutil
    import sysconfig

    shutil.copy(os.path.abspath(__file__), os.path.join(site, "engv2_autotune.py"))
    spec = importlib.util.find_spec("sitecustomize")
    sc = spec.origin if spec and spec.origin else os.path.join(sysconfig.get_paths()["stdlib"], "sitecustomize.py")
    with open(sc, "a") as f:
        f.write("\n# engv2 autotune_hidden.py install: Inductor autotune log / hidden pins hook\n# (other interpreters, e.g. the q35 venv, do not have the module: skip quietly)\ntry:\n    import engv2_autotune\nexcept ModuleNotFoundError as e:\n    if e.name != \"engv2_autotune\":\n        raise\n")
    print(f"autotune_hidden: installed into {site}, imported from {sc}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("install")
    i.add_argument("site")
    p = sub.add_parser("pins")
    p.add_argument("log")
    p.add_argument("-o", required=True)
    p.add_argument("--io", required=True, help="the logged process's kept Inductor cache dir")
    p.add_argument("--cand-io", help="another image's kept Inductor cache dir: map the pins onto its kernels")
    a = ap.parse_args()
    install(a.site) if a.cmd == "install" else pins(a)
elif os.environ.get("REAX_AUTOTUNE_LOG") or os.environ.get("REAX_HIDDEN_PINS"):
    sys.meta_path.insert(0, _Finder())
