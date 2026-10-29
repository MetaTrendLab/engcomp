import os
import sys

import torch

DENSE = ((8192, 2816), (10240, 2816), (2816, 8192), (2816, 4096), (4224, 2816), (2816, 2112))
_STATE = {"g": None, "w": {}, "cfg": {}, "shapes": None}


def _shapes():
    if _STATE["shapes"] is None:
        v = os.environ.get("REAX_FP8_DENSE", "0")
        if v in ("", "0"):
            _STATE["shapes"] = frozenset()
        elif v == "all":
            _STATE["shapes"] = frozenset(DENSE)
        else:
            _STATE["shapes"] = frozenset(tuple(int(t) for t in s.split("x")) for s in v.split(","))
    return _STATE["shapes"]


def _g():
    if _STATE["g"] is None:
        import reax_f8cubin
        _STATE["g"] = reax_f8cubin.F8Gemm(os.environ.get("REAX_F8GEMM_CUBIN", "/opt/engv2/f8gemm.cubin"))
    return _STATE["g"]


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _quant_rows(x_ptr, q_ptr, s_ptr, K, BLOCK: tl.constexpr):
        r = tl.program_id(0).to(tl.int64)
        o = tl.arange(0, BLOCK)
        v = tl.load(x_ptr + r * K + o, mask=o < K, other=0.0).to(tl.float32)
        s = tl.maximum(tl.max(tl.abs(v), axis=0), 1e-12) / 448.0
        q = tl.minimum(tl.maximum(tl.math.div_rn(v, s), -448.0), 448.0).to(tl.float8e4nv)
        tl.store(q_ptr + r * K + o, q, mask=o < K)
        tl.store(s_ptr + r, s)
except ImportError:
    _quant_rows = None


def quant(x):
    M, K = x.shape
    q = torch.empty(M, K, device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(M, device=x.device, dtype=torch.float32)
    if M:
        b = triton.next_power_of_2(K)
        _quant_rows[(M,)](x, q, s, K, BLOCK=b, num_warps=8 if b >= 4096 else 4)
    return q, s


def quant_ref(x):
    s = x.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-12) / 448.0
    return (x.float() / s).clamp(-448, 448).to(torch.float8_e4m3fn), s.squeeze(-1)


def _fits(cfg, N, K):
    BM, BN, BK, th, st = _cfgs()[cfg]
    return N % BN == 0 and K % BK == 0 and cfg in _g().fn


def _cfgs():
    import reax_f8cubin
    return {k: v for k, v in reax_f8cubin.CFGS.items() if k.startswith("f8_")}


EDGES = (128, 256, 384, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 2048, 2560, 1 << 30)
_FLUSH = []


def _time(fn, n=7):
    if not _FLUSH:
        _FLUSH.append(torch.empty(256 << 20, dtype=torch.uint8, device="cuda"))
    fn()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n)]
    for a, b in ev:
        _FLUSH[0].zero_()
        a.record()
        fn()
        b.record()
    torch.cuda.synchronize()
    return sorted(a.elapsed_time(b) for a, b in ev)[n // 2]


def _sm(q, w8, s, sw):
    return torch._scaled_mm(q, w8.t(), scale_a=s[:, None], scale_b=sw[None, :], out_dtype=torch.bfloat16)


def _run(p, q, w8, s, sw):
    return _sm(q, w8, s, sw) if p == "sm" else _g()(q, w8, s, sw, cfg=p)


def _tune(N, K, w8, sw):
    force = os.environ.get("REAX_FP8_CFG")
    cands = [force] if force else [c for c in _cfgs() if _fits(c, N, K)]
    g = _g()
    x = torch.randn(4096, K, device=w8.device, generator=torch.Generator(w8.device).manual_seed(0)).bfloat16()
    q, s = quant(x)
    ref = g(q, w8, s, sw, cfg=cands[0])
    ok = [c for c in cands if all(torch.equal(g(q[:m], w8, s[:m], sw, cfg=c), ref[:m]) for m in (1, 77, 1024, 4096))]
    if len(ok) < len(cands):
        print(f"reax fp8d: {N}x{K}: {len(cands) - len(ok)} configs not bit-identical, dropped", flush=True)
    sm = not force and os.environ.get("REAX_FP8_SM", "1") == "1"
    if sm:
        try:
            sm = all(torch.equal(_sm(q[:m], w8, s[:m], sw), ref[:m]) for m in range(1, 1025))
        except Exception as e:
            print(f"reax fp8d: _scaled_mm unavailable ({type(e).__name__})", flush=True)
            sm = False
    table = []
    for e in EDGES:
        m = min(e, 4096)
        cs = ok + (["sm"] if sm and e <= 1024 else [])
        t = [_time(lambda: _run(c, q[:m], w8, s[:m], sw)) for c in cs]
        table.append(cs[t.index(min(t))])
    names = sorted(set(table))
    print(f"reax fp8d: {N}x{K}: {len(ok)} configs exact, _scaled_mm {'exact' if sm else 'off'}; buckets "
          + " ".join(str(names.index(p)) for p in table) + " = " + ", ".join(names), flush=True)
    return table


def _pick(N, K, M):
    t = _STATE["cfg"][(N, K)]
    for e, p in zip(EDGES, t):
        if M <= e:
            return p
    return t[-1]


def prep(w):
    sw = w.abs().amax(dim=1).float().clamp(min=1e-12) / 448.0
    w8 = (w.float() / sw[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return w8.contiguous(), sw.contiguous()


_MODE = os.environ.get("REAX_FP8_MODE", "fp8")
_HAD = int(os.environ.get("REAX_FP8_HAD", "0"))
_HCACHE = {}


def _hblock(K, dev):
    if not _HAD:
        return None
    b = 1
    while b * 2 <= _HAD and K % (b * 2) == 0:
        b *= 2
    if (b, dev) not in _HCACHE:
        h = torch.ones(1, 1, dtype=torch.float32)
        while h.shape[0] < b:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        h = h / b ** 0.5
        if os.environ.get("REAX_FP8_HADSIGN", "1") == "1":
            d = torch.randint(0, 2, (b,), generator=torch.Generator().manual_seed(0x52454158)).float() * 2 - 1
            h = d[:, None] * h
        _HCACHE[(b, dev)] = h.to(dev)
    return _HCACHE[(b, dev)]


def _rot(x):
    h = _hblock(x.shape[1], x.device)
    x = x.float()
    if h is None:
        return x
    M, K = x.shape
    return (x.reshape(M, K // h.shape[0], h.shape[0]) @ h).reshape(M, K)


def _q8(x):
    s = x.abs().amax(dim=1).clamp(min=1e-12) / 127.0
    return torch.round(x / s[:, None]).clamp(-127, 127).to(torch.int8).contiguous(), s.contiguous()


def prep_int8(w):
    return _q8(_rot(w))


def int8_mm(x2, w8, sw, si=None):
    if si is not None and si.numel():
        x2 = x2.float() * si[None, :]
    q, sa = _q8(_rot(x2))
    M = q.shape[0]
    Mp = max(32, (M + 7) // 8 * 8)
    if Mp != M:
        q = torch.cat([q, q.new_zeros(Mp - M, q.shape[1])])
    acc = torch._int_mm(q, w8.t())[:M]
    return ((acc.float() * sa[:, None]) * sw[None, :]).to(torch.bfloat16)


def mm(a, w):
    N, K = w.shape
    if (N, K) not in _shapes() or a.dtype != torch.bfloat16 or not a.is_contiguous():
        return None
    e = _STATE["w"].get(w.data_ptr())
    if e is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"reax fp8d: weight {N}x{K} first seen during graph capture")
        e = _STATE["w"][w.data_ptr()] = prep(w)
        if (N, K) not in _STATE["cfg"]:
            _STATE["cfg"][(N, K)] = _tune(N, K, *e)
        n = len(_STATE["w"])
        if n in (1, 30, 60, 90, 120, 150, 180):
            print(f"reax fp8d: {n} weights on FP8", flush=True)
    q, s = quant(a)
    return _run(_pick(N, K, a.shape[0]), q, e[0], s, e[1])


def _layer_ok(prefix):
    rng = os.environ.get("REAX_FP8_LAYERS", "")
    if not rng:
        return True
    import re
    m = re.search(r"layers\.(\d+)\.", prefix)
    lo, hi = (int(t) for t in rng.split("-"))
    return m is not None and lo <= int(m.group(1)) <= hi


_OBS = {"stats": {}, "n": 0, "route": {}}


def _stat(prefix, x):
    x2 = x.reshape(-1, x.shape[-1])
    if x2.shape[0] >= 8192:
        return
    a = x2.abs().amax(0).float()
    prev = _OBS["stats"].get(prefix)
    _OBS["stats"][prefix] = a if prev is None else torch.maximum(prev, a)
    _OBS["n"] += 1
    if _OBS["n"] % 240 == 0:
        torch.save({k: v.cpu() for k, v in _OBS["stats"].items()}, os.environ["REAX_SQ_STATS"])


def _route(prefix, out):
    o = out[0] if isinstance(out, tuple) else out
    if o.shape[0] >= 8192:
        return
    _OBS["route"].setdefault(prefix, []).append(o.topk(8, dim=-1).indices.sort(-1).values.to(torch.int16).cpu())
    if _OBS.get("last") is None:
        import re
        _OBS["last"] = max(_OBS.get("routers", [prefix]), key=lambda p: int(re.search(r"layers\.(\d+)\.", p).group(1)))
    if prefix == _OBS["last"]:
        torch.save(_OBS["route"], os.environ["REAX_ROUTE_DUMP"])


def swap(layer):
    w = getattr(layer, "weight", None)
    prefix = getattr(layer, "prefix", "")
    if os.environ.get("REAX_ROUTE_DUMP") and prefix.endswith("router.proj"):
        _OBS.setdefault("routers", []).append(prefix)
        if len(_OBS["routers"]) == 1:
            print(f"reax fp8d: router dump on ({prefix} ...)", flush=True)
        layer.register_forward_hook(lambda m, args, out, p=prefix: _route(p, out))
    if (not isinstance(w, torch.Tensor) or w.dim() != 2 or w.dtype != torch.bfloat16 or not w.is_cuda
            or tuple(w.shape) not in _shapes() or not _layer_ok(prefix)):
        return
    N, K = w.shape
    if _MODE == "stats":
        layer.register_forward_pre_hook(lambda m, args, p=prefix: _stat(p, args[0]))
        return
    si = torch.empty(0, device=w.device)
    if _MODE == "int8" and os.environ.get("REAX_SQ_FILE"):
        if "sq" not in _OBS:
            _OBS["sq"] = torch.load(os.environ["REAX_SQ_FILE"])
        ax = _OBS["sq"].get(prefix)
        if ax is not None:
            al = float(os.environ.get("REAX_SQ_ALPHA", "0.5"))
            aw = w.data.abs().amax(0).float()
            sm = (ax.to(w.device).clamp(min=1e-5) ** al / aw.clamp(min=1e-5) ** (1 - al)).clamp(min=1e-5)
            w.data = (w.data.float() * sm[None, :]).to(torch.bfloat16)
            si = (1.0 / sm).contiguous()
        else:
            print(f"reax fp8d: no SQ stats for {prefix}", flush=True)
    layer.register_buffer("reax_si", si, persistent=False)
    if _MODE == "int8":
        w8, sw = prep_int8(w.data)
    else:
        w8, sw = prep(w.data)
        if (N, K) not in _STATE["cfg"]:
            _STATE["cfg"][(N, K)] = _tune(N, K, w8, sw)
    layer.register_buffer("reax_w8", w8, persistent=False)
    layer.register_buffer("reax_sw", sw, persistent=False)
    layer.weight.data = torch.zeros(1, dtype=w.dtype, device=w.device).expand(N, K)
    _STATE["n"] = _STATE.get("n", 0) + 1
    if _STATE["n"] in (1, 30, 60, 90, 120, 150, 180):
        print(f"reax fp8d: {_STATE['n']} linear layers on {_MODE} (Hadamard block <= {_HAD})", flush=True)


@torch.library.custom_op("reax::fp8_linear", mutates_args=())
def fp8_linear(x: torch.Tensor, w8: torch.Tensor, sw: torch.Tensor, si: torch.Tensor) -> torch.Tensor:
    x2 = x.reshape(-1, x.shape[-1])
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    if w8.dtype == torch.int8:
        return int8_mm(x2, w8, sw, si).reshape(*x.shape[:-1], w8.shape[0])
    q, s = quant(x2)
    y = _run(_pick(w8.shape[0], w8.shape[1], x2.shape[0]), q, w8, s, sw)
    return y.reshape(*x.shape[:-1], w8.shape[0])


@fp8_linear.register_fake
def _fp8_linear_fake(x, w8, sw, si):
    return x.new_empty((*x.shape[:-1], w8.shape[0]))


OLD_P = '''    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if current_platform.is_cpu():
'''
NEW_P = '''    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _REAX_FP8D is not None:
            _REAX_FP8D.swap(layer)
        if current_platform.is_cpu():
'''
OLD_A = '''        if envs.VLLM_BATCH_INVARIANT and (
            current_platform.is_cuda_alike() or current_platform.is_xpu()
        ):
            return linear_batch_invariant(x, layer.weight, bias)
'''
NEW_A = '''        if hasattr(layer, "reax_w8"):
            _y = torch.ops.reax.fp8_linear(x, layer.reax_w8, layer.reax_sw, layer.reax_si)
            return _y if bias is None else _y + bias
''' + OLD_A
ADD = '''

# engv2 Blink fp8d: FP8 W8A8 dense layers for the process with REAX_FP8_DENSE set (tools/engv2/blink/fp8d/fp8d.py)
import os as _reax_os

_REAX_FP8D = None
if _reax_os.environ.get("REAX_FP8_DENSE", "0") not in ("", "0"):
    import reax_fp8d as _REAX_FP8D
'''


def install(path):
    src = open(path).read()
    for old, new in ((OLD_P, NEW_P), (OLD_A, NEW_A)):
        assert src.count(old) == 1, f"hunk not found: {old[:60]!r}"
        src = src.replace(old, new)
    open(path, "w").write(src + ADD)
    print("fp8d: UnquantizedLinearMethod FP8 swap installed")


def test():
    torch.manual_seed(0)
    from vllm.model_executor.determinism import batch_invariant
    bf = torch.ops.reax.dense_mm
    os.environ["REAX_FP8_DENSE"] = "all"
    for N, K in DENSE:
        w = (torch.randn(N, K, device="cuda") * 0.02).bfloat16()
        x = torch.randn(1500, K, device="cuda").bfloat16()
        q, s = quant(x)
        q0, s0 = quant_ref(x)
        assert torch.equal(q.view(torch.uint8), q0.view(torch.uint8)) and torch.equal(s, s0), "quant != torch reference"
        y = mm(x, w)
        for m in (1, 7, 64, 333, 900, 1024, 1100):
            assert torch.equal(mm(x[:m].contiguous(), w), y[:m]), f"{N}x{K}: rows differ at M {m}"
        try:
            w8, sw = _STATE["w"][w.data_ptr()]
            ref = torch._scaled_mm(q, w8.t(), scale_a=s[:, None], scale_b=sw[None, :], out_dtype=torch.bfloat16)
            same = "== _scaled_mm" if torch.equal(ref, y) else f"!= _scaled_mm ({(ref != y).float().mean().item():.2%})"
        except Exception as e:
            same = f"_scaled_mm n/a ({type(e).__name__})"
        lay = torch.nn.Module()
        lay.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
        swap(lay)
        assert lay.weight.untyped_storage().nbytes() == 2, "bf16 weight not released"
        assert torch.equal(torch.ops.reax.fp8_linear(x, lay.reax_w8, lay.reax_sw, lay.reax_si), y), f"{N}x{K}: op != mm"
        err = ((y.float() - x.float() @ w.float().t()).abs().max() / (x.float() @ w.float().t()).abs().max()).item()
        row = []
        for m in (645, 900, 1014):
            xm = x[:m].contiguous()
            t8, tq, tb = _time(lambda: mm(xm, w)), _time(lambda: quant(xm)), _time(lambda: bf(xm, w))
            row.append(f"M {m} {t8 * 1e3:.1f} (q {tq * 1e3:.1f}) vs {tb * 1e3:.1f} us {tb / t8:.2f}x")
        print(f"{N}x{K}: batch-invariant, {same}, rel max err {err:.4f}; fp8 vs bf16 dense_mm: " + "; ".join(row),
              flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "install":
        install(sys.argv[2])
    elif sys.argv[1] == "test":
        test()
