import sys

OLD = '''    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
'''
NEW = '''    def forward(self, x):
        _g = self.gate_proj(x)
        _u = self.up_proj(x)
        _h = _reax_gelu_mul(_g, _u, self.act_fn)
        if _h is None:
            _h = self.act_fn(_g) * _u
        down_proj = self.down_proj(_h)
        return down_proj
'''
ADD = '''

# ---------------------------------------------------------------- engv2 vision: gelu_tanh(gate) * up fused, eager-exact (vis_mlp.py)
import functools as _reax_mlp_functools
import os as _reax_mlp_os

import triton as _reax_mlp_triton
import triton.language as _reax_mlp_tl
from triton.language.extra.cuda import libdevice as _reax_mlp_libdevice


@_reax_mlp_triton.jit
def _reax_mlp_mul_rn(a, b):
    return _reax_mlp_tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=_reax_mlp_tl.float32,
                                               is_pure=True, pack=1)


@_reax_mlp_triton.jit
def _reax_mlp_add_rn(a, b):
    return _reax_mlp_tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=_reax_mlp_tl.float32,
                                               is_pure=True, pack=1)


@_reax_mlp_triton.jit
def _reax_mlp_gelu(x, VARIANT: _reax_mlp_tl.constexpr):
    kbeta = 0.7978845608028654  # float(M_SQRT2 * M_2_SQRTPI * 0.5), rounded to fp32 by Triton like ATen's constexpr
    kkappa = 0.044715
    if VARIANT == 0:
        x3 = _reax_mlp_mul_rn(_reax_mlp_mul_rn(x, x), x)
        inner = _reax_mlp_mul_rn(kbeta + _reax_mlp_tl.zeros_like(x), _reax_mlp_libdevice.fma(kkappa + _reax_mlp_tl.zeros_like(x), x3, x))
        t = _reax_mlp_libdevice.tanh(inner)
        y = _reax_mlp_mul_rn(_reax_mlp_mul_rn(0.5 + _reax_mlp_tl.zeros_like(x), x), _reax_mlp_add_rn(1.0 + _reax_mlp_tl.zeros_like(x), t))
    elif VARIANT == 1:
        x3 = _reax_mlp_mul_rn(_reax_mlp_mul_rn(x, x), x)
        inner = _reax_mlp_mul_rn(kbeta + _reax_mlp_tl.zeros_like(x),
                                 _reax_mlp_add_rn(x, _reax_mlp_mul_rn(kkappa + _reax_mlp_tl.zeros_like(x), x3)))
        t = _reax_mlp_libdevice.tanh(inner)
        y = _reax_mlp_mul_rn(_reax_mlp_mul_rn(0.5 + _reax_mlp_tl.zeros_like(x), x), _reax_mlp_add_rn(1.0 + _reax_mlp_tl.zeros_like(x), t))
    else:
        x3 = x * x * x
        inner = kbeta * (x + kkappa * x3)
        y = 0.5 * x * (1.0 + _reax_mlp_libdevice.tanh(inner))
    return y


@_reax_mlp_triton.jit
def _reax_gelu_mul_kernel(g_ptr, u_ptr, y_ptr, n, HAS_U: _reax_mlp_tl.constexpr, VARIANT: _reax_mlp_tl.constexpr,
                          BLOCK: _reax_mlp_tl.constexpr):
    i = _reax_mlp_tl.program_id(0).to(_reax_mlp_tl.int64) * BLOCK + _reax_mlp_tl.arange(0, BLOCK)
    m = i < n
    g = _reax_mlp_tl.load(g_ptr + i, mask=m, other=0.0).to(_reax_mlp_tl.float32)
    a = _reax_mlp_gelu(g, VARIANT).to(_reax_mlp_tl.bfloat16)
    if HAS_U:
        u = _reax_mlp_tl.load(u_ptr + i, mask=m, other=0.0).to(_reax_mlp_tl.float32)
        a = _reax_mlp_mul_rn(a.to(_reax_mlp_tl.float32), u).to(_reax_mlp_tl.bfloat16)
    _reax_mlp_tl.store(y_ptr + i, a, mask=m)


_REAX_MLP = {"variant": None}


def _reax_gelu_mul_launch(g, u, variant):
    y = torch.empty_like(g)
    n = g.numel()
    BLOCK = 2048
    _reax_gelu_mul_kernel[(_reax_mlp_triton.cdiv(n, BLOCK),)](g, u if u is not None else g, y, n, HAS_U=u is not None,
                                                              VARIANT=variant, BLOCK=BLOCK, num_warps=4)
    return y


def _reax_gelu_act_ok(act_fn):
    """HF GELUTanh in its default form (F.gelu(approximate="tanh")), not the python formula variant."""
    act = getattr(act_fn, "act", None)
    return (type(act_fn).__name__ == "GELUTanh" and isinstance(act, _reax_mlp_functools.partial)
            and act.func is torch.nn.functional.gelu and act.keywords == {"approximate": "tanh"})


def _reax_gelu_mul_check():
    """First variant that reproduces F.gelu(tanh) on all 2^16 bf16 inputs and gelu(g) * u on random data, or -1."""
    allb = torch.arange(-32768, 32768, device="cuda", dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    ref_act = torch.nn.functional.gelu(allb, approximate="tanh")
    g = torch.Generator(device="cuda").manual_seed(5154)
    for variant in (0, 1, 2):
        out = _reax_gelu_mul_launch(allb, None, variant)
        bad = int((out.view(torch.int16) != ref_act.view(torch.int16)).sum())
        if bad:
            print(f"reax vis_mlp: variant {variant}: gelu differs on {bad} of 65536 bf16 inputs", flush=True)
            continue
        ok = True
        for rows, cols in ((1, 4304), (7, 4304), (2520, 4304), (333, 1000)):
            for kind in (0, 1):
                gt = torch.randn(rows, cols, device="cuda", generator=g)
                ut = torch.randn(rows, cols, device="cuda", generator=g)
                if kind == 1:
                    gt = gt * torch.exp2(torch.randint(-60, 61, (rows, cols), device="cuda", generator=g).float())
                    ut = ut * torch.exp2(torch.randint(-60, 61, (rows, cols), device="cuda", generator=g).float())
                gt, ut = gt.bfloat16(), ut.bfloat16()
                ref = torch.nn.functional.gelu(gt, approximate="tanh") * ut
                out = _reax_gelu_mul_launch(gt, ut, variant)
                if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                    print(f"reax vis_mlp: variant {variant}: gelu * up differs at {(rows, cols)} kind {kind}: "
                          f"{int((out.view(torch.int16) != ref.view(torch.int16)).sum())} elements", flush=True)
                    ok = False
                    break
            if not ok:
                break
        if ok:
            print(f"reax vis_mlp: variant {variant} bit-identical (all 65536 bf16 gelu inputs + gelu * up)", flush=True)
            return variant
    return -1


def _reax_gelu_mul(g, u, act_fn):
    st = _REAX_MLP
    if st["variant"] is None:
        if not g.is_cuda or torch.cuda.is_current_stream_capturing():
            return None
        st["variant"] = -1
        if _reax_mlp_os.environ.get("REAX_VIS_MLP", "1") == "1":
            path = _reax_mlp_os.environ.get("REAX_VIS_MLP_FILE", "")
            try:
                if path and _reax_mlp_os.path.exists(path):
                    st["variant"] = int(open(path).read().split()[0])
                    how = f"loaded from {path}"
                else:
                    st["variant"] = _reax_gelu_mul_check()
                    how = "checked in-process"
            except Exception as e:  # noqa: BLE001 - keep the eager MLP
                how = f"disabled ({e!r})"
            print(f"reax vis_mlp: {'on, variant ' + str(st['variant']) if st['variant'] >= 0 else 'off'} ({how})",
                  flush=True)
    if st["variant"] < 0 or not _reax_gelu_act_ok(act_fn):
        return None
    if not (g.is_cuda and g.dtype == torch.bfloat16 and u.dtype == torch.bfloat16 and g.shape == u.shape
            and g.is_contiguous() and u.is_contiguous()):
        return None
    return _reax_gelu_mul_launch(g, u, st["variant"])
'''

if len(sys.argv) > 1 and sys.argv[1] in ("test", "derive"):
    import torch
    from transformers.models.gemma4 import modeling_gemma4 as mg
    if sys.argv[1] == "derive":
        v = -1
        try:
            v = mg._reax_gelu_mul_check()
        except Exception as e:
            print(f"reax vis_mlp: check failed ({e!r})", flush=True)
        open(sys.argv[2], "w").write(f"{v}\n")
        sys.exit(0)
    print("check picks variant", mg._reax_gelu_mul_check(), flush=True)
    allb = torch.arange(-32768, 32768, device="cuda", dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    ref = torch.nn.functional.gelu(allb, approximate="tanh")
    for v in (0, 1, 2):
        out = mg._reax_gelu_mul_launch(allb, None, v)
        print(f"variant {v}: gelu mismatches on all bf16 inputs {int((out.view(torch.int16) != ref.view(torch.int16)).sum())}",
              flush=True)
    from triton.testing import do_bench
    g = torch.Generator(device="cuda").manual_seed(4)
    for rows in (2520, 4 * 2520):
        gt = torch.randn(rows, 4304, device="cuda", generator=g).bfloat16()
        ut = torch.randn(rows, 4304, device="cuda", generator=g).bfloat16()
        te = do_bench(lambda: torch.nn.functional.gelu(gt, approximate="tanh") * ut)
        tf = do_bench(lambda: mg._reax_gelu_mul_launch(gt, ut, 0))
        print(f"rows {rows}: eager {te * 1e3:7.1f} us  fused {tf * 1e3:7.1f} us  x{te / tf:.2f}", flush=True)
    sys.exit(0)
path = sys.argv[1]
src = open(path).read()
i = src.index("class Gemma4VisionMLP(")
j = src.index(OLD, i)
assert j < src.index("\nclass ", i + 1), "Gemma4VisionMLP.forward not found"
src = src[:j] + NEW + src[j + len(OLD):] + ADD
open(path, "w").write(src)
print("vis_mlp: installed")
