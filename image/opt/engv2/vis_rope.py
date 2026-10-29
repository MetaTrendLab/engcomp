import sys

OLD = '''    ndim = position_ids.shape[-1]
    num_input_channels = x.shape[-1]
    num_rotated_channels_per_dim = 2 * (num_input_channels // (2 * ndim))
'''
NEW = '''    _fused = _reax_rope(x, cos, sin, position_ids, unsqueeze_dim)
    if _fused is not None:
        return _fused
    ndim = position_ids.shape[-1]
    num_input_channels = x.shape[-1]
    num_rotated_channels_per_dim = 2 * (num_input_channels // (2 * ndim))
'''
ADD = '''

# ---------------------------------------------------------------- engv2 vision: multidimensional RoPE fused, eager-exact (vis_rope.py)
import os as _reax_rope_os

import triton as _reax_rope_triton
import triton.language as _reax_rope_tl


@_reax_rope_triton.jit
def _reax_rope_mul_rn(a, b):
    """fp32 a * b as an opaque mul.rn: with plain `*`, LLVM narrows bf16(bf16(x*cos) + bf16(rh*sin)) to bf16 ops and
    contracts them into one fma (the product roundings vanish: 23 % of the elements differ from eager)."""
    return _reax_rope_tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=r,r,r", [a, b],
                                                dtype=_reax_rope_tl.float32, is_pure=True, pack=1)


@_reax_rope_triton.jit
def _reax_rope_kernel(x_ptr, cos_ptr, sin_ptr, y_ptr, NH: _reax_rope_tl.constexpr, H: _reax_rope_tl.constexpr,
                      P: _reax_rope_tl.constexpr, BLOCK: _reax_rope_tl.constexpr):
    tok = _reax_rope_tl.program_id(0).to(_reax_rope_tl.int64)  # one (batch, position): all NH heads, cos/sin row shared
    j = _reax_rope_tl.arange(0, BLOCK)
    m = j < NH * H
    c = j % H
    part = c // P
    cp = c - part * P
    half = P // 2
    src = _reax_rope_tl.where(cp < half, j + half, j - half)
    base = tok * NH * H
    x = _reax_rope_tl.load(x_ptr + base + j, mask=m, other=0.0).to(_reax_rope_tl.float32)
    xs = _reax_rope_tl.load(x_ptr + base + src, mask=m, other=0.0).to(_reax_rope_tl.float32)
    # torch.neg flips the sign bit (-(+0) = -0); Triton's unary minus is 0 - x (+0): negate with neg.f32
    nxs = _reax_rope_tl.inline_asm_elementwise("neg.f32 $0, $1;", "=r,r", [xs], dtype=_reax_rope_tl.float32,
                                               is_pure=True, pack=1)
    rh = _reax_rope_tl.where(cp < half, nxs, xs)
    cs = _reax_rope_tl.load(cos_ptr + tok * H + c, mask=m, other=0.0).to(_reax_rope_tl.float32)
    sn = _reax_rope_tl.load(sin_ptr + tok * H + c, mask=m, other=0.0).to(_reax_rope_tl.float32)
    a = _reax_rope_mul_rn(x, cs).to(_reax_rope_tl.bfloat16).to(_reax_rope_tl.float32)
    b = _reax_rope_mul_rn(rh, sn).to(_reax_rope_tl.bfloat16).to(_reax_rope_tl.float32)
    _reax_rope_tl.store(y_ptr + base + j, (a + b).to(_reax_rope_tl.bfloat16), mask=m)


_REAX_ROPE = {"on": None}


def _reax_rope_launch(x, cos, sin, ndim):
    B, L, NH, H = x.shape
    P = 2 * (H // (2 * ndim))
    y = torch.empty_like(x)
    _reax_rope_kernel[(B * L,)](x, cos, sin, y, NH=NH, H=H, P=P, BLOCK=_reax_rope_triton.next_power_of_2(NH * H),
                                num_warps=4)
    return y


def _reax_rope_ok(x, cos, sin, position_ids, unsqueeze_dim):
    if position_ids is None or unsqueeze_dim != 2 or x.dim() != 4 or position_ids.dim() + 2 == x.dim():
        return False
    ndim = position_ids.shape[-1]
    H = x.shape[-1]
    P = 2 * (H // (2 * ndim))
    return (x.is_cuda and x.dtype == torch.bfloat16 and cos.dtype == torch.bfloat16 and sin.dtype == torch.bfloat16
            and x.is_contiguous() and cos.is_contiguous() and sin.is_contiguous() and P > 0 and P * ndim == H
            and x.shape[2] * H <= 8192  # one program holds every head of a token
            and tuple(cos.shape) == (x.shape[0], x.shape[1], H) and tuple(sin.shape) == tuple(cos.shape))


def _reax_rope_eager(x, cos, sin, position_ids, unsqueeze_dim=2):
    ndim = position_ids.shape[-1]
    P = 2 * (x.shape[-1] // (2 * ndim))
    xs = torch.split(x, [P] * ndim, dim=-1)
    cs = torch.split(cos, [P] * ndim, dim=-1)
    ss = torch.split(sin, [P] * ndim, dim=-1)
    return torch.cat([apply_rotary_pos_emb(x=xs[k], cos=cs[k], sin=ss[k], unsqueeze_dim=unsqueeze_dim)
                      for k in range(ndim)], dim=-1)


def _reax_rope_check():
    g = torch.Generator(device="cuda").manual_seed(5153)
    n = 0
    for B, L, NH, H in ((1, 2520, 16, 72), (2, 300, 16, 72), (1, 17, 4, 72), (3, 64, 8, 64)):
        pos = torch.randint(0, 64, (B, L, 2), device="cuda", generator=g)
        for kind in (0, 1):
            if kind == 0:
                x = torch.randn(B, L, NH, H, device="cuda", generator=g).bfloat16()
            else:  # exponents 2^-130 .. 2^120: products that underflow/overflow bf16, subnormals, inf
                ex = torch.randint(-130, 121, (B, L, NH, H), device="cuda", generator=g).float()
                x = (torch.randn(B, L, NH, H, device="cuda", generator=g) * torch.exp2(ex)).bfloat16()
            ang = torch.rand(B, L, H, device="cuda", generator=g) * 100
            cos, sin = ang.cos().bfloat16(), ang.sin().bfloat16()
            if kind == 1:
                cos = cos * torch.exp2(torch.randint(-20, 21, (B, L, H), device="cuda", generator=g).float()).bfloat16()
            ref = _reax_rope_eager(x, cos, sin, pos)
            out = _reax_rope_launch(x, cos, sin, 2)
            if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                print(f"reax vis_rope: differs at {(B, L, NH, H)} kind {kind}: "
                      f"{int((out.view(torch.int16) != ref.view(torch.int16)).sum())} elements", flush=True)
                return False
            n += 1
    print(f"reax vis_rope: bit-identical in {n} cases", flush=True)
    return True


def _reax_rope(x, cos, sin, position_ids, unsqueeze_dim):
    st = _REAX_ROPE
    if st["on"] is None:
        if not x.is_cuda or torch.cuda.is_current_stream_capturing():
            return None
        st["on"] = False
        if _reax_rope_os.environ.get("REAX_VIS_ROPE", "1") == "1":
            path = _reax_rope_os.environ.get("REAX_VIS_ROPE_FILE", "")
            try:
                if path and _reax_rope_os.path.exists(path):
                    st["on"] = open(path).read().split()[0] == "1"
                    how = f"loaded from {path}"
                else:
                    st["on"] = _reax_rope_check()
                    how = "checked in-process"
            except Exception as e:  # noqa: BLE001 - keep the eager rope
                how = f"disabled ({e!r})"
            print(f"reax vis_rope: {'on' if st['on'] else 'off'} ({how})", flush=True)
    if not st["on"] or not _reax_rope_ok(x, cos, sin, position_ids, unsqueeze_dim):
        return None
    return _reax_rope_launch(x, cos, sin, position_ids.shape[-1])
'''

if len(sys.argv) > 1 and sys.argv[1] in ("test", "derive"):
    import torch
    from transformers.models.gemma4 import modeling_gemma4 as mg
    if sys.argv[1] == "derive":
        ok = False
        try:
            ok = mg._reax_rope_check()
        except Exception as e:
            print(f"reax vis_rope: check failed ({e!r})", flush=True)
        open(sys.argv[2], "w").write("1\n" if ok else "0\n")
        sys.exit(0)
    print("check", mg._reax_rope_check(), flush=True)
    from triton.testing import do_bench
    g = torch.Generator(device="cuda").manual_seed(4)
    for L in (2520, 4 * 2520):
        x = torch.randn(1, L, 16, 72, device="cuda", generator=g).bfloat16()
        ang = torch.rand(1, L, 72, device="cuda", generator=g) * 100
        cos, sin = ang.cos().bfloat16(), ang.sin().bfloat16()
        pos = torch.randint(0, 64, (1, L, 2), device="cuda", generator=g)
        te = do_bench(lambda: mg._reax_rope_eager(x, cos, sin, pos))
        tf = do_bench(lambda: mg._reax_rope_launch(x, cos, sin, 2))
        print(f"L {L}: eager {te * 1e3:7.1f} us  fused {tf * 1e3:7.1f} us  x{te / tf:.2f}", flush=True)
    sys.exit(0)
path = sys.argv[1]
src = open(path).read()
i = src.index("def apply_multidimensional_rope(")
j = src.index(OLD, i)
assert j < src.index("\ndef ", i + 1), "apply_multidimensional_rope body not found"
src = src[:j] + NEW + src[j + len(OLD):] + ADD
open(path, "w").write(src)
print("vis_rope: installed")
