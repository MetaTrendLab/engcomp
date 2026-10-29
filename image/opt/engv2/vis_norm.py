import sys

OLD = '''    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed_output = self._norm(hidden_states.float())
        if self.with_scale:
            normed_output = normed_output * self.weight.float()
        return normed_output.type_as(hidden_states)
'''
NEW = '''    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _fused = _reax_rmsnorm(hidden_states, self.weight if self.with_scale else None, self.eps)
        if _fused is not None:
            return _fused
        normed_output = self._norm(hidden_states.float())
        if self.with_scale:
            normed_output = normed_output * self.weight.float()
        return normed_output.type_as(hidden_states)
'''
ADD = '''

# ---------------------------------------------------------------- engv2 vision: Gemma4RMSNorm fused, ATen-exact (vis_norm.py)
import os as _reax_os

import numpy as _reax_np
import triton as _reax_triton
import triton.language as _reax_tl


@_reax_triton.jit
def _reax_mul_rn(a, b):
    """a * b rounded on its own (an asm mul cannot be contracted into an FMA with the add that follows, as eager's
    separate pow / mean / add kernels never are)."""
    return _reax_tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=_reax_tl.float32,
                                           is_pure=True, pack=1)


@_reax_triton.jit
def _reax_bi_tree(v):
    """[128] thread values (thread = warp * 32 + lane) -> the scalar tl.sum returns for a 4-warp block: butterfly over the
    32 lanes of each warp (v[l] + v[l + 16], ...), then (w0 + w2) + (w1 + w3) across the warps."""
    v = _reax_tl.reshape(v, (4, 32))
    v = _reax_tl.permute(_reax_tl.reshape(v, (4, 2, 16)), (0, 2, 1))
    lo, hi = _reax_tl.split(v)
    v = lo + hi
    v = _reax_tl.permute(_reax_tl.reshape(v, (4, 2, 8)), (0, 2, 1))
    lo, hi = _reax_tl.split(v)
    v = lo + hi
    v = _reax_tl.permute(_reax_tl.reshape(v, (4, 2, 4)), (0, 2, 1))
    lo, hi = _reax_tl.split(v)
    v = lo + hi
    v = _reax_tl.permute(_reax_tl.reshape(v, (4, 2, 2)), (0, 2, 1))
    lo, hi = _reax_tl.split(v)
    v = lo + hi
    lo, hi = _reax_tl.split(v)  # [4, 2] -> lanes 0 and 1
    v = lo + hi  # [4] warp sums
    v = _reax_tl.permute(_reax_tl.reshape(v, (2, 2)), (1, 0))
    lo, hi = _reax_tl.split(v)  # (w0, w1), (w2, w3)
    v = lo + hi
    lo, hi = _reax_tl.split(v)
    return lo + hi


@_reax_triton.jit
def _reax_rsqrt_v(ms, RSQRT: _reax_tl.constexpr):
    """ATen's pow(x, -0.5): rsqrt variant 0 (rsqrt.approx.f32), 1 (.ftz) or 2 (1 / sqrt.rn), picked by `check`."""
    if RSQRT == 0:
        r = _reax_tl.inline_asm_elementwise("rsqrt.approx.f32 $0, $1;", "=r,r", [ms], dtype=_reax_tl.float32,
                                            is_pure=True, pack=1)
    elif RSQRT == 1:
        r = _reax_tl.inline_asm_elementwise("rsqrt.approx.ftz.f32 $0, $1;", "=r,r", [ms], dtype=_reax_tl.float32,
                                            is_pure=True, pack=1)
    else:
        r = 1.0 / _reax_tl.sqrt_rn(ms)
    return r


@_reax_triton.jit
def _reax_bi_tree_rows(v, R: _reax_tl.constexpr):
    """_reax_bi_tree on each row of [R, 128]: butterfly over the 32 lanes of each warp, then (w0 + w2) + (w1 + w3)."""
    v = _reax_tl.reshape(v, (R, 4, 32))
    v = _reax_tl.permute(_reax_tl.reshape(v, (R, 4, 2, 16)), (0, 1, 3, 2))
    lo, hi = _reax_tl.split(v)
    v = lo + hi
    v = _reax_tl.permute(_reax_tl.reshape(v, (R, 4, 2, 8)), (0, 1, 3, 2))
    lo, hi = _reax_tl.split(v)
    v = lo + hi
    v = _reax_tl.permute(_reax_tl.reshape(v, (R, 4, 2, 4)), (0, 1, 3, 2))
    lo, hi = _reax_tl.split(v)
    v = lo + hi
    v = _reax_tl.permute(_reax_tl.reshape(v, (R, 4, 2, 2)), (0, 1, 3, 2))
    lo, hi = _reax_tl.split(v)
    v = lo + hi
    lo, hi = _reax_tl.split(v)  # [R, 4, 2] -> lanes 0 and 1
    v = lo + hi  # [R, 4] warp sums
    v = _reax_tl.permute(_reax_tl.reshape(v, (R, 2, 2)), (0, 2, 1))
    lo, hi = _reax_tl.split(v)  # (w0, w1), (w2, w3)
    v = lo + hi
    lo, hi = _reax_tl.split(v)
    return lo + hi


@_reax_triton.jit
def _reax_rmsnorm_small_kernel(x_ptr, w_ptr, y_ptr, eps, rows, N: _reax_tl.constexpr, R: _reax_tl.constexpr,
                               HAS_W: _reax_tl.constexpr, RSQRT: _reax_tl.constexpr):
    """N <= 128 and N % 16 != 0 (q/k/v norms, N = 72): vLLM's mean_kernel loads one fp32 per thread, so element c sits
    in warp c // 32, lane c % 32 and the other 7 per-thread slots of the 1024 block are +0 (adding +0 to a sum of
    squares is exact). R rows per program instead of one 4-warp program per 72-element row."""
    rid = _reax_tl.program_id(0).to(_reax_tl.int64) * R + _reax_tl.arange(0, R)
    cols = _reax_tl.arange(0, 128)
    m = (rid[:, None] < rows) & (cols[None, :] < N)
    xv = _reax_tl.load(x_ptr + rid[:, None] * N + cols[None, :], mask=m, other=0.0).to(_reax_tl.float32)
    acc = _reax_bi_tree_rows(_reax_mul_rn(xv, xv), R)
    ms = _reax_tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=r,r,r", [acc / N, eps + _reax_tl.zeros_like(acc)],
                                         dtype=_reax_tl.float32, is_pure=True, pack=1)
    r = _reax_rsqrt_v(ms, RSQRT)
    y = xv * r[:, None]
    if HAS_W:
        y = y * _reax_tl.load(w_ptr + cols, mask=cols < N, other=0.0).to(_reax_tl.float32)[None, :]
    _reax_tl.store(y_ptr + rid[:, None] * N + cols[None, :], y.to(_reax_tl.bfloat16), mask=m)


@_reax_triton.jit
def _reax_rmsnorm_kernel(x_ptr, w_ptr, y_ptr, eps, N: _reax_tl.constexpr, BLOCK: _reax_tl.constexpr,
                         HAS_W: _reax_tl.constexpr, SPT: _reax_tl.constexpr, RSQRT: _reax_tl.constexpr):
    row = _reax_tl.program_id(0).to(_reax_tl.int64)
    xr = x_ptr + row * N
    acc = 0.0
    for n0 in _reax_tl.static_range(0, N, 1024):  # vLLM mean_kernel: acc += tl.sum(chunk of 1024), chunk padded with 0
        if SPT == 4:  # 16-divisible row stride: element ((r * 4 + w) * 32 + l) * 4 + j, thread sums (r, j) in order
            t = _reax_tl.arange(0, 128)[:, None] * 4 + _reax_tl.arange(0, 4)[None, :]
            for r in _reax_tl.static_range(2):
                idx = n0 + r * 512 + t
                v = _reax_tl.load(xr + idx, mask=idx < N, other=0.0).to(_reax_tl.float32)
                v = _reax_mul_rn(v, v)
                e, o = _reax_tl.split(_reax_tl.reshape(v, (128, 2, 2)))  # e = slots (0, 2), o = slots (1, 3)
                j0, j2 = _reax_tl.split(e)
                j1, j3 = _reax_tl.split(o)
                if r == 0:
                    th = ((j0 + j1) + j2) + j3
                else:
                    th = (((th + j0) + j1) + j2) + j3
        else:  # scalar loads: element (r * 4 + w) * 32 + l, thread sums r = 0..7 in order
            t = _reax_tl.arange(0, 128)
            for r in _reax_tl.static_range(8):
                idx = n0 + r * 128 + t
                v = _reax_tl.load(xr + idx, mask=idx < N, other=0.0).to(_reax_tl.float32)
                v = _reax_mul_rn(v, v)
                if r == 0:
                    th = v
                else:
                    th = th + v
        acc = acc + _reax_bi_tree(th)
    # the same Triton fp32 division as vLLM's mean_kernel (div.full, not div.rn). vLLM stores the mean and eager adds eps
    # in a separate kernel; here ptxas expands div.full into rcp + mul and contracts that mul with a plain add.f32 into
    # one FFMA (2218 of 8192 rows off by an ulp) - add.rn.f32 is never contracted
    ms = _reax_tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=r,r,r", [acc / N, eps + _reax_tl.zeros_like(acc)],
                                         dtype=_reax_tl.float32, is_pure=True, pack=1)
    r = _reax_rsqrt_v(ms, RSQRT)
    cols = _reax_tl.arange(0, BLOCK)
    m = cols < N
    xv = _reax_tl.load(xr + cols, mask=m, other=0.0).to(_reax_tl.float32)
    y = xv * r  # two multiplies and a bf16 round: nothing for ptxas to contract
    if HAS_W:
        y = y * _reax_tl.load(w_ptr + cols, mask=m, other=0.0).to(_reax_tl.float32)
    _reax_tl.store(y_ptr + row * N + cols, y.to(_reax_tl.bfloat16), mask=m)
'''
ADD += '''

_REAX_RN = {"on": None, "rsqrt": 0, "bi": False, "small": _reax_os.environ.get("REAX_VIS_NORM_SMALL", "1") == "1"}


def _reax_rmsnorm_launch(x, w, eps, rsqrt):
    N = x.shape[-1]
    rows = x.numel() // N
    y = torch.empty_like(x)
    if N <= 128 and N % 16 != 0 and _REAX_RN["small"]:
        R = 16
        _reax_rmsnorm_small_kernel[(_reax_triton.cdiv(rows, R),)](x, w if w is not None else x, y,
                                                                   float(_reax_np.float32(eps)), rows, N=N, R=R,
                                                                   HAS_W=w is not None, RSQRT=rsqrt, num_warps=4)
        return y
    # the x^2 tensor vLLM's mean_kernel reads has row stride N: Triton vectorizes (4 fp32 per thread) iff N % 16 == 0
    _reax_rmsnorm_kernel[(rows,)](x, w if w is not None else x, y, float(_reax_np.float32(eps)), N=N,
                                  BLOCK=_reax_triton.next_power_of_2(N), HAS_W=w is not None,
                                  SPT=4 if N % 16 == 0 else 1, RSQRT=rsqrt, num_warps=4)
    return y


def _reax_rmsnorm_ok(x, w):
    N = x.shape[-1] if x.dim() else 0
    # vLLM's batch-invariant mean (aten::mean.dim override) has one program per row whatever the row count
    return (x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous() and x.numel() > 0 and 0 < N <= 4096
            and _REAX_RN["bi"]
            and (w is None or (w.dtype == torch.bfloat16 and w.is_contiguous() and w.numel() == N)))


def _reax_rmsnorm_eager(x, w, eps):
    h = x.float()
    n = h * torch.pow(h.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    if w is not None:
        n = n * w.float()
    return n.type_as(x)


def _reax_bi_enable():
    """The engine runs vLLM with VLLM_BATCH_INVARIANT=1, which overrides aten::mean.dim (Triton mean_kernel): the eager
    norm is only the reference under that mode. True when the override is active in this process (enabled if needed)."""
    from vllm.model_executor.determinism import batch_invariant as _bi
    if not getattr(_bi, "_batch_invariant_MODE", False):
        _reax_os.environ["VLLM_BATCH_INVARIANT"] = "1"
        _bi.init_batch_invariance()
    return bool(getattr(_bi, "_batch_invariant_MODE", False))


def _reax_rmsnorm_check():
    """The first rsqrt variant that reproduces the eager norm (vLLM batch-invariant mode) bit for bit, or -1."""
    if not _reax_bi_enable():
        return -1
    g = torch.Generator(device="cuda").manual_seed(5152)
    cases = []
    for N in (1152, 72, 2816, 256, 40, 100):
        for rows in (1, 7, 16, 280, 2520, 16 * 2520, 15000):
            for kind in (0, 1, 2):
                if kind == 0:
                    x = torch.randn(rows, N, device="cuda", generator=g).bfloat16()
                elif kind == 1:  # exponents 2^-70 .. 2^60: x^2 under/overflow in fp32, subnormal sums, inf rows
                    ex = torch.randint(-70, 61, (rows, N), device="cuda", generator=g).float()
                    x = (torch.randn(rows, N, device="cuda", generator=g) * torch.exp2(ex)).bfloat16()
                else:  # model-like scale spread per row + exact zeros
                    s = torch.exp2(torch.randint(-12, 13, (rows, 1), device="cuda", generator=g).float())
                    x = (torch.randn(rows, N, device="cuda", generator=g) * s).bfloat16()
                    x[torch.rand(rows, N, device="cuda", generator=g) < 0.05] = 0
                w = (1 + torch.randn(N, device="cuda", generator=g) * 0.3).bfloat16()
                cases.append((x, w))
    for rsqrt in (0, 1, 2):
        ok = True
        for x, w in cases:
            for ww in (w, None):
                for eps in (1e-6, 1e-5):
                    ref = _reax_rmsnorm_eager(x, ww, eps)
                    out = _reax_rmsnorm_launch(x, ww, eps, rsqrt)
                    if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                        print(f"reax vis_norm: rsqrt variant {rsqrt} differs at N {x.shape[-1]} rows {x.shape[0]} "
                              f"scale {ww is not None} eps {eps}: {int((out.view(torch.int16) != ref.view(torch.int16)).sum())}"
                              f" elements", flush=True)
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return rsqrt
    return -1


def _reax_rmsnorm(x, w, eps):
    st = _REAX_RN
    if st["on"] is None:
        if not x.is_cuda or torch.cuda.is_current_stream_capturing():
            return None
        st["on"] = False
        from vllm.model_executor.determinism import batch_invariant as _bi
        st["bi"] = bool(getattr(_bi, "_batch_invariant_MODE", False))  # else the eager mean is ATen's: stay eager
        if _reax_os.environ.get("REAX_VIS_NORM", "1") == "1":
            path = _reax_os.environ.get("REAX_VIS_NORM_FILE", "")
            try:
                if path and _reax_os.path.exists(path):
                    st["rsqrt"] = int(open(path).read().split()[0])
                    how = f"loaded from {path}"
                else:
                    st["rsqrt"] = _reax_rmsnorm_check()
                    how = "checked in-process"
                st["on"] = st["rsqrt"] >= 0
            except Exception as e:  # noqa: BLE001 - keep the eager norm
                how = f"disabled ({e!r})"
            print(f"reax vis_norm: {'on, rsqrt variant ' + str(st['rsqrt']) if st['on'] else 'off'} ({how})", flush=True)
    if not st["on"] or not _reax_rmsnorm_ok(x, w):
        return None
    return _reax_rmsnorm_launch(x, w, eps, st["rsqrt"])
'''

if len(sys.argv) > 1 and sys.argv[1] in ("test", "derive"):
    import torch
    from transformers.models.gemma4 import modeling_gemma4 as mg
    if sys.argv[1] == "derive":
        v = -1
        try:
            v = mg._reax_rmsnorm_check()
        except Exception as e:
            print(f"reax vis_norm: check failed ({e!r})", flush=True)
        open(sys.argv[2], "w").write(f"{v}\n")
        print(f"reax vis_norm: derived rsqrt variant {v}", flush=True)
        sys.exit(0)
    print("check picks rsqrt variant", mg._reax_rmsnorm_check(), flush=True)
    g = torch.Generator(device="cuda").manual_seed(3)
    for rsqrt in (0, 1, 2):
        bad = 0
        for N, rows in ((1152, 2520), (72, 16 * 2520)):
            ex = torch.randint(-70, 61, (rows, N), device="cuda", generator=g).float()
            x = (torch.randn(rows, N, device="cuda", generator=g) * torch.exp2(ex)).bfloat16()
            w = (1 + torch.randn(N, device="cuda", generator=g) * 0.3).bfloat16()
            bad += int((mg._reax_rmsnorm_launch(x, w, 1e-6, rsqrt).view(torch.int16)
                        != mg._reax_rmsnorm_eager(x, w, 1e-6).view(torch.int16)).sum())
        print(f"rsqrt variant {rsqrt}: mismatching elements {bad}", flush=True)
    from triton.testing import do_bench
    for N, rows in ((1152, 2520), (72, 16 * 2520), (1152, 4 * 2520)):
        x = torch.randn(rows, N, device="cuda", generator=g).bfloat16()
        w = (1 + torch.randn(N, device="cuda", generator=g) * 0.3).bfloat16()
        te = do_bench(lambda: mg._reax_rmsnorm_eager(x, w, 1e-6))
        tf = do_bench(lambda: mg._reax_rmsnorm_launch(x, w, 1e-6, 0))
        print(f"N {N} rows {rows}: eager {te * 1e3:7.1f} us  fused {tf * 1e3:7.1f} us  x{te / tf:.2f}", flush=True)
    sys.exit(0)
path = sys.argv[1]
src = open(path).read()
i = src.index("class Gemma4RMSNorm(nn.Module):")
j = src.index(OLD, i)
assert j < src.index("\nclass ", i + 1), "Gemma4RMSNorm.forward not found"
src = src[:j] + NEW + src[j + len(OLD):] + ADD
open(path, "w").write(src)
print("vis_norm: installed")
