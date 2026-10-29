from __future__ import annotations

import os

import torch
import torch.nn.functional as F

ON = os.environ.get("Q35_COPIES", "0") == "1"
_G = {}

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _gnq2_kernel(op_ptr, os_ptr, Tp, z_ptr, w_ptr, q_ptr, s_ptr, z_stride, eps, NH: tl.constexpr, NHP: tl.constexpr,
                     HD: tl.constexpr):
        t = tl.program_id(0)
        h = tl.arange(0, NHP)[:, None]
        d = tl.arange(0, HD)[None, :]
        m = h < NH
        if t < Tp:
            o = tl.load(op_ptr + t * NH * HD + h * HD + d, mask=m, other=0.0).to(tl.float32)
        else:
            o = tl.load(os_ptr + (t - Tp) * NH * HD + h * HD + d, mask=m, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + t * z_stride + h * HD + d, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + d).to(tl.float32)
        r = tl.rsqrt(tl.sum(o * o, axis=1) / HD + eps)
        y = (o * r[:, None]).to(tl.bfloat16).to(tl.float32)
        y = (w * y).to(tl.bfloat16).to(tl.float32)
        y = (y * (z / (1.0 + tl.exp(-z)))).to(tl.bfloat16).to(tl.float32)
        amax = tl.max(tl.max(tl.where(m, tl.abs(y), 0.0), axis=1), axis=0)
        s = tl.maximum(amax, 1e-12) / 448.0
        q = tl.minimum(tl.maximum(y / s, -448.0), 448.0).to(tl.float8e4nv)
        tl.store(q_ptr + t * NH * HD + h * HD + d, q, mask=m)
        tl.store(s_ptr + t, s)
except ImportError:
    _gnq2_kernel = None


FLA_EMPTY = os.environ.get("Q35_FLA_EMPTY", "0") == "1"


def _fla_empty_final_state():
    import inspect
    import textwrap

    import fla.ops.common.chunk_delta_h as m
    fn = inspect.unwrap(m.chunk_gated_delta_rule_fwd_h)
    src = textwrap.dedent(inspect.getsource(fn))
    old = ("k.new_zeros(N, HV, V, K, dtype=torch.float32) if output_final_state",
           "k.new_zeros(N, HV, K, V, dtype=torch.float32) if output_final_state")
    if not all(src.count(o) == 1 for o in old) or fn.__code__.co_freevars:
        print("q35 fla_empty: FLA source differs, left as is", flush=True)
        return
    for o in old:
        src = src.replace(o, o.replace("new_zeros", "new_empty"))
    src = "\n".join(line for line in src.splitlines() if not line.startswith("@"))
    ns = {}
    exec(compile(src, inspect.getsourcefile(fn), "exec"), m.__dict__, ns)
    fn.__code__ = ns[fn.__name__].__code__
    print("q35 fla_empty: chunk_gated_delta_rule_fwd_h final states via new_empty", flush=True)


if FLA_EMPTY:
    _fla_empty_final_state()


ATT1 = os.environ.get("Q35_ATT1", "0") == "1"


def att1_meta(sh, cu_p, cu_s, cu_kx, kx, T_):
    if not ATT1:
        return
    import numpy as np
    Tp = int(cu_p[-1])
    sh.att1 = (T_(np.concatenate([np.arange(Tp)] + list(kx))), T_(np.concatenate([cu_p, Tp + cu_s[1:]]), torch.int32),
               T_(np.concatenate([cu_p, Tp + cu_kx[1:]]), torch.int32), max(sh.max_p, sh.max_s), max(sh.max_p, sh.max_kx))


def attn_shared(eng, q, k, v, sh):
    m = getattr(sh, "att1", None)
    if m is None or eng._varlen is None or getattr(sh, "kp_idx", None) is not None:
        return None
    idx, cq, ck, mq, mk = m
    if not _G.get("att1_said"):
        _G["att1_said"] = True
        print(f"q35 att1: one-call shared attention (first step T={q.shape[0]}, Tp={sh.Tp})", flush=True)
    return eng._varlen(q.contiguous(), k.index_select(0, idx), v.index_select(0, idx), cq, ck, mq, mk,
                       scale=eng.hd ** -0.5, window_size=(-1, 0) if eng.causal else (-1, -1), enable_gqa=True)


def install(g):
    _G.update(g)
    orig = g["_inba"]

    def inba(lin, x):
        mode = g["_Q35_INBA"]
        if not mode.startswith("pad") or x.shape[0] <= int(mode[3:]):
            return orig(lin, x)
        P, T = int(mode[3:]), x.shape[0]
        outs = []
        for a in range(0, T, P):
            n = min(P, T - a)
            if n == P:
                outs.append(F.linear(x[a:a + P], lin.w))
            else:
                outs.append(F.linear(x[T - P:], lin.w)[P - n:])
        return torch.cat(outs, 0) if len(outs) > 1 else outs[0]
    g["_inba"] = inba


def s_par(par, n_groups, T_):
    if len(par) == n_groups and all(p == j for j, p in enumerate(par)):
        return slice(None)
    return T_(par)


def deltanet_out_parts(eng, L, parts, z, T, xq):
    if xq is None or L["out"].mode != "fp8" or not eng.triton_gnq or _gnq2_kernel is None or len(parts) != 2:
        return None
    op, os_ = parts
    nh, hd = op.shape[-2], op.shape[-1]
    assert op.shape[1] + os_.shape[1] == T and op.is_contiguous() and os_.is_contiguous()
    q = torch.empty(T, nh * hd, device=z.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(T, 1, device=z.device, dtype=torch.float32)
    _gnq2_kernel[(T,)](op, os_, op.shape[1], z, L["gnorm"], q, s, z.stride(0), eng.eps, NH=nh,
                       NHP=triton.next_power_of_2(nh), HD=hd)
    return L["out"](None, (q, s))
