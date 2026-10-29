from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from fla.ops.utils.op import exp2

ON = os.environ.get("Q35_GDN_HO", "0") == "1"
BV = int(os.environ.get("Q35_GDN_HO_BV", "64"))
NW = int(os.environ.get("Q35_GDN_HO_WARPS", "4"))
NS = int(os.environ.get("Q35_GDN_HO_STAGES", "2"))
AQK_BK = int(os.environ.get("Q35_GDN_HO_AQK_BK", "64"))
EPI = int(os.environ.get("Q35_GDN_HO_EPI", "2"))
FUSE = os.environ.get("Q35_GDN_HO_FPFUSE", "1") == "1"
FOLD = os.environ.get("Q35_GDN_HO_FOLD", "0") == "1"
PRE = os.environ.get("Q35_GDN_PRE", "0") == "1"


@triton.jit(do_not_specialize=['T'])
def gdn_aqk_kernel(q, k, g, A, cu_seqlens, chunk_indices, T, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                   BT: tl.constexpr, BK: tl.constexpr):
    i_t, i_h = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    g += bos * HV + i_h
    A += (bos * HV + i_h) * BT

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_q = q + o_t[:, None] * (H*K) + o_k[None, :]
        p_k = k + o_k[:, None] + o_t[None, :] * (H*K)
        b_q = tl.load(p_q, mask=m_t[:, None] & m_k[None, :], other=0.0)
        b_k = tl.load(p_k, mask=m_k[:, None] & m_t[None, :], other=0.0)
        b_A += tl.dot(b_q, b_k)
    b_g = tl.load(g + o_t * HV, mask=m_t, other=0.0)
    b_A = b_A * exp2(b_g[:, None] - b_g[None, :])
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    p_A = A + o_t[:, None] * (HV*BT) + tl.arange(0, BT)[None, :]
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), mask=m_t[:, None])


@triton.jit(do_not_specialize=['T'])
def gdn_ho_kernel(q, k, w, u, g, Aqk, o, h0, ht, cu_seqlens, scale, T, H: tl.constexpr, HV: tl.constexpr,
                  K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr, USE_INITIAL_STATE: tl.constexpr,
                  STORE_FINAL_STATE: tl.constexpr, EPI: tl.constexpr = 0, FOLD_WU: tl.constexpr = False,
                  Akk=None, v=None, beta=None):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    NT = tl.cdiv(T, BT)

    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    q += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    u += (bos * HV + i_h).to(tl.int64) * V
    o += (bos * HV + i_h).to(tl.int64) * V
    Aqk += (bos * HV + i_h).to(tl.int64) * BT
    g += (bos * HV + i_h).to(tl.int64)
    if FOLD_WU:
        Akk += (bos * HV + i_h).to(tl.int64) * BT
        v += (bos * HV + i_h).to(tl.int64) * V
        beta += (bos * HV + i_h).to(tl.int64)

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    o_k2 = 64 + o_k1
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K*V
        b_h1 += tl.load(h0 + o_k1[:, None] * V + o_v[None, :], mask=m_v[None, :], other=0.0).to(tl.float32)
        b_h2 += tl.load(h0 + o_k2[:, None] * V + o_v[None, :], mask=m_v[None, :], other=0.0).to(tl.float32)

    for i_t in range(NT):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        if FOLD_WU:
            m_A = m_t[:, None] & (tl.arange(0, BT)[None, :] < BT)
            b_Ak = tl.load(Akk + o_t[:, None] * (HV*BT) + tl.arange(0, BT)[None, :], mask=m_A, other=0.0)
            b_b = tl.load(beta + o_t * HV, mask=m_t, other=0.0)
            b_eg = exp2(tl.load(g + o_t * HV, mask=m_t, other=0.0))
            b_vv = tl.load(v + o_t[:, None] * (HV*V) + o_v[None, :], mask=m_t[:, None] & m_v[None, :], other=0.0)
            b_vb = (b_vv * b_b[:, None]).to(b_vv.dtype)
            b_u = tl.dot(b_Ak, b_vb, allow_tf32=False).to(u.dtype.element_ty)
            b_kk = tl.load(k + o_t[:, None] * (H*K) + o_k1[None, :], mask=m_t[:, None], other=0.0)
            b_kb = b_kk * b_b[:, None]
            b_kb *= b_eg[:, None]
            b_w = tl.dot(b_Ak, b_kb.to(b_kk.dtype)).to(w.dtype.element_ty)
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
            b_kk = tl.load(k + o_t[:, None] * (H*K) + o_k2[None, :], mask=m_t[:, None], other=0.0)
            b_kb = b_kk * b_b[:, None]
            b_kb *= b_eg[:, None]
            b_w = tl.dot(b_Ak, b_kb.to(b_kk.dtype)).to(w.dtype.element_ty)
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
            b_v = b_u - b_v
        else:
            b_w = tl.load(w + o_t[:, None] * (HV*K) + o_k1[None, :], mask=m_t[:, None], other=0.0)
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
            b_w = tl.load(w + o_t[:, None] * (HV*K) + o_k2[None, :], mask=m_t[:, None], other=0.0)
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
            b_v = tl.load(u + o_t[:, None] * (HV*V) + o_v[None, :], mask=m_t[:, None] & m_v[None, :], other=0.0) - b_v
        b_vn = b_v.to(u.dtype.element_ty)

        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        b_q = tl.load(q + o_t[:, None] * (H*K) + o_k1[None, :], mask=m_t[:, None], other=0.0)
        b_o += tl.dot(b_q, b_h1.to(b_q.dtype))
        b_q = tl.load(q + o_t[:, None] * (H*K) + o_k2[None, :], mask=m_t[:, None], other=0.0)
        b_o += tl.dot(b_q, b_h2.to(b_q.dtype))
        b_g = tl.load(g + o_t * HV, mask=m_t, other=0.0)
        b_o = b_o * exp2(b_g)[:, None]
        b_A = tl.load(Aqk + o_t[:, None] * (HV*BT) + tl.arange(0, BT)[None, :], mask=m_t[:, None], other=0.0)
        b_d = tl.dot(b_A.to(b_vn.dtype), b_vn)
        if EPI == 0:
            b_o = b_o * scale + b_d * scale
        elif EPI == 1:
            b_o = tl.fma(b_o, scale, b_d * scale)
        elif EPI == 2:
            b_o = tl.fma(b_d, scale, b_o * scale)
        else:
            b_o = (b_o + b_d) * scale
        tl.store(o + o_t[:, None] * (HV*V) + o_v[None, :], b_o.to(o.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])

        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + last_idx * HV).to(tl.float32)
        b_g = b_g.to(tl.float32)
        b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
        b_g_last = exp2(b_g_last)
        b_h1 *= b_g_last
        b_h2 *= b_g_last
        b_v = b_v.to(k.dtype.element_ty)
        b_k = tl.load(k + o_k1[:, None] + o_t[None, :] * (H*K), mask=m_t[None, :], other=0.0)
        b_h1 += tl.dot(b_k, b_v)
        b_k = tl.load(k + o_k2[:, None] + o_t[None, :] * (H*K), mask=m_t[None, :], other=0.0)
        b_h2 += tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        ht = ht + i_nh * K*V
        tl.store(ht + o_k1[:, None] * V + o_v[None, :], b_h1.to(ht.dtype.element_ty), mask=m_v[None, :])
        tl.store(ht + o_k2[:, None] * V + o_v[None, :], b_h2.to(ht.dtype.element_ty), mask=m_v[None, :])


@triton.jit
def _ho_seq(q, k, w, u, g, Aqk, o, bos, eos, i_h, b_h1, b_h2, scale, H: tl.constexpr, HV: tl.constexpr,
            K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr, EPI: tl.constexpr):
    T = eos - bos
    NT = tl.cdiv(T, BT)
    q += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    u += (bos * HV + i_h).to(tl.int64) * V
    o += (bos * HV + i_h).to(tl.int64) * V
    Aqk += (bos * HV + i_h).to(tl.int64) * BT
    g += (bos * HV + i_h).to(tl.int64)
    o_v = tl.program_id(0) * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    o_k2 = 64 + o_k1
    for i_t in range(NT):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        b_w = tl.load(w + o_t[:, None] * (HV*K) + o_k1[None, :], mask=m_t[:, None], other=0.0)
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        b_w = tl.load(w + o_t[:, None] * (HV*K) + o_k2[None, :], mask=m_t[:, None], other=0.0)
        b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        b_v = tl.load(u + o_t[:, None] * (HV*V) + o_v[None, :], mask=m_t[:, None] & m_v[None, :], other=0.0) - b_v
        b_vn = b_v.to(u.dtype.element_ty)

        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        b_q = tl.load(q + o_t[:, None] * (H*K) + o_k1[None, :], mask=m_t[:, None], other=0.0)
        b_o += tl.dot(b_q, b_h1.to(b_q.dtype))
        b_q = tl.load(q + o_t[:, None] * (H*K) + o_k2[None, :], mask=m_t[:, None], other=0.0)
        b_o += tl.dot(b_q, b_h2.to(b_q.dtype))
        b_g = tl.load(g + o_t * HV, mask=m_t, other=0.0)
        b_o = b_o * exp2(b_g)[:, None]
        b_A = tl.load(Aqk + o_t[:, None] * (HV*BT) + tl.arange(0, BT)[None, :], mask=m_t[:, None], other=0.0)
        b_d = tl.dot(b_A.to(b_vn.dtype), b_vn)
        if EPI == 0:
            b_o = b_o * scale + b_d * scale
        elif EPI == 1:
            b_o = tl.fma(b_o, scale, b_d * scale)
        elif EPI == 2:
            b_o = tl.fma(b_d, scale, b_o * scale)
        else:
            b_o = (b_o + b_d) * scale
        tl.store(o + o_t[:, None] * (HV*V) + o_v[None, :], b_o.to(o.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])

        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + last_idx * HV).to(tl.float32)
        b_g = b_g.to(tl.float32)
        b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
        b_g_last = exp2(b_g_last)
        b_h1 *= b_g_last
        b_h2 *= b_g_last
        b_v = b_v.to(k.dtype.element_ty)
        b_k = tl.load(k + o_k1[:, None] + o_t[None, :] * (H*K), mask=m_t[None, :], other=0.0)
        b_h1 += tl.dot(b_k, b_v)
        b_k = tl.load(k + o_k2[:, None] + o_t[None, :] * (H*K), mask=m_t[None, :], other=0.0)
        b_h2 += tl.dot(b_k, b_v)
    return b_h1, b_h2


@triton.jit(do_not_specialize=['T'])
def gdn_ho2_kernel(q, k, w, u, g, Aqk, o, st, cu_seqlens, ch_ptr, ch_idx, scale, T, NP, H: tl.constexpr,
                   HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr,
                   EPI: tl.constexpr):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    b_h1, b_h2 = _ho_seq(q, k, w, u, g, Aqk, o, bos, eos, i_h, b_h1, b_h2, scale, H, HV, K, V, BT, BV, EPI)
    c0, c1 = tl.load(ch_ptr + i_n).to(tl.int32), tl.load(ch_ptr + i_n + 1).to(tl.int32)
    if c1 > c0:
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        o_k1 = tl.arange(0, 64)
        o_k2 = 64 + o_k1
        st += i_nh * K*V
        tl.store(st + o_k1[:, None] * V + o_v[None, :], b_h1.to(st.dtype.element_ty), mask=m_v[None, :])
        tl.store(st + o_k2[:, None] * V + o_v[None, :], b_h2.to(st.dtype.element_ty), mask=m_v[None, :])
        tl.debug_barrier()
        for c in range(c0, c1):
            j = tl.load(ch_idx + c).to(tl.int32)
            sb, se = tl.load(cu_seqlens + NP + j).to(tl.int32), tl.load(cu_seqlens + NP + j + 1).to(tl.int32)
            s_h1 = tl.zeros([64, BV], dtype=tl.float32)
            s_h2 = tl.zeros([64, BV], dtype=tl.float32)
            s_h1 += tl.load(st + o_k1[:, None] * V + o_v[None, :], mask=m_v[None, :], other=0.0).to(tl.float32)
            s_h2 += tl.load(st + o_k2[:, None] * V + o_v[None, :], mask=m_v[None, :], other=0.0).to(tl.float32)
            _ho_seq(q, k, w, u, g, Aqk, o, sb, se, i_h, s_h1, s_h2, scale, H, HV, K, V, BT, BV, EPI)


def gdn_forward(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False, cu_seqlens=None,
                cu_seqlens_cpu=None, bv=None, num_warps=None, num_stages=None, aqk_bk=None, epi=None, fp_fusion=None, fold=None,
                pre=None, gates=None):
    import fla.ops.gated_delta_rule.chunk as fc

    fold = FOLD if fold is None else fold
    pre = PRE if pre is None else pre
    assert gates is None or (pre and not fold), "gates: gdn_pre's prologue only"
    if pre and not fold:
        q, k, v = (x if x.dim() == 4 and x.shape[0] == 1 and x.stride(-1) == 1 and x.stride(-2) == x.shape[-1]
                   else x.contiguous() for x in (q, k, v))
        if gates is None:
            g, beta = g.contiguous(), beta.contiguous()
    else:
        q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    if initial_state is not None:
        initial_state = initial_state.contiguous()
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    if scale is None:
        scale = K ** -0.5
    if pre and not fold:
        import gdn_pre
        chunk_indices = fc.prepare_chunk_indices(cu_seqlens, 64, cu_seqlens_cpu=cu_seqlens_cpu)
        N = len(cu_seqlens) - 1
        q, k, g, w, u, Aqk = gdn_pre.prologue(q, k, v, g, beta, cu_seqlens, chunk_indices, aqk_bk=aqk_bk or AQK_BK,
                                              gates=gates, **(pre if isinstance(pre, dict) else {}))
        return _ho(q, k, v, w, u, g, Aqk, initial_state, output_final_state, cu_seqlens, scale, N, bv, num_warps,
                   num_stages, epi, fp_fusion, False, None, beta)
    q, _ = fc.l2norm_fwd(q)
    k, _ = fc.l2norm_fwd(k)
    chunk_indices = fc.prepare_chunk_indices(cu_seqlens, 64, cu_seqlens_cpu=cu_seqlens_cpu)
    g = fc.chunk_local_cumsum(g, chunk_size=64, scale=fc.RCP_LN2, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    N, NT = len(cu_seqlens) - 1, len(chunk_indices)
    if fold:
        import fla.ops.gated_delta_rule.chunk_fwd as cfw
        Akk = torch.zeros(B, T, HV, 64, device=k.device, dtype=k.dtype)
        cfw.chunk_gated_delta_rule_fwd_kkt_solve_kernel[(NT, B * HV)](
            k=k, g=g, beta=beta, A=Akk, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, T=T, H=H, HV=HV, K=K,
            BT=64, BC=16)
        w, u = k, v
    else:
        Akk = None
        w, u, _ = fc.chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g, beta=beta, cu_seqlens=cu_seqlens,
                                                      chunk_indices=chunk_indices, chunk_size=64)
    Aqk = torch.empty(B, T, HV, 64, device=q.device, dtype=q.dtype)
    gdn_aqk_kernel[(NT, HV)](q, k, g, Aqk, cu_seqlens, chunk_indices, T, H=H, HV=HV, K=K, BT=64,
                             BK=aqk_bk or AQK_BK, num_warps=4, num_stages=2)
    return _ho(q, k, v, w, u, g, Aqk, initial_state, output_final_state, cu_seqlens, scale, N, bv, num_warps, num_stages,
               epi, fp_fusion, fold, Akk, beta)


def gdn_forward_ps(q, k, v, Tp, cu_p, cu_s, cu_all, ci_all, s_par, g=None, beta=None, gates=None):
    import gdn_pre
    if gates is None:
        g, beta = g.contiguous(), beta.contiguous()
    K = k.shape[-1]
    qn, kn, gc, w, u, Aqk = gdn_pre.prologue(q, k, v, g, beta, cu_all, ci_all, aqk_bk=AQK_BK, gates=gates)
    scale = K ** -0.5
    a, b = slice(None, Tp), slice(Tp, None)
    op, st = _ho(qn[:, a], kn[:, a], v[:, a], w[:, a], u[:, a], gc[:, a], Aqk[:, a], None, True, cu_p, scale,
                 len(cu_p) - 1, None, None, None, None, None, False, None, None)
    os_, _ = _ho(qn[:, b], kn[:, b], v[:, b], w[:, b], u[:, b], gc[:, b], Aqk[:, b], st[s_par], False, cu_s, scale,
                 len(cu_s) - 1, None, None, None, None, None, False, None, None)
    return op, os_


def gdn_forward_ps2(q, k, v, Tp, NP, cu_all, ci_all, ch_ptr, ch_idx, g=None, beta=None, gates=None):
    import gdn_pre
    if gates is None:
        g, beta = g.contiguous(), beta.contiguous()
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    qn, kn, gc, w, u, Aqk = gdn_pre.prologue(q, k, v, g, beta, cu_all, ci_all, aqk_bk=AQK_BK, gates=gates)
    o = torch.empty(v.shape, device=v.device, dtype=v.dtype)
    st = k.new_empty(NP, HV, K, V, dtype=torch.float32)
    gdn_ho2_kernel[(triton.cdiv(V, BV), NP * HV)](
        qn, kn, w, u, gc, Aqk, o, st, cu_all, ch_ptr, ch_idx, K ** -0.5, T, NP, H=H, HV=HV, K=K, V=V, BT=64, BV=BV,
        EPI=EPI, num_warps=NW, num_stages=NS, enable_fp_fusion=FUSE)
    return o[:, :Tp], o[:, Tp:]


def _ho(q, k, v, w, u, g, Aqk, initial_state, output_final_state, cu_seqlens, scale, N, bv, num_warps, num_stages, epi,
        fp_fusion, fold, Akk, beta):
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    o = torch.empty(v.shape, device=v.device, dtype=v.dtype)
    final_state = k.new_empty(N, HV, K, V, dtype=torch.float32) if output_final_state else None
    bv = bv or BV
    gdn_ho_kernel[(triton.cdiv(V, bv), N * HV)](
        q, k, w, u, g, Aqk, o, initial_state, final_state, cu_seqlens, scale, T, H=H, HV=HV, K=K, V=V, BT=64, BV=bv,
        USE_INITIAL_STATE=initial_state is not None, STORE_FINAL_STATE=output_final_state, EPI=EPI if epi is None else epi,
        FOLD_WU=fold, Akk=Akk, v=v if fold else None, beta=beta if fold else None,
        num_warps=num_warps or NW, num_stages=num_stages or NS, enable_fp_fusion=FUSE if fp_fusion is None else fp_fusion)
    return o, final_state


def install():
    import fla.ops.gated_delta_rule as M

    orig = M.chunk_gated_delta_rule
    if getattr(orig, "_q35_gdn_ho", False):
        return

    def chunk_gated_delta_rule(q, k, v, g=None, beta=None, scale=None, initial_state=None, output_final_state=False,
                               use_qk_l2norm_in_kernel=False, cu_seqlens=None, cu_seqlens_cpu=None, **kw):
        gates = kw.pop("q35_gates", None)
        if (use_qk_l2norm_in_kernel and cu_seqlens is not None and not kw and q.shape[0] == 1 and q.shape[-1] == 128
                and v.shape[-1] == 128 and (g is not None and beta is not None or gates is not None and PRE)):
            return gdn_forward(q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens, cu_seqlens_cpu,
                               gates=gates)
        assert gates is None, "q35_gates needs Q35_GDN_PRE=1 and gdn_forward's scope"
        return orig(q, k, v, g=g, beta=beta, scale=scale, initial_state=initial_state,
                    output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, **kw)

    chunk_gated_delta_rule._q35_gdn_ho = True
    M.chunk_gated_delta_rule = chunk_gated_delta_rule
    print(f"q35 gdn_ho: fla fwd_h + fwd_o fused (BV {BV}, warps {NW}, stages {NS}, Aqk BK {AQK_BK})"
          f"{'; prologue from gdn_pre.py' if PRE else ''}", flush=True)


if ON:
    install()
