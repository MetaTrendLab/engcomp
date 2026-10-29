from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.op import exp2
from triton.language.extra import libdevice

ON = os.environ.get("Q35_GDN_PRE", "0") == "1"
NW = int(os.environ.get("Q35_GDN_PRE_WARPS", "1"))
NS = int(os.environ.get("Q35_GDN_PRE_STAGES", "2"))
ILP = os.environ.get("Q35_GDN_PRE_ILP", "1") == "1"
QNORM = {"0": False, "1": True}.get(os.environ.get("Q35_GDN_PRE_QNORM", "1"), "pre")
SPLIT = os.environ.get("Q35_GDN_PRE_SPLIT", "1") == "1"
AQK_NW = int(os.environ.get("Q35_GDN_PRE_AQK_WARPS", "2"))
SOLVE = tl.constexpr('tf32')
GATES = os.environ.get("Q35_GDN_GATES", "0") == "1"


@triton.jit
def _opaque(x):
    return tl.inline_asm_elementwise("mov.b32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _gate_beta(ba, o, m, i_h):
    b = tl.load(ba + o + i_h, mask=m, other=0.0).to(tl.float32)
    return tl.where(m, tl.sigmoid(b), 0.).to(tl.bfloat16)


@triton.jit
def _gate_g(ba, A_neg, dt_bias, o, m, i_h, HV: tl.constexpr):
    x = tl.load(ba + o + HV + i_h, mask=m, other=0.0).to(tl.float32) + tl.load(dt_bias + i_h).to(tl.float32)
    sp = tl.where(x > 20.0, x, libdevice.log1p(libdevice.exp(x)))
    return tl.where(m, _opaque(tl.load(A_neg + i_h) * sp), 0.)


@triton.jit
def _l2norm_tile(x, y, o_t, m_t, eps, RX, RS: tl.constexpr, K: tl.constexpr):
    o_d = tl.arange(0, K)
    p_x = x + o_t[:, None] * RX + o_d[None, :]
    b_x = tl.load(p_x, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
    b_y = b_x * b_rstd[:, None]
    tl.store(y + o_t[:, None] * RS + o_d[None, :], b_y.to(y.dtype.element_ty), mask=m_t[:, None])


@triton.jit(do_not_specialize=['T'])
def gdn_pre_kernel(k_in, g_in, beta, v, k, g, A, w, u, cu_seqlens, chunk_indices, T, scale, eps, RSK, RSV, RSQ,
                   H: tl.constexpr,
                   HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr,
                   BK: tl.constexpr, ILP: tl.constexpr, WU: tl.constexpr = True, QPRE: tl.constexpr = False, q_in=None,
                   q=None, GATES: tl.constexpr = False, ba=None, A_neg=None, dt_bias=None, RSB=1):
    i_t, i_h = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    if GATES:
        b_s = _gate_g(ba + bos * RSB, A_neg, dt_bias, o_t * RSB, m_t, i_h, HV)
    else:
        b_s = tl.load(g_in + bos*HV + i_h + o_t * HV, mask=m_t, other=0.0).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    b_o *= scale
    tl.store(g + bos*HV + i_h + o_t * HV, b_o.to(g.dtype.element_ty), mask=m_t)

    i_kh = i_h // (HV // H)
    _l2norm_tile(k_in + bos * RSK + i_kh * K, k + (bos * H + i_kh) * K, o_t, m_t, eps, RSK, H*K, K)
    if QPRE:
        if i_h % (HV // H) == 0:
            _l2norm_tile(q_in + bos * RSQ + i_kh * K, q + (bos * H + i_kh) * K, o_t, m_t, eps, RSQ, H*K, K)
    tl.debug_barrier()

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    k += (bos * H + i_kh) * K
    A += (bos * HV + i_h) * BT

    o_i = tl.arange(0, BC)
    m_tc0 = (i_tc0 + o_i) < T
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    if GATES:
        b_b0 = _gate_beta(ba + bos * RSB, (i_tc0 + o_i) * RSB, m_tc0, i_h).to(tl.float32)
        b_b1 = _gate_beta(ba + bos * RSB, (i_tc1 + o_i) * RSB, m_tc1, i_h).to(tl.float32)
        b_b2 = _gate_beta(ba + bos * RSB, (i_tc2 + o_i) * RSB, m_tc2, i_h).to(tl.float32)
        b_b3 = _gate_beta(ba + bos * RSB, (i_tc3 + o_i) * RSB, m_tc3, i_h).to(tl.float32)
    else:
        p_b0 = beta + bos * HV + i_h + (i_tc0 + o_i) * HV
        p_b1 = beta + bos * HV + i_h + (i_tc1 + o_i) * HV
        p_b2 = beta + bos * HV + i_h + (i_tc2 + o_i) * HV
        p_b3 = beta + bos * HV + i_h + (i_tc3 + o_i) * HV
        b_b0 = tl.load(p_b0, mask=m_tc0, other=0.0).to(tl.float32)
        b_b1 = tl.load(p_b1, mask=m_tc1, other=0.0).to(tl.float32)
        b_b2 = tl.load(p_b2, mask=m_tc2, other=0.0).to(tl.float32)
        b_b3 = tl.load(p_b3, mask=m_tc3, other=0.0).to(tl.float32)

    p_g0 = g + bos * HV + i_h + (i_tc0 + o_i) * HV
    p_g1 = g + bos * HV + i_h + (i_tc1 + o_i) * HV
    p_g2 = g + bos * HV + i_h + (i_tc2 + o_i) * HV
    p_g3 = g + bos * HV + i_h + (i_tc3 + o_i) * HV
    b_g0 = tl.load(p_g0, mask=m_tc0, other=0.0).to(tl.float32)
    b_g1 = tl.load(p_g1, mask=m_tc1, other=0.0).to(tl.float32)
    b_g2 = tl.load(p_g2, mask=m_tc2, other=0.0).to(tl.float32)
    b_g3 = tl.load(p_g3, mask=m_tc3, other=0.0).to(tl.float32)

    b_A00 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A11 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A22 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A33 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A32 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        p_k0 = k + (i_tc0 + o_i)[:, None] * (H*K) + o_k[None, :]
        b_k0 = tl.load(p_k0, mask=m_tc0[:, None] & (o_k[None, :] < K), other=0.0)
        b_A00 += tl.dot(b_k0, tl.trans(b_k0))
        if i_tc1 < T:
            p_k1 = k + (i_tc1 + o_i)[:, None] * (H*K) + o_k[None, :]
            b_k1 = tl.load(p_k1, mask=m_tc1[:, None] & (o_k[None, :] < K), other=0.0)
            b_A11 += tl.dot(b_k1, tl.trans(b_k1))
            b_A10 += tl.dot(b_k1, tl.trans(b_k0))
            if i_tc2 < T:
                p_k2 = k + (i_tc2 + o_i)[:, None] * (H*K) + o_k[None, :]
                b_k2 = tl.load(p_k2, mask=m_tc2[:, None] & (o_k[None, :] < K), other=0.0)
                b_A22 += tl.dot(b_k2, tl.trans(b_k2))
                b_A20 += tl.dot(b_k2, tl.trans(b_k0))
                b_A21 += tl.dot(b_k2, tl.trans(b_k1))
                if i_tc3 < T:
                    p_k3 = k + (i_tc3 + o_i)[:, None] * (H*K) + o_k[None, :]
                    b_k3 = tl.load(p_k3, mask=m_tc3[:, None] & (o_k[None, :] < K), other=0.0)
                    b_A33 += tl.dot(b_k3, tl.trans(b_k3))
                    b_A30 += tl.dot(b_k3, tl.trans(b_k0))
                    b_A31 += tl.dot(b_k3, tl.trans(b_k1))
                    b_A32 += tl.dot(b_k3, tl.trans(b_k2))

    m_d = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    b_A00 *= tl.where(m_d & m_tc0[:, None] & m_tc0[None, :], exp2(b_g0[:, None] - b_g0[None, :]), 0.)
    b_A11 *= tl.where(m_d & m_tc1[:, None] & m_tc1[None, :], exp2(b_g1[:, None] - b_g1[None, :]), 0.)
    b_A22 *= tl.where(m_d & m_tc2[:, None] & m_tc2[None, :], exp2(b_g2[:, None] - b_g2[None, :]), 0.)
    b_A33 *= tl.where(m_d & m_tc3[:, None] & m_tc3[None, :], exp2(b_g3[:, None] - b_g3[None, :]), 0.)
    b_A10 *= tl.where(m_tc1[:, None] & m_tc0[None, :], exp2(b_g1[:, None] - b_g0[None, :]), 0.)
    b_A20 *= tl.where(m_tc2[:, None] & m_tc0[None, :], exp2(b_g2[:, None] - b_g0[None, :]), 0.)
    b_A21 *= tl.where(m_tc2[:, None] & m_tc1[None, :], exp2(b_g2[:, None] - b_g1[None, :]), 0.)
    b_A30 *= tl.where(m_tc3[:, None] & m_tc0[None, :], exp2(b_g3[:, None] - b_g0[None, :]), 0.)
    b_A31 *= tl.where(m_tc3[:, None] & m_tc1[None, :], exp2(b_g3[:, None] - b_g1[None, :]), 0.)
    b_A32 *= tl.where(m_tc3[:, None] & m_tc2[None, :], exp2(b_g3[:, None] - b_g2[None, :]), 0.)

    b_A00 = b_A00 * b_b0[:, None]
    b_A11 = b_A11 * b_b1[:, None]
    b_A22 = b_A22 * b_b2[:, None]
    b_A33 = b_A33 * b_b3[:, None]
    b_A10 = b_A10 * b_b1[:, None]
    b_A20 = b_A20 * b_b2[:, None]
    b_A21 = b_A21 * b_b2[:, None]
    b_A30 = b_A30 * b_b3[:, None]
    b_A31 = b_A31 * b_b3[:, None]
    b_A32 = b_A32 * b_b3[:, None]

    b_Ai00 = -b_A00
    b_Ai11 = -b_A11
    b_Ai22 = -b_A22
    b_Ai33 = -b_A33
    if ILP:
        n1 = T - i_tc1
        n2 = T - i_tc2
        n3 = T - i_tc3
        for i in range(2, min(BC, T - i_tc0)):
            b_a00 = tl.sum(tl.where((o_i == i)[:, None], -b_A00, 0.), 0)
            b_a11 = tl.sum(tl.where((o_i == i)[:, None], -b_A11, 0.), 0)
            b_a22 = tl.sum(tl.where((o_i == i)[:, None], -b_A22, 0.), 0)
            b_a33 = tl.sum(tl.where((o_i == i)[:, None], -b_A33, 0.), 0)
            b_a00 = tl.where(o_i < i, b_a00, 0.)
            b_a11 = tl.where(o_i < i, b_a11, 0.)
            b_a22 = tl.where(o_i < i, b_a22, 0.)
            b_a33 = tl.where(o_i < i, b_a33, 0.)
            b_a00 = b_a00 + tl.sum(b_a00[:, None] * b_Ai00, 0)
            b_a11 = b_a11 + tl.sum(b_a11[:, None] * b_Ai11, 0)
            b_a22 = b_a22 + tl.sum(b_a22[:, None] * b_Ai22, 0)
            b_a33 = b_a33 + tl.sum(b_a33[:, None] * b_Ai33, 0)
            b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
            b_Ai11 = tl.where((o_i == i)[:, None] & (i < n1), b_a11, b_Ai11)
            b_Ai22 = tl.where((o_i == i)[:, None] & (i < n2), b_a22, b_Ai22)
            b_Ai33 = tl.where((o_i == i)[:, None] & (i < n3), b_a33, b_Ai33)
    else:
        for i in range(2, min(BC, T - i_tc0)):
            b_a00 = tl.sum(tl.where((o_i == i)[:, None], -b_A00, 0.), 0)
            b_a00 = tl.where(o_i < i, b_a00, 0.)
            b_a00 = b_a00 + tl.sum(b_a00[:, None] * b_Ai00, 0)
            b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
        for i in range(2, min(BC, T - i_tc1)):
            b_a11 = tl.sum(tl.where((o_i == i)[:, None], -b_A11, 0.), 0)
            b_a11 = tl.where(o_i < i, b_a11, 0.)
            b_a11 = b_a11 + tl.sum(b_a11[:, None] * b_Ai11, 0)
            b_Ai11 = tl.where((o_i == i)[:, None], b_a11, b_Ai11)
        for i in range(2, min(BC, T - i_tc2)):
            b_a22 = tl.sum(tl.where((o_i == i)[:, None], -b_A22, 0.), 0)
            b_a22 = tl.where(o_i < i, b_a22, 0.)
            b_a22 = b_a22 + tl.sum(b_a22[:, None] * b_Ai22, 0)
            b_Ai22 = tl.where((o_i == i)[:, None], b_a22, b_Ai22)
        for i in range(2, min(BC, T - i_tc3)):
            b_a33 = tl.sum(tl.where((o_i == i)[:, None], -b_A33, 0.), 0)
            b_a33 = tl.where(o_i < i, b_a33, 0.)
            b_a33 = b_a33 + tl.sum(b_a33[:, None] * b_Ai33, 0)
            b_Ai33 = tl.where((o_i == i)[:, None], b_a33, b_Ai33)

    b_Ai00 += m_I
    b_Ai11 += m_I
    b_Ai22 += m_I
    b_Ai33 += m_I

    b_Ai10 = -tl.dot(tl.dot(b_Ai11, b_A10, input_precision=SOLVE), b_Ai00, input_precision=SOLVE)
    b_Ai21 = -tl.dot(tl.dot(b_Ai22, b_A21, input_precision=SOLVE), b_Ai11, input_precision=SOLVE)
    b_Ai32 = -tl.dot(tl.dot(b_Ai33, b_A32, input_precision=SOLVE), b_Ai22, input_precision=SOLVE)
    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_A20, b_Ai00, input_precision=SOLVE) +
        tl.dot(b_A21, b_Ai10, input_precision=SOLVE),
        input_precision=SOLVE,
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_A31, b_Ai11, input_precision=SOLVE) +
        tl.dot(b_A32, b_Ai21, input_precision=SOLVE),
        input_precision=SOLVE,
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_A30, b_Ai00, input_precision=SOLVE) +
        tl.dot(b_A31, b_Ai10, input_precision=SOLVE) +
        tl.dot(b_A32, b_Ai20, input_precision=SOLVE),
        input_precision=SOLVE,
    )

    m_A0 = m_tc0[:, None] & (o_i[None, :] < BT)
    m_A1 = m_tc1[:, None] & (o_i[None, :] < BT)
    m_A2 = m_tc2[:, None] & (o_i[None, :] < BT)
    m_A3 = m_tc3[:, None] & (o_i[None, :] < BT)
    b_z = tl.zeros([BC, BC], dtype=tl.float32).to(A.dtype.element_ty)
    r0 = (i_tc0 + o_i)[:, None] * (HV*BT)
    r1 = (i_tc1 + o_i)[:, None] * (HV*BT)
    r2 = (i_tc2 + o_i)[:, None] * (HV*BT)
    r3 = (i_tc3 + o_i)[:, None] * (HV*BT)
    tl.store(A + r0 + o_i[None, :], b_Ai00.to(A.dtype.element_ty), mask=m_A0)
    tl.store(A + r0 + (BC + o_i)[None, :], b_z, mask=m_A0)
    tl.store(A + r0 + (2*BC + o_i)[None, :], b_z, mask=m_A0)
    tl.store(A + r0 + (3*BC + o_i)[None, :], b_z, mask=m_A0)
    tl.store(A + r1 + o_i[None, :], b_Ai10.to(A.dtype.element_ty), mask=m_A1)
    tl.store(A + r1 + (BC + o_i)[None, :], b_Ai11.to(A.dtype.element_ty), mask=m_A1)
    tl.store(A + r1 + (2*BC + o_i)[None, :], b_z, mask=m_A1)
    tl.store(A + r1 + (3*BC + o_i)[None, :], b_z, mask=m_A1)
    tl.store(A + r2 + o_i[None, :], b_Ai20.to(A.dtype.element_ty), mask=m_A2)
    tl.store(A + r2 + (BC + o_i)[None, :], b_Ai21.to(A.dtype.element_ty), mask=m_A2)
    tl.store(A + r2 + (2*BC + o_i)[None, :], b_Ai22.to(A.dtype.element_ty), mask=m_A2)
    tl.store(A + r2 + (3*BC + o_i)[None, :], b_z, mask=m_A2)
    tl.store(A + r3 + o_i[None, :], b_Ai30.to(A.dtype.element_ty), mask=m_A3)
    tl.store(A + r3 + (BC + o_i)[None, :], b_Ai31.to(A.dtype.element_ty), mask=m_A3)
    tl.store(A + r3 + (2*BC + o_i)[None, :], b_Ai32.to(A.dtype.element_ty), mask=m_A3)
    tl.store(A + r3 + (3*BC + o_i)[None, :], b_Ai33.to(A.dtype.element_ty), mask=m_A3)
    if WU:
        tl.debug_barrier()
        _w_u(k, v, beta, g, A, w, u, bos, i_h, o_t, m_t, RSV, H, HV, K, V, BT, GATES, ba, RSB)


@triton.jit
def _w_u(k, v, beta, g, A, w, u, bos, i_h, o_t, m_t, RSV, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
         V: tl.constexpr, BT: tl.constexpr, GATES: tl.constexpr, ba, RSB):
    o_A = tl.arange(0, BT)
    m_A = m_t[:, None] & (o_A[None, :] < BT)
    if GATES:
        b_b = _gate_beta(ba + bos * RSB, o_t * RSB, m_t, i_h)
    else:
        b_b = tl.load(beta + bos*HV + i_h + o_t * HV, mask=m_t, other=0.0)
    b_A = tl.load(A + o_t[:, None] * (HV*BT) + o_A[None, :], mask=m_A, other=0.0)
    for i_v in range(tl.cdiv(V, 64)):
        o_v = i_v * 64 + tl.arange(0, 64)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        p_v = v + bos * RSV + i_h * V + o_t[:, None] * RSV + o_v[None, :]
        p_u = u + (bos*HV + i_h) * V + o_t[:, None] * (HV*V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_v, other=0.0)
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), mask=m_v)
    b_eg = exp2(tl.load(g + (bos*HV + i_h) + o_t * HV, mask=m_t, other=0.0))
    for i_k in range(tl.cdiv(K, 64)):
        o_k = i_k * 64 + tl.arange(0, 64)
        m_k = m_t[:, None] & (o_k[None, :] < K)
        p_k = k + o_t[:, None] * (H*K) + o_k[None, :]
        p_w = w + (bos*HV + i_h) * K + o_t[:, None] * (HV*K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_k, other=0.0)
        b_kb = b_k * b_b[:, None]
        b_kb *= b_eg[:, None]
        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), mask=m_k)


@triton.jit(do_not_specialize=['T'])
def gdn_aqkn_kernel(q_in, q, k, g, A, cu_seqlens, chunk_indices, T, eps, RSQ, RSV, H: tl.constexpr, HV: tl.constexpr,
                    K: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, QNORM: tl.constexpr, WU: tl.constexpr = False,
                    V: tl.constexpr = 128, Akk=None, v=None, beta=None, w=None, u=None, GATES: tl.constexpr = False,
                    ba=None, RSB=1):
    i_t, i_h = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    if WU:
        _w_u(k + (bos * H + i_h // (HV // H)) * K, v, beta, g, Akk + (bos * HV + i_h) * BT, w, u, bos, i_h, o_t, m_t, RSV,
             H, HV, K, V, BT, GATES, ba, RSB)
    if QNORM:
        _l2norm_tile(q_in + bos * RSQ + (i_h // (HV // H)) * K, q + (bos * H + i_h // (HV // H)) * K, o_t, m_t, eps, RSQ,
                     H*K, K)
        tl.debug_barrier()
    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    g += bos * HV + i_h
    A += (bos * HV + i_h) * BT

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
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


def prologue(q, k, v, g, beta, cu_seqlens, chunk_indices, aqk_bk=64, num_warps=None, num_stages=None, ilp=None,
             qnorm=None, split=None, aqk_warps=None, eps=1e-6, gates=None):
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    NT = len(chunk_indices)
    for t, n in ((q, K), (k, K), (v, V)):
        assert t.stride(-1) == 1 and t.stride(-2) == n and B == 1, "gdn_pre: [1, T, heads, dim] with dense rows"
    kn = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    gc = torch.empty(B, T, HV, device=k.device, dtype=torch.float)
    ga = {}
    if gates is not None:
        ba, A_neg, dt_bias = gates
        assert ba.shape == (T, 2 * HV) and ba.stride(1) == 1 and A_neg.dtype == torch.float32, (ba.shape, A_neg.dtype)
        ga = dict(GATES=True, ba=ba, A_neg=A_neg, dt_bias=dt_bias, RSB=ba.stride(0))
    A = torch.empty(B, T, HV, 64, device=k.device, dtype=k.dtype)
    w = k.new_empty(B, T, HV, K)
    u = torch.empty(v.shape, device=v.device, dtype=v.dtype)
    split = SPLIT if split is None else split
    qnorm = QNORM if qnorm is None else qnorm
    qn = torch.empty(q.shape, device=q.device, dtype=q.dtype) if qnorm else None
    gdn_pre_kernel[(NT, B * HV)](k, g, beta, v, kn, gc, A, w, u, cu_seqlens, chunk_indices, T, RCP_LN2, eps, k.stride(1),
                                 v.stride(1), q.stride(1), H=H, HV=HV,
                                 K=K, V=V, BT=64, BC=16, BK=64, ILP=ILP if ilp is None else ilp, WU=not split,
                                 QPRE=qnorm == "pre", q_in=q, q=qn, num_warps=num_warps or NW,
                                 num_stages=num_stages or NS, **ga)
    if qnorm == "pre":
        qnorm = False
    elif not qnorm:
        import fla.ops.gated_delta_rule.chunk as fc
        qn, _ = fc.l2norm_fwd(q.contiguous())
    Aqk = torch.empty(B, T, HV, 64, device=q.device, dtype=q.dtype)
    gdn_aqkn_kernel[(NT, HV)](q, qn, kn, gc, Aqk, cu_seqlens, chunk_indices, T, eps, q.stride(1), v.stride(1), H=H,
                              HV=HV, K=K, BT=64,
                              BK=aqk_bk, QNORM=qnorm, WU=split, V=V, Akk=A, v=v, beta=beta, w=w, u=u,
                              num_warps=aqk_warps or AQK_NW, num_stages=2,
                              **({k_: ga[k_] for k_ in ("GATES", "ba", "RSB")} if ga and split else {}))
    return qn, kn, gc, w, u, Aqk
