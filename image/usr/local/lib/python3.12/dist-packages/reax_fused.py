from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

BLOCK = 1024


@triton.jit
def _inv_rms(row_ptr, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_ptr + col_idx, mask=mask, other=0.0)
        vals_f32 = vals.to(tl.float32)
        sq_vals = vals_f32 * vals_f32
        sum_sq += tl.sum(tl.where(mask, sq_vals, 0.0))
    mean_sq = sum_sq / n_cols
    rms = tl.sqrt(mean_sq + eps)
    inv_rms = 1.0 / rms
    return inv_rms


@triton.jit
def _norm_chunk(row_ptr, w_ptr, col_idx, mask, inv_rms, HAS_WEIGHT: tl.constexpr):
    vals = tl.load(row_ptr + col_idx, mask=mask, other=0.0)
    out = vals.to(tl.float32) * inv_rms
    if HAS_WEIGHT:
        w = tl.load(w_ptr + col_idx, mask=mask, other=1.0)
        out = out * w.to(tl.float32)
    return out.to(vals.dtype)


@triton.jit
def _norm_add_kernel(x_ptr, w_ptr, r_ptr, o_ptr, x_stride, r_stride, o_stride, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    xr = x_ptr + row * x_stride
    inv = _inv_rms(xr, n_cols, eps, BLOCK_SIZE)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        y = _norm_chunk(xr, w_ptr, col_idx, mask, inv, True)
        r = tl.load(r_ptr + row * r_stride + col_idx, mask=mask, other=0.0)
        o = (y.to(tl.float32) + r.to(tl.float32)).to(y.dtype)
        tl.store(o_ptr + row * o_stride + col_idx, o, mask=mask)


@triton.jit
def _tail_kernel(a_ptr, b_ptr, wa_ptr, wb_ptr, wc_ptr, r_ptr, s_ptr, o_ptr, a_stride, b_stride, r_stride, o_stride,
                 n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    ar = a_ptr + row * a_stride
    br = b_ptr + row * b_stride
    inv_a = _inv_rms(ar, n_cols, eps, BLOCK_SIZE)
    inv_b = _inv_rms(br, n_cols, eps, BLOCK_SIZE)
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        ya = _norm_chunk(ar, wa_ptr, col_idx, mask, inv_a, True)
        yb = _norm_chunk(br, wb_ptr, col_idx, mask, inv_b, True)
        h = (ya.to(tl.float32) + yb.to(tl.float32)).to(ya.dtype)
        h = tl.where(mask, h, 0.0)
        hf = h.to(tl.float32)
        sq_vals = hf * hf
        sum_sq += tl.sum(tl.where(mask, sq_vals, 0.0))
    mean_sq = sum_sq / n_cols
    rms = tl.sqrt(mean_sq + eps)
    inv_h = 1.0 / rms
    s = tl.load(s_ptr).to(tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        ya = _norm_chunk(ar, wa_ptr, col_idx, mask, inv_a, True)
        yb = _norm_chunk(br, wb_ptr, col_idx, mask, inv_b, True)
        h = (ya.to(tl.float32) + yb.to(tl.float32)).to(ya.dtype)
        wc = tl.load(wc_ptr + col_idx, mask=mask, other=1.0)
        yc = ((h.to(tl.float32) * inv_h) * wc.to(tl.float32)).to(h.dtype)
        r = tl.load(r_ptr + row * r_stride + col_idx, mask=mask, other=0.0)
        g = (yc.to(tl.float32) + r.to(tl.float32)).to(h.dtype)
        o = (g.to(tl.float32) * s).to(h.dtype)
        tl.store(o_ptr + row * o_stride + col_idx, o, mask=mask)


@triton.jit
def _router_kernel(x_ptr, w2_ptr, sc_ptr, root_ptr, h2_ptr, rt_ptr, x_stride, o_stride, n_cols, eps,
                   BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    xr = x_ptr + row * x_stride
    inv = _inv_rms(xr, n_cols, eps, BLOCK_SIZE)
    root = tl.load(root_ptr).to(tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        h2 = _norm_chunk(xr, w2_ptr, col_idx, mask, inv, True)
        tl.store(h2_ptr + row * o_stride + col_idx, h2, mask=mask)
        n = _norm_chunk(xr, w2_ptr, col_idx, mask, inv, False)
        t = (n.to(tl.float32) * root).to(n.dtype)
        sc = tl.load(sc_ptr + col_idx, mask=mask, other=1.0)
        rt = (t.to(tl.float32) * sc.to(tl.float32)).to(n.dtype)
        tl.store(rt_ptr + row * o_stride + col_idx, rt, mask=mask)


@triton.jit
def _head_norm_kernel(x_ptr, w_ptr, o_ptr, tok_stride, heads, head_dim, eps, BLOCK_SIZE: tl.constexpr,
                      HAS_WEIGHT: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    t = pid // heads
    hd = pid % heads
    xr = x_ptr + t * tok_stride + hd * head_dim
    inv = _inv_rms(xr, head_dim, eps, BLOCK_SIZE)
    for col_offset in range(0, head_dim, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < head_dim
        y = _norm_chunk(xr, w_ptr, col_idx, mask, inv, HAS_WEIGHT)
        tl.store(o_ptr + pid * head_dim + col_idx, y, mask=mask)


@triton.jit
def _head_norm_rope_kernel(x_ptr, w_ptr, pos_ptr, cache_ptr, o_ptr, tok_stride, heads, head_dim, rot_half, cache_stride,
                           eps, BLOCK_SIZE: tl.constexpr, HALF_BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    t = pid // heads
    hd = pid % heads
    xr = x_ptr + t * tok_stride + hd * head_dim
    inv = _inv_rms(xr, head_dim, eps, BLOCK_SIZE)
    p = tl.load(pos_ptr + t).to(tl.int64)
    i = tl.arange(0, HALF_BLOCK)
    m = i < rot_half
    y1 = _norm_chunk(xr, w_ptr, i, m, inv, True)
    y2 = _norm_chunk(xr, w_ptr, rot_half + i, m, inv, True)
    c = tl.load(cache_ptr + p * cache_stride + i, mask=m, other=0.0).to(tl.float32)
    s = tl.load(cache_ptr + p * cache_stride + rot_half + i, mask=m, other=0.0).to(tl.float32)
    a, b = y1.to(tl.float32), y2.to(tl.float32)
    o1 = (a * c - b * s).to(y1.dtype)
    o2 = (b * c + a * s).to(y1.dtype)
    ob = o_ptr + pid * head_dim
    tl.store(ob + i, o1, mask=m)
    tl.store(ob + rot_half + i, o2, mask=m)


@triton.jit
def _norm_vals(vals, w_ptr, col_idx, mask, inv_rms, HAS_WEIGHT: tl.constexpr):
    out = vals.to(tl.float32) * inv_rms
    if HAS_WEIGHT:
        w = tl.load(w_ptr + col_idx, mask=mask, other=1.0)
        out = out * w.to(tl.float32)
    return out.to(vals.dtype)


@triton.jit
def _moe_sum_chunk(c_row, col_idx, mask, n_cols, TOPK: tl.constexpr, ACC_F32: tl.constexpr):
    acc = tl.load(c_row + col_idx, mask=mask, other=0.0)
    out_dtype = acc.dtype
    if ACC_F32:
        acc = acc.to(tl.float32)
    for k in tl.static_range(1, TOPK):
        v = tl.load(c_row + k * n_cols + col_idx, mask=mask, other=0.0)
        if ACC_F32:
            acc += v.to(tl.float32)
        else:
            acc = (acc.to(tl.float32) + v.to(tl.float32)).to(out_dtype)
    return acc.to(out_dtype)


@triton.jit
def _moe_tail_kernel(a_ptr, c_ptr, wa_ptr, wb_ptr, wc_ptr, r_ptr, s_ptr, wn_ptr, o_ptr, n_ptr, a_stride, r_stride,
                     o_stride, n_cols, eps, BLOCK_SIZE: tl.constexpr, TOPK: tl.constexpr, ACC_F32: tl.constexpr,
                     HAS_NEXT: tl.constexpr, STASH: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    ar = a_ptr + row * a_stride
    cr = c_ptr + row * TOPK * n_cols
    inv_a = _inv_rms(ar, n_cols, eps, BLOCK_SIZE)
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        b = _moe_sum_chunk(cr, col_idx, mask, n_cols, TOPK, ACC_F32)
        if STASH:
            tl.store(o_ptr + row * o_stride + col_idx, b, mask=mask)
        bf = b.to(tl.float32)
        sum_sq += tl.sum(tl.where(mask, bf * bf, 0.0))
    inv_b = 1.0 / tl.sqrt(sum_sq / n_cols + eps)
    if STASH:
        tl.debug_barrier()
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        ya = _norm_chunk(ar, wa_ptr, col_idx, mask, inv_a, True)
        if STASH:
            yb = _norm_chunk(o_ptr + row * o_stride, wb_ptr, col_idx, mask, inv_b, True)
        else:
            yb = _norm_vals(_moe_sum_chunk(cr, col_idx, mask, n_cols, TOPK, ACC_F32), wb_ptr, col_idx, mask, inv_b, True)
        h = (ya.to(tl.float32) + yb.to(tl.float32)).to(ya.dtype)
        hf = tl.where(mask, h, 0.0).to(tl.float32)
        sum_sq += tl.sum(tl.where(mask, hf * hf, 0.0))
    inv_h = 1.0 / tl.sqrt(sum_sq / n_cols + eps)
    s = tl.load(s_ptr).to(tl.float32)
    orow = o_ptr + row * o_stride
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        ya = _norm_chunk(ar, wa_ptr, col_idx, mask, inv_a, True)
        if STASH:
            yb = _norm_chunk(o_ptr + row * o_stride, wb_ptr, col_idx, mask, inv_b, True)
        else:
            yb = _norm_vals(_moe_sum_chunk(cr, col_idx, mask, n_cols, TOPK, ACC_F32), wb_ptr, col_idx, mask, inv_b, True)
        h = (ya.to(tl.float32) + yb.to(tl.float32)).to(ya.dtype)
        yc = _norm_vals(h, wc_ptr, col_idx, mask, inv_h, True)
        r = tl.load(r_ptr + row * r_stride + col_idx, mask=mask, other=0.0)
        g = (yc.to(tl.float32) + r.to(tl.float32)).to(h.dtype)
        o = (g.to(tl.float32) * s).to(h.dtype)
        tl.store(orow + col_idx, o, mask=mask)
        if HAS_NEXT:
            of = tl.where(mask, o, 0.0).to(tl.float32)
            sum_sq += tl.sum(tl.where(mask, of * of, 0.0))
    if HAS_NEXT:
        inv_o = 1.0 / tl.sqrt(sum_sq / n_cols + eps)
        tl.debug_barrier()
        for col_offset in range(0, n_cols, BLOCK_SIZE):
            col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
            mask = col_idx < n_cols
            n = _norm_chunk(orow, wn_ptr, col_idx, mask, inv_o, True)
            tl.store(n_ptr + row * o_stride + col_idx, n, mask=mask)


@triton.jit
def _attn_tail_kernel(x_ptr, wp_ptr, r_ptr, w1_ptr, w2_ptr, sc_ptr, root_ptr, h_ptr, n1_ptr, h2_ptr, rt_ptr, x_stride,
                      r_stride, o_stride, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    xr = x_ptr + row * x_stride
    hr = h_ptr + row * o_stride
    inv_x = _inv_rms(xr, n_cols, eps, BLOCK_SIZE)
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        y = _norm_chunk(xr, wp_ptr, col_idx, mask, inv_x, True)
        r = tl.load(r_ptr + row * r_stride + col_idx, mask=mask, other=0.0)
        h = (y.to(tl.float32) + r.to(tl.float32)).to(y.dtype)
        tl.store(hr + col_idx, h, mask=mask)
        hf = tl.where(mask, h, 0.0).to(tl.float32)
        sum_sq += tl.sum(tl.where(mask, hf * hf, 0.0))
    inv = 1.0 / tl.sqrt(sum_sq / n_cols + eps)
    tl.debug_barrier()
    root = tl.load(root_ptr).to(tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        n1 = _norm_chunk(hr, w1_ptr, col_idx, mask, inv, True)
        tl.store(n1_ptr + row * o_stride + col_idx, n1, mask=mask)
        h2 = _norm_chunk(hr, w2_ptr, col_idx, mask, inv, True)
        tl.store(h2_ptr + row * o_stride + col_idx, h2, mask=mask)
        n = _norm_chunk(hr, w2_ptr, col_idx, mask, inv, False)
        t = (n.to(tl.float32) * root).to(n.dtype)
        sc = tl.load(sc_ptr + col_idx, mask=mask, other=1.0)
        rt = (t.to(tl.float32) * sc.to(tl.float32)).to(n.dtype)
        tl.store(rt_ptr + row * o_stride + col_idx, rt, mask=mask)


@triton.jit
def _kv_cache_kernel(k_ptr, v_ptr, wk_ptr, pos_ptr, rope_ptr, slot_ptr, kc_ptr, tok_stride, heads, head_dim, rot_half,
                     rope_stride, blk_size, c_blk_stride, c_head_stride, c_off_stride, eps_k, eps_v,
                     BLOCK_SIZE: tl.constexpr, HALF_BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    t = pid // heads
    hd = pid % heads
    slot = tl.load(slot_ptr + t).to(tl.int64)
    if slot < 0:
        return
    dst = kc_ptr + (slot // blk_size) * c_blk_stride + hd * c_head_stride + (slot % blk_size) * c_off_stride
    kr = k_ptr + t * tok_stride + hd * head_dim
    inv = _inv_rms(kr, head_dim, eps_k, BLOCK_SIZE)
    p = tl.load(pos_ptr + t).to(tl.int64)
    i = tl.arange(0, HALF_BLOCK)
    m = i < rot_half
    y1 = _norm_chunk(kr, wk_ptr, i, m, inv, True)
    y2 = _norm_chunk(kr, wk_ptr, rot_half + i, m, inv, True)
    c = tl.load(rope_ptr + p * rope_stride + i, mask=m, other=0.0).to(tl.float32)
    s = tl.load(rope_ptr + p * rope_stride + rot_half + i, mask=m, other=0.0).to(tl.float32)
    a, b = y1.to(tl.float32), y2.to(tl.float32)
    tl.store(dst + i, (a * c - b * s).to(y1.dtype), mask=m)
    tl.store(dst + rot_half + i, (b * c + a * s).to(y1.dtype), mask=m)
    vr = v_ptr + t * tok_stride + hd * head_dim
    inv = _inv_rms(vr, head_dim, eps_v, BLOCK_SIZE)
    for col_offset in range(0, head_dim, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < head_dim
        y = _norm_chunk(vr, wk_ptr, col_idx, mask, inv, False)
        tl.store(dst + head_dim + col_idx, y, mask=mask)


def _rows(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2 and x.stride(1) == 1
    return x


def norm_add(x, w, r, eps):
    x, r = _rows(x), _rows(r)
    o = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    _norm_add_kernel[(x.shape[0],)](x, w, r, o, x.stride(0), r.stride(0), o.stride(0), x.shape[1], eps,
                                    BLOCK_SIZE=BLOCK)
    return o


def tail(a, b, wa, wb, wc, r, s, eps):
    a, b, r = _rows(a), _rows(b), _rows(r)
    o = torch.empty(a.shape, device=a.device, dtype=a.dtype)
    _tail_kernel[(a.shape[0],)](a, b, wa, wb, wc, r, s, o, a.stride(0), b.stride(0), r.stride(0), o.stride(0),
                                a.shape[1], eps, BLOCK_SIZE=BLOCK)
    return o


def router(x, w2, scale, root, eps):
    x = _rows(x)
    h2 = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    rt = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    _router_kernel[(x.shape[0],)](x, w2, scale, root, h2, rt, x.stride(0), h2.stride(0), x.shape[1], eps,
                                  BLOCK_SIZE=BLOCK)
    return h2, rt


def head_norm(x, w, heads, head_dim, eps):
    assert x.dim() == 2 and x.stride(1) == 1 and x.shape[1] == heads * head_dim
    o = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    _head_norm_kernel[(x.shape[0] * heads,)](x, w if w is not None else x, o, x.stride(0), heads, head_dim, eps,
                                             BLOCK_SIZE=BLOCK, HAS_WEIGHT=w is not None)
    return o


def head_norm_rope(x, w, positions, cache, heads, head_dim, eps):
    assert x.dim() == 2 and x.stride(1) == 1 and x.shape[1] == heads * head_dim
    assert cache.shape[1] == head_dim and cache.stride(1) == 1 and cache.dtype == x.dtype
    o = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    _head_norm_rope_kernel[(x.shape[0] * heads,)](x, w, positions, cache, o, x.stride(0), heads, head_dim, head_dim // 2,
                                                  cache.stride(0), eps, BLOCK_SIZE=BLOCK,
                                                  HALF_BLOCK=triton.next_power_of_2(head_dim // 2))
    return o


MOE_STASH = True
MOE_ACC_F32 = True


def moe_tail(a, c3, wa, wb, wc, r, s, eps, wn=None):
    a, r = _rows(a), _rows(r)
    assert c3.dim() == 3 and c3.is_contiguous() and c3.shape[0] == a.shape[0] and c3.shape[2] == a.shape[1]
    o = torch.empty(a.shape, device=a.device, dtype=a.dtype)
    n = torch.empty_like(o) if wn is not None else o
    _moe_tail_kernel[(a.shape[0],)](a, c3, wa, wb, wc, r, s, wn if wn is not None else wa, o, n, a.stride(0),
                                    r.stride(0), o.stride(0), a.shape[1], eps, BLOCK_SIZE=BLOCK, TOPK=c3.shape[1],
                                    ACC_F32=MOE_ACC_F32, HAS_NEXT=wn is not None, STASH=MOE_STASH)
    return o, (n if wn is not None else None)


def attn_tail(x, wp, r, w1, w2, scale, root, eps):
    x, r = _rows(x), _rows(r)
    h, n1, h2, rt = (torch.empty(x.shape, device=x.device, dtype=x.dtype) for _ in range(4))
    _attn_tail_kernel[(x.shape[0],)](x, wp, r, w1, w2, scale, root, h, n1, h2, rt, x.stride(0), r.stride(0),
                                     h.stride(0), x.shape[1], eps, BLOCK_SIZE=BLOCK)
    return h, n1, h2, rt


def kv_to_cache(k, v, wk, positions, rope_cache, slot_mapping, kv_cache, heads, head_dim, eps_k, eps_v):
    assert k.stride(1) == 1 and v.stride(1) == 1 and k.stride(0) == v.stride(0)
    assert kv_cache.dim() == 4 and kv_cache.shape[1] == heads and kv_cache.shape[3] == 2 * head_dim
    assert kv_cache.stride(3) == 1 and kv_cache.dtype == k.dtype and rope_cache.shape[1] == head_dim
    n = slot_mapping.shape[0]
    if n == 0:
        return
    _kv_cache_kernel[(n * heads,)](k, v, wk, positions, rope_cache, slot_mapping, kv_cache, k.stride(0), heads,
                                   head_dim, head_dim // 2, rope_cache.stride(0), kv_cache.shape[2],
                                   kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2), eps_k, eps_v,
                                   BLOCK_SIZE=BLOCK, HALF_BLOCK=triton.next_power_of_2(head_dim // 2))


_DEFER = {"T": -1, "c3": None, "out": None}
_moe_sum_orig = None


def _moe_sum_hook(input, output, topk_ids=None, expert_map=None):
    d = _DEFER
    if (d["T"] >= 0 and d["c3"] is None and topk_ids is None and expert_map is None and input.dim() == 3
            and input.shape[0] == d["T"] and input.is_contiguous() and output.shape[0] == d["T"]):
        d["c3"], d["out"] = input, output
        return
    _moe_sum_orig(input, output, topk_ids, expert_map)


def install_moe_sum_hook():
    global _moe_sum_orig
    import vllm._custom_ops as ops
    if _moe_sum_orig is None:
        _moe_sum_orig = ops.moe_sum
        ops.moe_sum = _moe_sum_hook


def moe_deferred(moe, x, logits):
    d = _DEFER
    d["T"], d["c3"], d["out"] = x.shape[0], None, None
    try:
        out = moe(x, logits)
    finally:
        c3, o = d["c3"], d["out"]
        d["T"], d["c3"], d["out"] = -1, None, None
    if c3 is not None and o is not out:
        raise RuntimeError("reax_fused: deferred moe_sum target is not the MoE output; run with REAX_FUSED2=0")
    return out, c3


_PRE = {"x": None, "n": None}


def put_prenorm(x, n):
    _PRE["x"], _PRE["n"] = x, n


def take_prenorm(x):
    n = _PRE["n"] if _PRE["x"] is x else None
    _PRE["x"] = _PRE["n"] = None
    return n
