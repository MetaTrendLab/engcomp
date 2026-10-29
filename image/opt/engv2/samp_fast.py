import sys

LOGPROB_OLD = "    _topk_log_softmax_kernel[(batch_size,)](\n"
LOGPROB_NEW = "    (_reax_lsm_launch if _REAX_SAMP else _topk_log_softmax_kernel[(batch_size,)])(\n"
RANKS_OLD = "    _ranks_kernel[(batch_size,)](\n"
RANKS_NEW = "    (_reax_ranks_launch if _REAX_SAMP else _ranks_kernel[(batch_size,)])(\n"
BIAS_OLD = "    _bias_kernel[(num_tokens,)](\n"
BIAS_NEW = "    (_reax_bias_launch if _REAX_SAMP else _bias_kernel[(num_tokens,)])(\n"
GUMBEL_OLD = "    _gumbel_sample_kernel[(num_tokens, num_blocks)](\n"
GUMBEL_NEW = "    (_reax_gumbel_launch if _REAX_SAMP else _gumbel_sample_kernel[(num_tokens, num_blocks)])(\n"
HEAD_OLD = '    sub = lm_head.quant_method.apply(stub, h, bias=None)[..., : st["n"]]\n'
HEAD_NEW = ('    sub = (_reax_head_mm(h, st["w"]) if _REAX_HEAD_TILE and h.dim() == 2 else\n'
            '           lm_head.quant_method.apply(stub, h, bias=None))[..., : st["n"]]\n')

COMMON = '''

# ---------------------------------------------------------------- engv2 Blink: sampler tail on many CTAs (samp_fast.py)
import logging as _reax_logging
import os as _reax_os

_REAX_SAMP = _reax_os.environ.get("REAX_SAMP_FAST", "0") == "1"
_REAX_SAMP_CHECKS = int(_reax_os.environ.get("REAX_SAMP_FAST_CHECKS", "3"))
_reax_samp_log = _reax_logging.getLogger("vllm.reax_samp_fast")
_REAX_SAMP_ST = {}


def _reax_samp_check(name):
    """True while `name` still has bit-for-bit checks left (and no graph capture is running)."""
    st = _REAX_SAMP_ST.setdefault(name, {"left": _REAX_SAMP_CHECKS, "off": False})
    return st["left"] > 0 and not torch.cuda.is_current_stream_capturing()


def _reax_samp_verdict(name, same, what):
    st = _REAX_SAMP_ST[name]
    st["left"] -= 1
    if not same:
        st["off"] = True
        _reax_samp_log.warning("reax samp_fast: %s differs from the original (%s): off", name, what)
    elif st["left"] == 0:
        _reax_samp_log.info("reax samp_fast: %s bit-identical to the original on %d calls", name, _REAX_SAMP_CHECKS)


def _reax_samp_off(name):
    return _REAX_SAMP_ST.get(name, {}).get("off", False)


@triton.jit
def _reax_dev_copy2_kernel(s0_ptr, d0_ptr, n0, s1_ptr, d1_ptr, n1, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    tl.store(d0_ptr + o, tl.load(s0_ptr + o, mask=o < n0), mask=o < n0)
    tl.store(d1_ptr + o, tl.load(s1_ptr + o, mask=o < n1), mask=o < n1)


def _reax_dev_copy(*ts):
    """Device copies of small per-request state tensors (vLLM keeps them in UVA host memory: a wide kernel that reads
    them once per CTA pays a PCIe round trip in every CTA); one launch for up to two tensors, same values."""
    out = [torch.empty_like(t) for t in ts]
    a, b = (ts + ts[:1])[:2], (out + out[:1])[:2]
    n1 = ts[1].numel() if len(ts) > 1 else 0
    _reax_dev_copy2_kernel[(1,)](a[0], b[0], ts[0].numel(), a[1], b[1], n1,
                                 BLOCK=triton.next_power_of_2(max(t.numel() for t in ts)))
    return out


def _reax_bits_equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype in (torch.float32, torch.int32):
        return torch.equal(a.view(torch.int32), b.view(torch.int32))
    return torch.equal(a, b)
'''

LOGPROB_ADD = COMMON + '''

@triton.jit
def _reax_lsm_bmax_kernel(ws_ptr, cnt_ptr, logits_ptr, logits_stride, vocab_size, NB, BLOCK_SIZE: tl.constexpr):
    req_idx = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    row_ptr = logits_ptr + req_idx * logits_stride
    block = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
    tl.store(ws_ptr + req_idx * 2 * NB + b, tl.max(logits).to(tl.float32))
    if b == 0:
        tl.store(cnt_ptr + req_idx, 0)


@triton.jit
def _reax_lsm_bsum_kernel(output_ptr, ws_ptr, cnt_ptr, logits_ptr, logits_stride, topk_ids_ptr, topk, vocab_size,
                          NB: tl.constexpr, NBP: tl.constexpr, BLOCK_SIZE: tl.constexpr, TOPK_BLOCK_SIZE: tl.constexpr):
    req_idx = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    row_ptr = logits_ptr + req_idx * logits_stride
    nb = tl.arange(0, NBP)
    max_val = tl.max(tl.load(ws_ptr + req_idx * 2 * NB + nb, mask=nb < NB, other=float("-inf")))

    # the original's second loop, one iteration: the same load, exp and tl.sum over the same 1024-element block
    block = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    logits = tl.load(row_ptr + block, mask=block < vocab_size, other=0.0)
    logits = logits.to(tl.float32)
    e = tl.exp(logits - max_val)
    e = tl.where(block < vocab_size, e, 0.0)
    tl.store(ws_ptr + req_idx * 2 * NB + NB + b, tl.sum(e))

    # the last CTA of the row: se += s_b in block order (the original's scalar chain), then the original's tail; the
    # loads are unrolled so they issue together (only the adds are a chain)
    tl.debug_barrier()
    done = tl.atomic_add(cnt_ptr + req_idx, 1, sem="acq_rel", scope="gpu")
    if done == NB - 1:
        se = 0.0
        for i in tl.static_range(NB):
            se += tl.load(ws_ptr + req_idx * 2 * NB + NB + i, cache_modifier=".cg")
        lse = tl.log(se)

        for j in range(0, topk, TOPK_BLOCK_SIZE):
            k_offset = j + tl.arange(0, TOPK_BLOCK_SIZE)
            k_mask = k_offset < topk
            topk_ids = tl.load(topk_ids_ptr + req_idx * topk + k_offset, mask=k_mask, other=0)
            gl = tl.load(row_ptr + topk_ids, mask=k_mask)
            gl = gl.to(tl.float32)
            o = gl - max_val - lse
            tl.store(output_ptr + req_idx * topk + k_offset, o, mask=k_mask)


def _reax_lsm_run(output, logits, logits_stride, topk_ids, topk, vocab_size, BLOCK_SIZE, TOPK_BLOCK_SIZE):
    B = output.shape[0]
    NB = triton.cdiv(vocab_size, BLOCK_SIZE)
    ws = torch.empty((B, 2 * NB), dtype=torch.float32, device=logits.device)
    cnt = torch.empty((B,), dtype=torch.int32, device=logits.device)
    _reax_lsm_bmax_kernel[(B, NB)](ws, cnt, logits, logits_stride, vocab_size, NB, BLOCK_SIZE=BLOCK_SIZE)
    _reax_lsm_bsum_kernel[(B, NB)](output, ws, cnt, logits, logits_stride, topk_ids, topk, vocab_size, NB,
                                   NBP=triton.next_power_of_2(NB), BLOCK_SIZE=BLOCK_SIZE,
                                   TOPK_BLOCK_SIZE=TOPK_BLOCK_SIZE)


def _reax_lsm_launch(output, logits, logits_stride, topk_ids, topk, vocab_size, BLOCK_SIZE, TOPK_BLOCK_SIZE):
    B = output.shape[0]
    args = (logits, logits_stride, topk_ids, topk, vocab_size)
    if _reax_samp_off("logprob") or B == 0:
        _topk_log_softmax_kernel[(B,)](output, *args, BLOCK_SIZE=BLOCK_SIZE, TOPK_BLOCK_SIZE=TOPK_BLOCK_SIZE)
        return
    _reax_lsm_run(output, *args, BLOCK_SIZE, TOPK_BLOCK_SIZE)
    if _reax_samp_check("logprob"):
        ref = torch.empty_like(output)
        _topk_log_softmax_kernel[(B,)](ref, *args, BLOCK_SIZE=BLOCK_SIZE, TOPK_BLOCK_SIZE=TOPK_BLOCK_SIZE)
        same = _reax_bits_equal(ref, output)
        _reax_samp_verdict("logprob", same, f"B {B}")
        if not same:
            output.copy_(ref)


@triton.jit
def _reax_ranks_chunk_kernel(output_ptr, logits_ptr, logits_stride, token_ids_ptr, vocab_size, BLOCK_SIZE: tl.constexpr):
    req_idx = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    row_ptr = logits_ptr + req_idx * logits_stride
    token_id = tl.load(token_ids_ptr + req_idx)
    x = tl.load(row_ptr + token_id)
    block = c * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
    n = tl.sum((logits >= x).to(tl.int32))
    tl.atomic_add(output_ptr + req_idx, n.to(tl.int64), sem="relaxed")


def _reax_ranks_launch(output, logits, logits_stride, token_ids, vocab_size, BLOCK_SIZE):
    B = output.shape[0]
    if _reax_samp_off("ranks") or B == 0:
        _ranks_kernel[(B,)](output, logits, logits_stride, token_ids, vocab_size, BLOCK_SIZE=BLOCK_SIZE)
        return
    output.zero_()
    _reax_ranks_chunk_kernel[(B, triton.cdiv(vocab_size, BLOCK_SIZE))](output, logits, logits_stride, token_ids,
                                                                     vocab_size, BLOCK_SIZE=BLOCK_SIZE)
    if _reax_samp_check("ranks"):
        ref = torch.empty_like(output)
        _ranks_kernel[(B,)](ref, logits, logits_stride, token_ids, vocab_size, BLOCK_SIZE=BLOCK_SIZE)
        same = torch.equal(ref, output)
        _reax_samp_verdict("ranks", same, f"B {B}")
        if not same:
            output.copy_(ref)
'''

BIAS_ADD = COMMON + '''

@triton.jit
def _reax_allowed_chunk_kernel(logits_ptr, logits_stride, vocab_size, expanded_idx_mapping_ptr,
                               num_allowed_token_ids_ptr, allowed_token_ids_ptr, allowed_token_ids_stride,
                               BLOCK_SIZE: tl.constexpr, LOGITS_BLOCK_SIZE: tl.constexpr):
    token_idx = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    num_allowed_token_ids = tl.load(num_allowed_token_ids_ptr + req_state_idx)
    if num_allowed_token_ids > 0:
        lo = c * LOGITS_BLOCK_SIZE
        block = tl.arange(0, BLOCK_SIZE)
        allowed_token_ids = tl.load(allowed_token_ids_ptr + req_state_idx * allowed_token_ids_stride + block,
                                    mask=block < num_allowed_token_ids)
        # this chunk's allowed ids only: saved, the chunk -inf'd, restored (the original does it for the whole row)
        mask = (block < num_allowed_token_ids) & (allowed_token_ids >= lo) & (allowed_token_ids < lo + LOGITS_BLOCK_SIZE)
        logits = tl.load(logits_ptr + token_idx * logits_stride + allowed_token_ids, mask=mask)
        tl.debug_barrier()
        offset = lo + tl.arange(0, LOGITS_BLOCK_SIZE)
        tl.store(logits_ptr + token_idx * logits_stride + offset, -float("inf"), mask=offset < vocab_size)
        tl.debug_barrier()
        tl.store(logits_ptr + token_idx * logits_stride + allowed_token_ids, logits, mask=mask)


_REAX_ZEROS = {}


def _reax_bias_run(logits, logits_stride, vocab_size, expanded_idx_mapping, num_allowed_token_ids, allowed_token_ids,
                   allowed_token_ids_stride, *rest, BLOCK_SIZE, LOGITS_BLOCK_SIZE, CHECK_ALL_MASKED_ROWS):
    T = logits.shape[0]
    (na_dev,) = _reax_dev_copy(num_allowed_token_ids)
    _reax_allowed_chunk_kernel[(T, triton.cdiv(vocab_size, LOGITS_BLOCK_SIZE))](
        logits, logits_stride, vocab_size, expanded_idx_mapping, na_dev, allowed_token_ids,
        allowed_token_ids_stride, BLOCK_SIZE=BLOCK_SIZE, LOGITS_BLOCK_SIZE=LOGITS_BLOCK_SIZE)
    key = (num_allowed_token_ids.shape, num_allowed_token_ids.dtype, num_allowed_token_ids.device)
    zeros = _REAX_ZEROS.get(key)
    if zeros is None:
        zeros = _REAX_ZEROS[key] = torch.zeros_like(num_allowed_token_ids)
    # logit bias and min tokens: the original kernel, its allowed-token part skipped (count 0)
    _bias_kernel[(T,)](logits, logits_stride, vocab_size, expanded_idx_mapping, zeros, allowed_token_ids,
                       allowed_token_ids_stride, *rest, BLOCK_SIZE=BLOCK_SIZE, LOGITS_BLOCK_SIZE=LOGITS_BLOCK_SIZE,
                       CHECK_ALL_MASKED_ROWS=CHECK_ALL_MASKED_ROWS)


def _reax_bias_launch(logits, *args, BLOCK_SIZE, LOGITS_BLOCK_SIZE, CHECK_ALL_MASKED_ROWS):
    T = logits.shape[0]
    kw = dict(BLOCK_SIZE=BLOCK_SIZE, LOGITS_BLOCK_SIZE=LOGITS_BLOCK_SIZE, CHECK_ALL_MASKED_ROWS=CHECK_ALL_MASKED_ROWS)
    if _reax_samp_off("bias") or T == 0:
        _bias_kernel[(T,)](logits, *args, **kw)
        return
    ref = None
    if _reax_samp_check("bias"):
        ref = logits.clone()
        _bias_kernel[(T,)](ref, *args, **kw)
    _reax_bias_run(logits, *args, **kw)
    if ref is not None:
        same = _reax_bits_equal(ref, logits)
        _reax_samp_verdict("bias", same, f"T {T}")
        if not same:
            logits.copy_(ref)
'''

GUMBEL_ADD = COMMON + '''

def _reax_gumbel_launch(local_argmax, la_stride, local_max, lm_stride, logits_cache, lc_s0, lc_s1, logits_cache_col, logits,
                        logits_stride, expanded_idx_mapping, seed, pos, temperature, vocab_size, **kw):
    """The original kernel, its per-request temperature / seed read from device copies instead of UVA (same values)."""
    grid = (logits.shape[0], triton.cdiv(vocab_size, kw["BLOCK_SIZE"]))
    head = (local_argmax, la_stride, local_max, lm_stride, logits_cache, lc_s0, lc_s1, logits_cache_col, logits,
            logits_stride, expanded_idx_mapping)
    if _reax_samp_off("gumbel") or grid[0] == 0:
        _gumbel_sample_kernel[grid](*head, seed, pos, temperature, vocab_size, **kw)
        return
    temp_dev, seed_dev = _reax_dev_copy(temperature, seed)
    _gumbel_sample_kernel[grid](*head, seed_dev, pos, temp_dev, vocab_size, **kw)
    if _reax_samp_check("gumbel") and logits_cache is None:
        ra, rm = torch.empty_like(local_argmax), torch.empty_like(local_max)
        _gumbel_sample_kernel[grid](ra, la_stride, rm, lm_stride, *head[4:], seed, pos, temperature, vocab_size, **kw)
        same = torch.equal(ra, local_argmax) and torch.equal(rm.view(torch.int32) if rm.dtype == torch.float32 else rm,
                                                             local_max.view(torch.int32) if rm.dtype == torch.float32
                                                             else local_max)
        _reax_samp_verdict("gumbel", same, f"T {grid[0]}")
        if not same:
            local_argmax.copy_(ra)
            local_max.copy_(rm)
'''

HEAD_ADD = '''

# ---------------------------------------------------------------- engv2 Blink: head_sub on small tiles (samp_fast.py)
_REAX_HEAD_TILE = _reax_os.environ.get("REAX_SAMP_FAST", "0") == "1"


def _reax_head_mm(h, w):
    """h [M, K] @ w[N, K].T through the batch-invariant pointer kernel on 16 x 32 tiles (BLOCK_K 64 = the default's):
    the per-element K chain is the default config's, so the same bits (head_sub's own first-call checks still run)."""
    from vllm.model_executor.determinism import batch_invariant as _bi
    from vllm.triton_utils import triton as _triton
    M, K = h.shape
    N = w.shape[0]
    b = w.t()
    c = torch.empty((M, N), device=h.device, dtype=h.dtype)
    BM, BN = 16, 32
    sms = _bi.num_compute_units(h.device.index)
    grid = (min(sms, _triton.cdiv(M, BM) * _triton.cdiv(N, BN)),)
    _bi.matmul_kernel_persistent[grid](
        h, b, c, None, M, N, K, h.stride(0), h.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, NUM_SMS=sms,
        A_LARGE=h.numel() > 2**31, B_LARGE=b.numel() > 2**31, C_LARGE=c.numel() > 2**31, HAS_BIAS=False,
        num_stages=3, num_warps=4)
    return c
'''


def _patch(path, pairs, add):
    src = open(path).read()
    for old, _ in pairs:
        assert src.count(old) == 1, f"{path}: anchor not found once: {old!r}"
    for old, new in pairs:
        src = src.replace(old, new)
    open(path, "w").write(src + add)


def install(vllm_dir, overlay=None):
    _patch(f"{vllm_dir}/v1/worker/gpu/sample/logprob.py", [(LOGPROB_OLD, LOGPROB_NEW), (RANKS_OLD, RANKS_NEW)],
           LOGPROB_ADD)
    _patch(f"{vllm_dir}/v1/worker/gpu/sample/logit_bias.py", [(BIAS_OLD, BIAS_NEW)], BIAS_ADD)
    _patch(f"{vllm_dir}/v1/worker/gpu/sample/gumbel.py", [(GUMBEL_OLD, GUMBEL_NEW)], GUMBEL_ADD)
    if overlay:
        _patch(overlay, [(HEAD_OLD, HEAD_NEW)], HEAD_ADD)
    print("samp_fast: installed" + (" (+ head tile)" if overlay else ""))


def test():
    import os
    os.environ["REAX_SAMP_FAST"] = "1"
    os.environ["REAX_SAMP_FAST_CHECKS"] = "0"
    import importlib
    import torch
    import vllm.v1.worker.gpu.sample.logprob as lp
    import vllm.v1.worker.gpu.sample.logit_bias as lb
    lp, lb = importlib.reload(lp), importlib.reload(lb)
    assert hasattr(lp, "_reax_lsm_run") and hasattr(lb, "_reax_bias_run"), "samp_fast not installed in this vLLM"
    from vllm.model_executor.determinism import batch_invariant as bi
    from triton.testing import do_bench_cudagraph as do_bench
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(20261028)
    V = 262144
    codes = torch.randperm(V, generator=torch.Generator().manual_seed(7))[:255].sort().values.to(dev)

    def rows(B, kind):
        if kind == "dense":
            return torch.randn(B, V, device=dev, generator=g) * 4
        if kind == "wide":
            return torch.randn(B, V, device=dev, generator=g) * 30
        x = (torch.tanh(torch.randn(B, V, device=dev, generator=g) * 0.4) * 30).bfloat16().float()
        if kind == "soft":
            return x
        out = torch.full_like(x, float("-inf"))
        out[:, codes] = x[:, codes]
        return out

    bad = 0
    n = 0
    for kind in ("dense", "wide", "soft", "codes"):
        for B in (1, 2, 3, 5, 8, 13, 16, 32):
            for topk in (1, 21, 256):
                x = rows(B, kind)
                ids = torch.randint(0, V, (B, topk), device=dev, generator=g)
                ids[:, 0] = codes[torch.randint(0, 255, (B,), device=dev, generator=g)]
                tb = min(triton_next_pow2(topk), 1024)
                ref = torch.empty(B, topk, device=dev)
                new = torch.full((B, topk), 7.0, device=dev)
                lp._topk_log_softmax_kernel[(B,)](ref, x, x.stride(0), ids, topk, V, BLOCK_SIZE=1024, TOPK_BLOCK_SIZE=tb)
                lp._reax_lsm_run(new, x, x.stride(0), ids, topk, V, 1024, tb)
                n += 1
                if not torch.equal(ref.view(torch.int32), new.view(torch.int32)):
                    bad += 1
                    print(f"  logprob MISMATCH {kind} B {B} topk {topk}: max |diff| {(ref - new).abs().max().item()}")
            sampled = ids[:, 0].contiguous()
            r0 = torch.empty(B, dtype=torch.int64, device=dev)
            r1 = torch.full((B,), 99, dtype=torch.int64, device=dev)
            lp._ranks_kernel[(B,)](r0, x, x.stride(0), sampled, V, BLOCK_SIZE=8192)
            lp._reax_ranks_launch(r1, x, x.stride(0), sampled, V, 8192)
            n += 1
            if not torch.equal(r0, r1):
                bad += 1
                print(f"  ranks MISMATCH {kind} B {B}")
    print(f"logprob + ranks: {n - bad} of {n} cases bit-identical")

    nb = 0
    nt = 0
    for B in (1, 3, 8, 16, 32):
        for case in ("allowed", "allowed+bias", "bias", "none", "allowed+min"):
            x = rows(B, "soft")
            R = B + 2
            emap = torch.randperm(R, generator=torch.Generator().manual_seed(B))[:B].to(dev).int()
            na = torch.zeros(R, dtype=torch.int32, device=dev)
            al = torch.zeros(R, 256, dtype=torch.int32, device=dev)
            if "allowed" in case:
                na[:] = 255
                na[0] = 17
                al[:, :255] = codes.int()
            nlb = torch.zeros(R, dtype=torch.int32, device=dev)
            bt = torch.zeros(R, 256, dtype=torch.int32, device=dev)
            bv = torch.zeros(R, 256, dtype=torch.float32, device=dev)
            if "bias" in case:
                nlb[:] = 40
                bt[:, :40] = torch.randint(0, V, (R, 40), device=dev, generator=g).int()
                bt[:, :5] = codes[:5].int()
                bv[:, :40] = torch.randn(R, 40, device=dev, generator=g)
            pos = torch.zeros(B, dtype=torch.int64, device=dev)
            ml = torch.zeros(R, dtype=torch.int32, device=dev)
            nst = torch.zeros(R, dtype=torch.int32, device=dev)
            rwm = torch.zeros(R, dtype=torch.bool, device=dev)
            stt = torch.zeros(R, 256, dtype=torch.int32, device=dev)
            if "min" in case:
                ml[:] = 4
                nst[:] = 2
                stt[:, :2] = codes[3:5].int()
            for chk in (False, True):
                args = (x.stride(0), V, emap, na, al, al.stride(0), nlb, bt, bt.stride(0), bv, bv.stride(0), pos, ml, nst,
                        rwm, stt, stt.stride(0))
                kw = dict(BLOCK_SIZE=256, LOGITS_BLOCK_SIZE=8192, CHECK_ALL_MASKED_ROWS=chk)
                a, b = x.clone(), x.clone()
                lb._bias_kernel[(B,)](a, *args, **kw)
                lb._reax_bias_run(b, *args, **kw)
                nt += 1
                if not torch.equal(a.view(torch.int32), b.view(torch.int32)):
                    nb += 1
                    print(f"  bias MISMATCH B {B} {case} check_all {chk}")
    print(f"allowed mask / bias / min tokens: {nt - nb} of {nt} cases bit-identical")

    K = 2816
    w = (torch.randn(256, K, device=dev, generator=g) * 0.02).bfloat16()
    X = (torch.randn(300, K, device=dev, generator=g) * 3).bfloat16()
    hb = 0
    hn = 0
    ns = {}
    exec(HEAD_ADD.replace("_reax_os.environ", "os.environ"), {"torch": torch, "os": os}, ns)
    for M in list(range(1, 65)) + [77, 128, 255, 300]:
        bi._REAX_BYPASS[0] = True
        try:
            ref = bi.matmul_persistent(X[:M], w.t())
        finally:
            bi._REAX_BYPASS[0] = False
        hn += 1
        if not torch.equal(ref, ns["_reax_head_mm"](X[:M], w)):
            hb += 1
            print(f"  head MISMATCH M {M}")
    print(f"head_sub 16 x 32 tiles: {hn - hb} of {hn} M bit-identical")

    for B in (8, 16):
        x = rows(B, "codes")
        ids = torch.cat([codes[:1].expand(B, 1), codes[:21].expand(B, 21)], 1).contiguous()
        out = torch.empty(B, 22, device=dev)
        t0 = do_bench(lambda: lp._topk_log_softmax_kernel[(B,)](out, x, x.stride(0), ids, 22, V, BLOCK_SIZE=1024,
                                                                TOPK_BLOCK_SIZE=32), rep=200)
        t1 = do_bench(lambda: lp._reax_lsm_run(out, x, x.stride(0), ids, 22, V, 1024, 32), rep=200)
        s = ids[:, 0].contiguous()
        r = torch.empty(B, dtype=torch.int64, device=dev)
        t2 = do_bench(lambda: lp._ranks_kernel[(B,)](r, x, x.stride(0), s, V, BLOCK_SIZE=8192), rep=200)
        t3 = do_bench(lambda: lp._reax_ranks_launch(r, x, x.stride(0), s, V, 8192), rep=200)
        print(f"B {B}: logprob {t0 * 1e3:6.1f} -> {t1 * 1e3:5.1f} us | ranks {t2 * 1e3:5.1f} -> {t3 * 1e3:5.1f} us")
        R = B
        emap = torch.arange(B, device=dev, dtype=torch.int32)
        na = torch.full((R,), 255, dtype=torch.int32, device=dev)
        al = torch.zeros(R, 256, dtype=torch.int32, device=dev)
        al[:, :255] = codes.int()
        z = torch.zeros(R, dtype=torch.int32, device=dev)
        zt = torch.zeros(R, 256, dtype=torch.int32, device=dev)
        zf = torch.zeros(R, 256, dtype=torch.float32, device=dev)
        args = (x.stride(0), V, emap, na, al, al.stride(0), z, zt, zt.stride(0), zf, zf.stride(0),
                torch.zeros(B, dtype=torch.int64, device=dev), z, z, torch.zeros(R, dtype=torch.bool, device=dev), zt,
                zt.stride(0))
        kw = dict(BLOCK_SIZE=256, LOGITS_BLOCK_SIZE=8192, CHECK_ALL_MASKED_ROWS=False)
        t4 = do_bench(lambda: lb._bias_kernel[(B,)](x, *args, **kw), rep=200)
        t5 = do_bench(lambda: lb._reax_bias_run(x, *args, **kw), rep=200)
        h = X[:B].contiguous()
        bi._REAX_BYPASS[0] = True
        try:
            t6 = do_bench(lambda: bi.matmul_persistent(h, w.t()), rep=200)
        finally:
            bi._REAX_BYPASS[0] = False
        t7 = do_bench(lambda: ns["_reax_head_mm"](h, w), rep=200)
        print(f"B {B}: allowed mask {t4 * 1e3:5.1f} -> {t5 * 1e3:5.1f} us | head_sub mm {t6 * 1e3:5.1f} -> {t7 * 1e3:5.1f} us")
    import vllm.v1.worker.gpu.sample.gumbel as gm
    gm = importlib.reload(gm)
    from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
    gb = 0
    for B in (8, 16):
        x = rows(B, "codes")
        temp, seed, na_u = UvaBackedTensor(32, torch.float32), UvaBackedTensor(32, torch.int64), UvaBackedTensor(32, torch.int32)
        seed.np[:] = 12345
        na_u.np[:] = 255
        temp.copy_to_uva(), seed.copy_to_uva(), na_u.copy_to_uva()
        emap = torch.arange(B, device=dev, dtype=torch.int32)
        pos = torch.arange(B, device=dev, dtype=torch.int64)
        nbk = V // 1024
        outs = []
        for fn in (gm._gumbel_sample_kernel[(B, nbk)], gm._reax_gumbel_launch):
            la = torch.empty(B, nbk, dtype=torch.int64, device=dev)
            lm = torch.empty(B, nbk, dtype=torch.float32, device=dev)
            call = lambda: fn(la, la.stride(0), lm, lm.stride(0), None, 0, 0, None, x, x.stride(0), emap, seed.gpu, pos,
                              temp.gpu, V, BLOCK_SIZE=1024, IS_DRAFTING=False, APPLY_TEMPERATURE=False, USE_FP64=False,
                              PER_TOKEN_COL=False)
            call()
            outs.append((la.clone(), lm.clone(), do_bench(call, rep=200)))
        same = torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1].view(torch.int32), outs[1][1].view(torch.int32))
        gb += not same
        al = torch.zeros(32, 256, dtype=torch.int32, device=dev)
        al[:, :255] = codes.int()
        z = torch.zeros(32, dtype=torch.int32, device=dev)
        zt = torch.zeros(32, 256, dtype=torch.int32, device=dev)
        args = (x.stride(0), V, emap, na_u.gpu, al, al.stride(0), z, zt, zt.stride(0), zt.float(), 256,
                torch.zeros(B, dtype=torch.int64, device=dev), z, z, torch.zeros(32, dtype=torch.bool, device=dev), zt, 256)
        kw = dict(BLOCK_SIZE=256, LOGITS_BLOCK_SIZE=8192, CHECK_ALL_MASKED_ROWS=False)
        t8 = do_bench(lambda: lb._bias_kernel[(B,)](x, *args, **kw), rep=200)
        t9 = do_bench(lambda: lb._reax_bias_run(x, *args, **kw), rep=200)
        print(f"B {B} (UVA state): gumbel {outs[0][2] * 1e3:5.1f} -> {outs[1][2] * 1e3:5.1f} us ({'same' if same else 'DIFFER'})"
              f" | allowed mask {t8 * 1e3:5.1f} -> {t9 * 1e3:5.1f} us")
    print("PASS" if bad == 0 and nb == 0 and hb == 0 and gb == 0 else "FAIL")


def triton_next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


if __name__ == "__main__":
    if sys.argv[1] == "install":
        install(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif sys.argv[1] == "test":
        test()
