# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.sampling_params import MAX_LOGPROB_TOKEN_IDS, SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor

# Upper bound on the topk kernel's per-iteration gather width.
_MAX_TOPK_BLOCK = 1024


@triton.jit
def _topk_log_softmax_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    topk_ids_ptr,
    topk,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    TOPK_BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + req_idx * logits_stride

    max_val = float("-inf")
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        max_val = tl.max(tl.maximum(logits, max_val))
    max_val = max_val.to(tl.float32)  # type: ignore

    se = 0.0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=0.0)
        # NOTE(woosuk): Make sure that logits and all following operations use FP32.
        logits = logits.to(tl.float32)
        e = tl.exp(logits - max_val)
        e = tl.where(block < vocab_size, e, 0.0)
        se += tl.sum(e)
    lse = tl.log(se)

    for j in range(0, topk, TOPK_BLOCK_SIZE):
        k_offset = j + tl.arange(0, TOPK_BLOCK_SIZE)
        k_mask = k_offset < topk
        topk_ids = tl.load(
            topk_ids_ptr + req_idx * topk + k_offset, mask=k_mask, other=0
        )
        logits = tl.load(row_ptr + topk_ids, mask=k_mask)
        logits = logits.to(tl.float32)
        o = logits - max_val - lse
        tl.store(output_ptr + req_idx * topk + k_offset, o, mask=k_mask)


@triton.jit
def _ranks_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    token_ids_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + req_idx * logits_stride

    token_id = tl.load(token_ids_ptr + req_idx)
    x = tl.load(row_ptr + token_id)

    n = 0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        n += tl.sum((logits >= x).to(tl.int32))
    tl.store(output_ptr + req_idx, n)


def compute_token_logprobs(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    # NOTE(woosuk): To save GPU memory, we do not materialize the full
    # [batch_size, vocab_size] logprobs tensor. The kernel computes
    # max + logsumexp per row and only emits logprobs at `token_ids`.
    batch_size, vocab_size = logits.shape
    token_ids = token_ids.to(torch.int64)
    num_logprobs = token_ids.shape[1]
    logprobs = logits.new_empty((batch_size, num_logprobs), dtype=torch.float32)
    # Cap the kernel's per-iteration width so very large num_logprobs requests
    # stream the gather in bounded-size chunks, avoiding excessive mem use.
    topk_block_size = min(triton.next_power_of_2(num_logprobs), _MAX_TOPK_BLOCK)
    (_reax_lsm_launch if _REAX_SAMP else _topk_log_softmax_kernel[(batch_size,)])(
        logprobs,
        logits,
        logits.stride(0),
        token_ids,
        num_logprobs,
        vocab_size,
        BLOCK_SIZE=1024,  # type: ignore
        TOPK_BLOCK_SIZE=topk_block_size,
    )
    return logprobs


def compute_topk_scores(
    logits: torch.Tensor,
    num_logprobs: int,
    sampled_token_ids: torch.Tensor,
    cu_num_logits: list[int] | torch.Tensor | None = None,
    logprob_token_ids_state: "LogprobTokenIdsState | None" = None,
    expanded_idx_mapping: torch.Tensor | None = None,
    max_per_req_token_ids: int = 0,
    logits_mode: bool = False,
) -> LogprobsTensors:
    assert num_logprobs >= 0
    batch_size, vocab_size = logits.shape

    if max_per_req_token_ids == 0:
        # Fast path: no request asked for custom logprob_token_ids.
        logprob_token_ids = sampled_token_ids.unsqueeze(-1)
        if num_logprobs > 0:
            topk_indices = torch.topk(logits, num_logprobs, dim=-1).indices
            logprob_token_ids = torch.cat((logprob_token_ids, topk_indices), dim=1)
        if logits_mode:
            scores = logits.gather(-1, logprob_token_ids).to(torch.float32)
        else:
            scores = compute_token_logprobs(logits, logprob_token_ids)
    else:
        # Some requests specified logprob_token_ids. Build the [batch_size,
        # 1 + max_cols] token_ids matrix and validity mask on the GPU via a
        # single triton kernel, overriding the topk columns with per-request
        # tokens where applicable.
        assert logprob_token_ids_state is not None
        assert expanded_idx_mapping is not None

        if num_logprobs > 0:
            topk_token_ids = torch.topk(logits, num_logprobs, dim=-1).indices
            topk_token_ids = topk_token_ids.to(torch.int32)
        else:
            # This tensor just used as an int32 pointer, data not accessed.
            topk_token_ids = logprob_token_ids_state.token_ids.gpu

        num_cols = max(num_logprobs, max_per_req_token_ids)
        logprob_token_ids = sampled_token_ids.new_zeros((batch_size, 1 + num_cols))
        valid_mask = torch.zeros_like(logprob_token_ids, dtype=torch.bool)
        _fill_logprob_token_ids_kernel[(batch_size,)](
            logprob_token_ids,
            logprob_token_ids.stride(0),
            valid_mask,
            valid_mask.stride(0),
            sampled_token_ids,
            topk_token_ids,
            topk_token_ids.stride(0),
            expanded_idx_mapping,
            logprob_token_ids_state.num_token_ids.gpu,
            logprob_token_ids_state.token_ids.gpu,
            logprob_token_ids_state.token_ids.gpu.stride(0),
            NUM_TOPK=num_logprobs,
            PADDED_COLS=triton.next_power_of_2(num_cols),
        )
        if logits_mode:
            scores = logits.gather(-1, logprob_token_ids).to(torch.float32)
        else:
            scores = compute_token_logprobs(logits, logprob_token_ids)
        scores = scores.masked_fill(~valid_mask, float("-inf"))

    token_ranks = torch.empty(batch_size, dtype=torch.int64, device=logits.device)
    (_reax_ranks_launch if _REAX_SAMP else _ranks_kernel[(batch_size,)])(
        token_ranks,
        logits,
        logits.stride(0),
        sampled_token_ids,
        vocab_size,
        BLOCK_SIZE=8192,  # type: ignore
    )
    is_tensor = isinstance(cu_num_logits, torch.Tensor)
    return LogprobsTensors(
        logprob_token_ids=logprob_token_ids,
        logprobs=scores,
        selected_token_ranks=token_ranks,
        cu_num_generated_tokens=None if is_tensor else cu_num_logits,
        cu_num_generated_tokens_tensor=cu_num_logits if is_tensor else None,
    )


@triton.jit
def _fill_logprob_token_ids_kernel(
    # [batch_size, 1 + num_cols]
    out_token_ids_ptr,
    out_token_ids_stride,
    # [batch_size, 1 + num_cols]
    out_valid_mask_ptr,
    out_valid_mask_stride,
    sampled_token_ids_ptr,  # [batch_size]
    topk_indices_ptr,  # [batch_size, NUM_TOPK] (unused when NUM_TOPK == 0)
    topk_indices_stride,
    expanded_idx_mapping_ptr,  # [batch_size] -> req_state_idx
    num_per_req_token_ids_ptr,  # [max_num_reqs]
    per_req_token_ids_ptr,  # [max_num_reqs, MAX_LOGPROB_TOKEN_IDS]
    per_req_token_ids_stride,
    NUM_TOPK: tl.constexpr,
    PADDED_COLS: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    # Column 0: always the sampled token, always valid.
    sampled = tl.load(sampled_token_ids_ptr + batch_idx)
    tl.store(out_token_ids_ptr + batch_idx * out_token_ids_stride, sampled)
    tl.store(out_valid_mask_ptr + batch_idx * out_valid_mask_stride, 1)

    req_state_idx = tl.load(expanded_idx_mapping_ptr + batch_idx)
    num_custom = tl.load(num_per_req_token_ids_ptr + req_state_idx)

    col = tl.arange(0, PADDED_COLS)
    tid_base = out_token_ids_ptr + batch_idx * out_token_ids_stride + 1
    mask_base = out_valid_mask_ptr + batch_idx * out_valid_mask_stride + 1

    if num_custom > 0:
        # Override topk with per-request custom tokens.
        src = per_req_token_ids_ptr + req_state_idx * per_req_token_ids_stride
        valid = col < num_custom
    else:
        # Fill with topk indices (no-op when NUM_TOPK == 0).
        src = topk_indices_ptr + batch_idx * topk_indices_stride
        valid = col < NUM_TOPK

    tokens = tl.load(src + col, mask=valid, other=0).to(tl.int64)
    tl.store(tid_base + col, tokens, mask=valid)
    tl.store(mask_base + col, tl.full([PADDED_COLS], 1, tl.int1), mask=valid)


class LogprobTokenIdsState:
    """Per-request override of which token ids' logprobs to return.

    See `SamplingParams.logprob_token_ids`.
    """

    def __init__(self, max_num_reqs: int, device: torch.device):
        self.max_num_reqs = max_num_reqs
        self.num_token_ids = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.token_ids = StagedWriteTensor(
            (max_num_reqs, MAX_LOGPROB_TOKEN_IDS),
            dtype=torch.int32,
            device=device,
        )

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        token_ids = sampling_params.logprob_token_ids
        if not token_ids:
            self.num_token_ids.np[req_idx] = 0
            return
        n = len(token_ids)
        if n > MAX_LOGPROB_TOKEN_IDS:
            raise ValueError(
                f"Too many logprob_token_ids: {n}. The max is {MAX_LOGPROB_TOKEN_IDS}."
            )
        self.num_token_ids.np[req_idx] = n
        self.token_ids.stage_write(req_idx, 0, token_ids)

    def apply_staged_writes(self) -> None:
        self.num_token_ids.copy_to_uva()
        self.token_ids.apply_write()

    def max_num_token_ids(self, idx_mapping_np: np.ndarray) -> int:
        return int(self.num_token_ids.np[idx_mapping_np].max(initial=0))


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
