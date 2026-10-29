# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor

MAX_NUM_ALLOWED_TOKEN_IDS = 1024
MAX_NUM_LOGIT_BIAS_TOKENS = 1024
MAX_NUM_STOP_TOKEN_IDS = 128


class LogitBiasState:
    def __init__(self, max_num_reqs: int, device: torch.device):
        self.max_num_reqs = max_num_reqs

        # Allowed token IDs.
        self.num_allowed_token_ids = UvaBackedTensor(
            self.max_num_reqs, dtype=torch.int32
        )
        self.allowed_token_ids = StagedWriteTensor(
            (self.max_num_reqs, MAX_NUM_ALLOWED_TOKEN_IDS),
            dtype=torch.int32,
            device=device,
        )
        # Logit bias.
        self.num_logit_bias = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        self.logit_bias_token_ids = StagedWriteTensor(
            (self.max_num_reqs, MAX_NUM_LOGIT_BIAS_TOKENS),
            dtype=torch.int32,
            device=device,
        )
        self.logit_bias = StagedWriteTensor(
            (self.max_num_reqs, MAX_NUM_LOGIT_BIAS_TOKENS),
            dtype=torch.float32,
            device=device,
        )
        # Min tokens.
        self.min_lens = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        self.num_stop_token_ids = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        self.restore_when_all_masked = UvaBackedTensor(
            self.max_num_reqs, dtype=torch.int32
        )
        self.stop_token_ids = StagedWriteTensor(
            (self.max_num_reqs, MAX_NUM_STOP_TOKEN_IDS),
            dtype=torch.int32,
            device=device,
        )

        # Using any of the above.
        self.use_logit_bias = np.zeros(max_num_reqs, dtype=bool)

    def add_request(
        self, req_idx: int, prompt_len: int, sampling_params: SamplingParams
    ) -> None:
        # Using any logit bias.
        use_logit_bias = False

        # Allowed token IDs.
        allowed_token_ids = sampling_params.allowed_token_ids
        if allowed_token_ids:
            num_allowed_token_ids = len(allowed_token_ids)
            if num_allowed_token_ids > MAX_NUM_ALLOWED_TOKEN_IDS:
                raise ValueError(
                    f"Too many allowed token IDs: {num_allowed_token_ids}. "
                    f"The max size is {MAX_NUM_ALLOWED_TOKEN_IDS}."
                )
            self.num_allowed_token_ids.np[req_idx] = num_allowed_token_ids
            self.allowed_token_ids.stage_write(req_idx, 0, allowed_token_ids)
            use_logit_bias = True
        else:
            self.num_allowed_token_ids.np[req_idx] = 0

        # Logit bias.
        logit_bias = sampling_params.logit_bias
        if logit_bias:
            num_logit_bias = len(logit_bias)
            if num_logit_bias > MAX_NUM_LOGIT_BIAS_TOKENS:
                raise ValueError(
                    f"Too many logit bias tokens: {num_logit_bias}. "
                    f"The max size is {MAX_NUM_LOGIT_BIAS_TOKENS}."
                )
            self.num_logit_bias.np[req_idx] = num_logit_bias
            self.logit_bias_token_ids.stage_write(req_idx, 0, logit_bias.keys())
            self.logit_bias.stage_write(req_idx, 0, logit_bias.values())
            use_logit_bias = True
        else:
            self.num_logit_bias.np[req_idx] = 0

        # Min tokens.
        min_tokens = sampling_params.min_tokens
        min_len = prompt_len + min_tokens
        self.min_lens.np[req_idx] = min_len
        stop_token_ids = sampling_params.all_stop_token_ids
        if min_tokens > 0 and stop_token_ids:
            num_stop_token_ids = len(stop_token_ids)
            if num_stop_token_ids > MAX_NUM_STOP_TOKEN_IDS:
                raise ValueError(
                    f"Too many stop tokens: {num_stop_token_ids}. "
                    f"The max size is {MAX_NUM_STOP_TOKEN_IDS}."
                )
            self.num_stop_token_ids.np[req_idx] = num_stop_token_ids
            self.stop_token_ids.stage_write(req_idx, 0, stop_token_ids)
            self.restore_when_all_masked.np[req_idx] = int(
                sampling_params.structured_outputs is not None
            )
            use_logit_bias = True
        else:
            self.num_stop_token_ids.np[req_idx] = 0
            self.restore_when_all_masked.np[req_idx] = 0

        self.use_logit_bias[req_idx] = use_logit_bias

    def apply_staged_writes(self) -> None:
        self.num_allowed_token_ids.copy_to_uva()
        self.allowed_token_ids.apply_write()

        self.num_logit_bias.copy_to_uva()
        self.logit_bias_token_ids.apply_write()
        self.logit_bias.apply_write()

        self.min_lens.copy_to_uva()
        self.num_stop_token_ids.copy_to_uva()
        self.restore_when_all_masked.copy_to_uva()
        self.stop_token_ids.apply_write()

    def apply_logit_bias(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
    ) -> None:
        if not np.any(self.use_logit_bias[idx_mapping_np]):
            # No request uses logit bias. Skip the kernel launch.
            return

        enable_stop_token_restore = bool(
            np.any(self.restore_when_all_masked.np[idx_mapping_np])
        )

        apply_logit_bias(
            logits,
            expanded_idx_mapping,
            pos,
            self.num_allowed_token_ids.gpu,
            self.allowed_token_ids.gpu,
            self.num_logit_bias.gpu,
            self.logit_bias_token_ids.gpu,
            self.logit_bias.gpu,
            self.min_lens.gpu,
            self.num_stop_token_ids.gpu,
            self.restore_when_all_masked.gpu,
            self.stop_token_ids.gpu,
            enable_stop_token_restore,
        )


@triton.jit
def _bias_kernel(
    logits_ptr,
    logits_stride,
    vocab_size,
    expanded_idx_mapping_ptr,
    # Allowed token IDs.
    num_allowed_token_ids_ptr,
    allowed_token_ids_ptr,
    allowed_token_ids_stride,
    # Logit bias.
    num_logit_bias_ptr,
    bias_token_ids_ptr,
    bias_token_ids_stride,
    bias_ptr,
    bias_stride,
    # Min tokens.
    pos_ptr,
    min_lens_ptr,
    num_stop_token_ids_ptr,
    restore_when_all_masked_ptr,
    stop_token_ids_ptr,
    stop_token_ids_stride,
    BLOCK_SIZE: tl.constexpr,
    LOGITS_BLOCK_SIZE: tl.constexpr,
    CHECK_ALL_MASKED_ROWS: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)

    block = tl.arange(0, BLOCK_SIZE)

    # Allowed token IDs.
    num_allowed_token_ids = tl.load(num_allowed_token_ids_ptr + req_state_idx)
    if num_allowed_token_ids > 0:
        block = tl.arange(0, BLOCK_SIZE)
        mask = block < num_allowed_token_ids

        # Save logits for allowed token IDs.
        allowed_token_ids = tl.load(
            allowed_token_ids_ptr + req_state_idx * allowed_token_ids_stride + block,
            mask=mask,
        )
        logits = tl.load(
            logits_ptr + token_idx * logits_stride + allowed_token_ids, mask=mask
        )

        tl.debug_barrier()  # save must read original logits before the -inf overwrite

        # Set logits to -inf for all tokens.
        for i in range(0, vocab_size, LOGITS_BLOCK_SIZE):
            offset = i + tl.arange(0, LOGITS_BLOCK_SIZE)
            tl.store(
                logits_ptr + token_idx * logits_stride + offset,
                -float("inf"),
                mask=offset < vocab_size,
            )

        tl.debug_barrier()  # -inf overwrite must finish before restoring saved logits

        # Restore logits for allowed token IDs.
        tl.store(
            logits_ptr + token_idx * logits_stride + allowed_token_ids,
            logits,
            mask=mask,
        )

    # Logit bias.
    num_logit_bias = tl.load(num_logit_bias_ptr + req_state_idx)
    if num_logit_bias > 0:
        mask = block < num_logit_bias
        token_ids = tl.load(
            bias_token_ids_ptr + req_state_idx * bias_token_ids_stride + block,
            mask=mask,
        )
        bias = tl.load(bias_ptr + req_state_idx * bias_stride + block, mask=mask)
        logits = tl.load(logits_ptr + token_idx * logits_stride + token_ids, mask=mask)
        logits += bias
        tl.store(logits_ptr + token_idx * logits_stride + token_ids, logits, mask=mask)

    # Apply min tokens.
    num_stop_token_ids = tl.load(num_stop_token_ids_ptr + req_state_idx)
    pos = tl.load(pos_ptr + token_idx)
    min_len = tl.load(min_lens_ptr + req_state_idx)
    if num_stop_token_ids > 0 and pos + 1 < min_len:
        mask = block < num_stop_token_ids
        stop_token_ids = tl.load(
            stop_token_ids_ptr + req_state_idx * stop_token_ids_stride + block,
            mask=mask,
        )
        if CHECK_ALL_MASKED_ROWS:
            should_restore_stop_logits = tl.load(
                restore_when_all_masked_ptr + req_state_idx
            )
            if should_restore_stop_logits:
                stop_logits = tl.load(
                    logits_ptr + token_idx * logits_stride + stop_token_ids,
                    mask=mask,
                    other=-float("inf"),
                )

                # Save must read original logits before the -inf overwrite.
                tl.debug_barrier()

                tl.store(
                    logits_ptr + token_idx * logits_stride + stop_token_ids,
                    -float("inf"),
                    mask=mask,
                )

                # The -inf overwrite must be visible to the row scan below.
                tl.debug_barrier()

                row_max = tl.full((), -float("inf"), tl.float32)
                for i in range(0, vocab_size, LOGITS_BLOCK_SIZE):
                    offset = i + tl.arange(0, LOGITS_BLOCK_SIZE)
                    logits = tl.load(
                        logits_ptr + token_idx * logits_stride + offset,
                        mask=offset < vocab_size,
                        other=-float("inf"),
                    )
                    row_max = tl.maximum(row_max, tl.max(logits, axis=0))

                if row_max == -float("inf"):
                    tl.store(
                        logits_ptr + token_idx * logits_stride + stop_token_ids,
                        stop_logits,
                        mask=mask
                        & (stop_logits > -float("inf"))
                        & (stop_logits < float("inf")),
                    )
            else:
                tl.store(
                    logits_ptr + token_idx * logits_stride + stop_token_ids,
                    -float("inf"),
                    mask=mask,
                )
        else:
            tl.store(
                logits_ptr + token_idx * logits_stride + stop_token_ids,
                -float("inf"),
                mask=mask,
            )


def apply_logit_bias(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    pos: torch.Tensor,
    num_allowed_token_ids: torch.Tensor,
    allowed_token_ids: torch.Tensor,
    num_logit_bias: torch.Tensor,
    logit_bias_token_ids: torch.Tensor,
    logit_bias: torch.Tensor,
    min_lens: torch.Tensor,
    num_stop_token_ids: torch.Tensor,
    restore_when_all_masked: torch.Tensor,
    stop_token_ids: torch.Tensor,
    check_all_masked_rows: bool = False,
) -> None:
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = triton.next_power_of_2(
        max(
            allowed_token_ids.shape[-1],
            logit_bias_token_ids.shape[-1],
            stop_token_ids.shape[-1],
        )
    )
    LOGITS_BLOCK_SIZE = 8192
    (_reax_bias_launch if _REAX_SAMP else _bias_kernel[(num_tokens,)])(
        logits,
        logits.stride(0),
        vocab_size,
        expanded_idx_mapping,
        num_allowed_token_ids,
        allowed_token_ids,
        allowed_token_ids.stride(0),
        num_logit_bias,
        logit_bias_token_ids,
        logit_bias_token_ids.stride(0),
        logit_bias,
        logit_bias.stride(0),
        pos,
        min_lens,
        num_stop_token_ids,
        restore_when_all_masked,
        stop_token_ids,
        stop_token_ids.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        LOGITS_BLOCK_SIZE=LOGITS_BLOCK_SIZE,
        CHECK_ALL_MASKED_ROWS=check_all_masked_rows,
    )


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
