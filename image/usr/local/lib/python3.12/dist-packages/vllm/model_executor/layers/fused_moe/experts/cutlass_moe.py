# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUTLASS based Fused MoE kernels."""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import (
    ApplyMoEActivationConfig,
    MoEActivation,
    apply_moe_activation,
    apply_moe_activation_supported,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    MoEPermuteScratch,
    moe_permute,
    moe_permute_unpermute_supported,
    moe_unpermute,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kInt4Static,
    kMxfp4Dynamic,
    kMxfp4Static,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    cutlass_group_gemm_supported,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)


def run_cutlass_moe_fp8(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    a1q_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    ab_strides1: torch.Tensor,
    ab_strides2: torch.Tensor,
    c_strides1: torch.Tensor,
    c_strides2: torch.Tensor,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    expert_num_tokens: torch.Tensor | None,
    out_dtype: torch.dtype,
    per_act_token: bool,
    per_out_ch: bool,
    use_batched_format: bool,
    topk_weights: torch.Tensor | None,
    permute_scratch: MoEPermuteScratch | None,
    *,
    activation_config: ApplyMoEActivationConfig | None = None,
):
    a1q = hidden_states

    assert activation.is_gated, "Only gated activation is supported"
    assert w1_scale is not None
    assert w2_scale is not None
    assert w1.dtype == torch.float8_e4m3fn
    assert w2.dtype == torch.float8_e4m3fn
    assert a1q.size(-1) == w1.size(2), "Hidden size mismatch w1"
    assert w1.size(1) == w2.size(2) * 2, "Hidden size mismatch w2"
    assert (
        w1_scale.dim() == 1 or w1_scale.size(1) == 1 or w1_scale.shape[1] == w1.size(1)
    ), "W1 scale shape mismatch"
    assert (
        w2_scale.dim() == 1 or w2_scale.size(1) == 1 or w2_scale.shape[1] == w2.size(1)
    ), "W2 scale shape mismatch"
    assert w1.size(0) == w2.size(0), "Expert number mismatch"
    assert (
        a1q_scale is None
        or a1q_scale.dim() == 0
        or a1q_scale.size(0) == 1
        or a1q_scale.size(0) == a1q.shape[0]
    ), "Input scale shape mismatch"
    assert w1.size(0) == w2.size(0), "Weights expert number mismatch"
    assert w1.size(0) == w1_scale.size(0), "w1 scales expert number mismatch"
    assert w1.size(0) == w2_scale.size(0), "w2 scales expert number mismatch"
    assert (
        a2_scale is None
        or a2_scale.dim() == 0
        or a2_scale.size(0) == 1
        or a2_scale.size(0) == a1q.shape[0]
    ), "Intermediate scale shape mismatch"
    assert out_dtype in [torch.half, torch.bfloat16], "Invalid output dtype"

    # NOTE(rob): the expert_map is used for the STANDARD case and
    # the batched format is used by the BATCHED case.
    # TODO(rob): update the MK interface to only pass the expert_map
    # during the STANDARD case to make this clearer across all kernels.
    if use_batched_format:
        assert expert_num_tokens is not None
    else:
        assert expert_num_tokens is None

    # We have two modes: batched experts and non-batched experts.
    # In the non-batched mode, the input tokens are not padded: thus, the shape
    # of the input is [total_num_tokens, hidden_size]. The input and output
    # require shuffling by a_map and c_map such that the tokens assigned to
    # each expert are contiguous.
    # In the batched mode, the input tokens are padded per expert to ensure that
    # the batched dispatch and combine functions work correctly: thus, the shape
    # of the input is [num_experts, max_num_tokens_per_expert, hidden_size].
    # The batched input and output require no shuffling by a_map and c_map since
    # their tokens are already contiguous for each expert as a result of
    # the dispatch function.

    M = a1q.size(0)  # non batched expert M
    padded_M = a1q.size(1)  # batched expert M
    _, K, N = w2.shape
    device = a1q.device

    assert w1.size(2) == K
    assert global_num_experts != -1
    assert a1q_scale is not None

    topk = topk_ids.size(1)
    local_E = w1.size(0)

    if use_batched_format:
        mm1_out = _resize_cache(workspace13, (local_E * padded_M, N * 2))
        act_out = _resize_cache(workspace2, (local_E * padded_M, N))
        quant_out = _resize_cache(
            workspace13.view(dtype=torch.float8_e4m3fn), (local_E * padded_M, N)
        )
        mm2_out = _resize_cache(workspace2, (local_E * padded_M, K))
    else:
        a1q_perm = _resize_cache(
            workspace2.view(dtype=torch.float8_e4m3fn), (M * topk, K)
        )
        mm1_out = _resize_cache(workspace13, (M * topk, N * 2))
        act_out = _resize_cache(workspace2, (M * topk, N))
        # original workspace are based on input hidden_states dtype (bf16)
        quant_out = _resize_cache(
            workspace13.view(dtype=torch.float8_e4m3fn), (M * topk, N)
        )
        mm2_out = _resize_cache(workspace2, (M * topk, K))

    if use_batched_format:
        assert expert_num_tokens is not None

        expert_offsets = torch.empty((local_E), dtype=torch.int32, device=device)
        problem_sizes1 = torch.empty((local_E, 3), dtype=torch.int32, device=device)
        problem_sizes2 = torch.empty((local_E, 3), dtype=torch.int32, device=device)

        ops.get_cutlass_batched_moe_mm_data(
            expert_offsets,
            problem_sizes1,
            problem_sizes2,
            expert_num_tokens,
            local_E,
            padded_M,
            N,
            K,
        )

        w1_scale = w1_scale.reshape(w1_scale.size(0), -1)
        w2_scale = w2_scale.reshape(w2_scale.size(0), -1)
        a1q = a1q.reshape(-1, a1q.size(2))
        a1q_scale = a1q_scale.reshape(-1, a1q_scale.size(2)).contiguous()
        # c3x get_group_gemm_starts expects int64 to avoid overflow
        # during offset calculations
        expert_offsets = expert_offsets.to(torch.int64)
    else:
        problem_sizes1 = torch.empty((local_E, 3), dtype=torch.int32, device=device)
        problem_sizes2 = torch.empty((local_E, 3), dtype=torch.int32, device=device)

        num_expert = global_num_experts if expert_map is None else expert_map.size(0)
        # permuted a1q reuses workspace2
        a1q, a1q_scale, expert_first_token_offset, inv_perm, _ = moe_permute(
            a1q,
            a1q_scale,
            topk_ids,
            num_expert,
            local_E,
            expert_map,
            permuted_hidden_states=a1q_perm,
            scratch=permute_scratch,
        )
        # swap_ab is a CUTLASS grouped-GEMM optimization (M <= 64 reduces padding).
        swap_ab = a1q.size(0) <= 64
        ops.get_cutlass_moe_mm_problem_sizes_from_expert_offsets(
            expert_first_token_offset, problem_sizes1, problem_sizes2, N, K, swap_ab
        )
        expert_offsets = expert_first_token_offset[:-1]

    if not per_act_token and (expert_map is not None or use_batched_format):
        # this is necessary to avoid imprecise scale calculation caused by
        # random data in the unused workspace. The workspace is unused when
        # this rank handles only partial tokens, or when it is batched .
        mm1_out.fill_(0)

    ops.cutlass_moe_mm(
        mm1_out,
        a1q,
        w1,
        a1q_scale,
        w1_scale,
        expert_offsets,
        problem_sizes1,
        ab_strides1,
        ab_strides1,
        c_strides1,
        per_act_token,
        per_out_ch,
    )

    apply_moe_activation(
        activation,
        act_out,
        mm1_out,
        activation_config=activation_config,
    )

    a2q, a2q_scale = ops.scaled_fp8_quant(
        act_out, a2_scale, use_per_token_if_dynamic=per_act_token, output=quant_out
    )

    ops.cutlass_moe_mm(
        mm2_out,
        a2q,
        w2,
        a2q_scale,
        w2_scale,
        expert_offsets,
        problem_sizes2,
        ab_strides2,
        ab_strides2,
        c_strides2,
        per_act_token,
        per_out_ch,
    )

    if use_batched_format:
        output.copy_(mm2_out.reshape(local_E, padded_M, K), non_blocking=True)
    else:
        # for non-chunking mode the output is resized from workspace13
        # so we need to make sure mm2_out uses workspace2.
        moe_unpermute(
            out=output,
            permuted_hidden_states=mm2_out,
            topk_weights=topk_weights,
            inv_permuted_idx=inv_perm,
            expert_first_token_offset=expert_first_token_offset,
        )


class CutlassExpertsFp8Base(mk.FusedMoEExpertsModular):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
        assert quant_config.use_fp8_w8a8

        e = moe_config.num_local_experts
        n = moe_config.intermediate_size_per_partition
        k = moe_config.hidden_dim
        device = moe_config.device
        ab_strides1_c_strides2 = torch.full((e,), k, device=device, dtype=torch.int64)
        ab_strides2 = torch.full((e,), n, device=device, dtype=torch.int64)
        c_strides1 = torch.full((e,), 2 * n, device=device, dtype=torch.int64)

        self.out_dtype = moe_config.in_dtype
        self.ab_strides1 = ab_strides1_c_strides2
        self.ab_strides2 = ab_strides2
        self.c_strides1 = c_strides1
        self.c_strides2 = ab_strides1_c_strides2
        self._permute_scratch: MoEPermuteScratch | None = None

    @staticmethod
    def _supports_current_device() -> bool:
        return cutlass_group_gemm_supported()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kFp8StaticChannelSym, kFp8DynamicTokenSym),
            (kFp8StaticTensorSym, kFp8DynamicTensorSym),
            (kFp8StaticTensorSym, kFp8StaticTensorSym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation.is_gated and apply_moe_activation_supported(activation)

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Let PrepareAndFinalize::finalize() decide the impl.
        return TopKWeightAndReduceDelegate()

    def _get_permute_scratch(self) -> MoEPermuteScratch | None:
        if self._permute_scratch is None and moe_permute_unpermute_supported():
            self._permute_scratch = MoEPermuteScratch(
                max_num_tokens=self.moe_config.max_num_tokens,
                topk=self.moe_config.experts_per_token,
                num_experts=self.moe_config.num_experts,
                num_local_experts=self.moe_config.num_local_experts,
                device=torch.device(self.moe_config.device),
            )
        return self._permute_scratch

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert self.w1_zp is None, "w1_zp is not supported in CUTLASS MoE"
        assert self.w2_zp is None, "w2_zp is not supported in CUTLASS MoE"

        expert_num_tokens = None
        if expert_tokens_meta is not None:
            expert_num_tokens = expert_tokens_meta.expert_num_tokens

        use_batched_format = (
            self.activation_format() == mk.FusedMoEActivationFormat.BatchedExperts
        )

        in_dtype = hidden_states.dtype
        run_cutlass_moe_fp8(
            output,
            hidden_states,
            w1,
            w2,
            topk_ids,
            activation,
            global_num_experts,
            expert_map,
            self.w1_scale,
            self.w2_scale,
            a1q_scale,
            a2_scale,
            self.ab_strides1,
            self.ab_strides2,
            self.c_strides1,
            self.c_strides2,
            workspace13,
            workspace2,
            expert_num_tokens,
            self.out_dtype if self.out_dtype is not None else in_dtype,
            self.per_act_token_quant,
            self.per_out_ch_quant,
            use_batched_format,
            topk_weights,
            self._get_permute_scratch(),
            activation_config=self.activation_config,
        )


class CutlassExpertsFp8(CutlassExpertsFp8Base):
    """CUTLASS FP8 fused MoE expert implementation."""

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # CutlassExpertsFp8 does not support expert map, which is
        # needed for STANDARD activation format kernels in DP/EP mode.
        # Note that the BATCHED activation format does not use
        # the expert map for identifying experts.
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_deepep_ht_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # topk weights and reduction are fused in moe_unpermute cuda kernel
        return TopKWeightAndReduceNoOP()

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        return self.out_dtype if self.out_dtype is not None else act_dtype

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M * topk, max(N, K))
        workspace2 = (M * topk, max(activation_out_dim, K))
        output = (M, K)
        return (workspace1, workspace2, output)


class CutlassBatchedExpertsFp8(CutlassExpertsFp8Base):
    """Batched CUTLASS FP8 fused MoE expert implementation."""

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # BATCHED activation format works with EP because
        # expert_map is not used to identify experts (the
        # info is encoded/managed by the P/F logic).
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        return self.out_dtype if self.out_dtype is not None else act_dtype

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        experts_per_worker = self.moe_config.num_local_experts
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (experts_per_worker, M, max(N, K))
        workspace2 = (
            experts_per_worker,
            M,
            max(activation_out_dim, K),
        )
        output = (experts_per_worker, M, K)
        return (workspace1, workspace2, output)


FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def run_cutlass_moe_fp4(
    output: torch.Tensor,
    a: torch.Tensor,
    a1_gscale: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    a2_gscale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    m: int,
    n: int,
    k: int,
    e: int,
    device: torch.device,
    apply_router_weight_on_input: bool = False,
    *,
    activation_config: ApplyMoEActivationConfig | None = None,
) -> None:
    """
    MoE implementation for FP4 Inputs

    # Gemm 1
    a: Input tensor: [m, k] (half/bfloat16)
    a1_gscale: Activation scale per expert: [e]  (float32)
    w1 (not an argument to cutlass_moe_fp4): [e, w1_n, k]
    w1_fp4: [e, w1_n, k // 2], dtype: torch.uint8 (stacked fp4: E2M1)
    where w1_n = 2*n for gated activations (gate+up), n for non-gated (up only).
    (Note: `n` is the up projection output dim, `k` is the input dim in
     full precision)
    w1_blockscale: [e, w1_n, k // block_size] (float8_e4m3)
                   (Block size = 16 for NVFP4)

    # Gemm 2
    a2_gscale: Activation scale per expert: [e]
    w2(down projection) (not an argument to cutlass_moe_fp4): [e, k, n]
    w2_fp4: [e, k, n // 2], dtype: torch.uint8 (stacked E2M1)
    w2_blockscale: [e, k, n // block_size], dtype: float8_e4m3

    topk_weights: [m, topk] dtype: float8
    topk_ids: [m, topk] dtype: float8

    m, n, k: Unquantized weight shapes, dtype: int
    e: number of experts, dtype: int

    assumes that topk < k < n to satisfy - up/down projection expectations.
    """
    is_gated = activation.is_gated
    # For gated activations (e.g. SiLU), w1 output is 2*n (gate + up).
    # For non-gated activations (e.g. SiLU_NO_MUL), w1 output is n (up only).
    w1_n = n * 2 if is_gated else n

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1_fp4.dtype == torch.uint8, "weight 1 must be uint8"
    assert w2_fp4.dtype == torch.uint8, "weight 2 must be uint8"
    assert (
        w1_fp4.ndim == 3
        and w2_fp4.ndim == 3
        and w1_blockscale.ndim == 3
        and w2_blockscale.ndim == 3
    ), "All Weights must be of rank 3 for cutlass_moe_fp4"
    m_a, k_a = a.shape
    e_w1, w1_n_actual, half_k_w1 = w1_fp4.shape
    e_w2, k_w2, half_n_w2 = w2_fp4.shape

    assert e_w1 == e_w2 and e_w1 == e, (
        "Number of experts must match",
        f" between weights. {e_w1}, {e_w2}, {e}",
    )
    assert k_a == half_k_w1 * 2 and k == k_w2, (
        "Hidden size mismatch between a, w1 and w2"
    )
    assert w1_n_actual == w1_n and half_n_w2 * 2 == n, "mismatch in expected `n`"
    assert m == m_a, "input shape mismatch"
    assert 2 * half_k_w1 == k_w2, "Hidden size mismatch w2 and w1"
    assert a.dtype in [torch.half, torch.bfloat16], "Invalid input dtype"
    assert topk_weights.size(0) == m and topk_ids.size(0) == m, (
        "topk must be provided for each row of a"
    )
    topk = topk_ids.size(1)
    out_dtype = a.dtype
    num_topk = topk_ids.size(1)

    expert_offsets = torch.empty((e + 1), dtype=torch.int32, device=device)
    blockscale_offsets = torch.empty((e + 1), dtype=torch.int32, device=device)
    # Problem size:  (num_experts, (m,2n,k))
    problem_sizes1 = torch.empty((e, 3), dtype=torch.int32, device=device)
    # Problem size:  (num_experts, (m,n,k))
    problem_sizes2 = torch.empty((e, 3), dtype=torch.int32, device=device)

    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)

    if apply_router_weight_on_input:
        # TODO: this only works for topK=1, will need to update for topK>1
        assert num_topk == 1, (
            "apply_router_weight_on_input is only implemented for topk=1"
        )
        a.mul_(topk_weights.to(out_dtype))

    # problem shapes should have [m, n, k]
    # Note that problem sizes are based on logical number of elements.
    ops.get_cutlass_moe_mm_data(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        e,
        n,
        k,
        blockscale_offsets,
        is_gated=is_gated,
    )

    _gq = _reax_gquant(a, a_map, a1_gscale, expert_offsets, blockscale_offsets, num_topk)
    if _gq is not None:
        rep_a_fp4, rep_a_blockscale = _gq
    else:
        a = ops.shuffle_rows(a, a_map)
        rep_a_fp4, rep_a_blockscale = ops.scaled_fp4_experts_quant(
            a,
            a1_gscale,
            expert_offsets,
            blockscale_offsets,
            num_topk,
        )
    c1 = _resize_cache(workspace13, (m * topk, w1_n))
    c2 = _resize_cache(workspace2, (m * topk, n))
    c3 = _resize_cache(workspace13, (m * topk, k))
    ops.cutlass_fp4_moe_mm(
        c1,
        rep_a_fp4,
        w1_fp4,
        rep_a_blockscale,
        w1_blockscale,
        w1_alphas,
        problem_sizes1,
        expert_offsets[:-1],
        blockscale_offsets[:-1],
    )
    del rep_a_fp4, rep_a_blockscale
    if activation == MoEActivation.SILU and (
        activation_config is None or activation_config.clamp_limit is None
    ):
        # Fused SiLU+Mul+NVFP4 quantization
        # Note: c2 workspace is no longer needed since SiLU is fused with quantization.
        # c3 reuses workspace13 after c1 is consumed.
        int_fp4, int_blockscale = ops.silu_and_mul_scaled_fp4_experts_quant(
            c1, a2_gscale, expert_offsets, blockscale_offsets, num_topk
        )
    else:
        _ga = _reax_gact(activation, activation_config, c1, a2_gscale, expert_offsets, blockscale_offsets, num_topk)
        if _ga is not None:
            int_fp4, int_blockscale = _ga
        else:
            apply_moe_activation(
                activation,
                c2,
                c1,
                activation_config=activation_config,
            )
            int_fp4, int_blockscale = ops.scaled_fp4_experts_quant(
                c2, a2_gscale, expert_offsets, blockscale_offsets, num_topk
            )

    _reax_moe_dn(
        c3,
        int_fp4,
        w2_fp4,
        int_blockscale,
        w2_blockscale,
        w2_alphas,
        problem_sizes2,
        expert_offsets[:-1],
        blockscale_offsets[:-1],
    )
    del int_fp4, int_blockscale

    if not apply_router_weight_on_input and _reax_unperm(output, c3, c_map, topk_weights, m, num_topk, k):
        return
    c3 = ops.shuffle_rows(c3, c_map)

    assert output.dtype == out_dtype
    if not apply_router_weight_on_input:
        output.copy_(
            (
                c3.view(m, num_topk, k)
                * topk_weights.view(m, num_topk, 1).to(out_dtype)
            ).sum(dim=1),
            non_blocking=True,
        )
    else:
        output.copy_(c3.view(m, num_topk, k).sum(dim=1), non_blocking=True)
    return


class CutlassExpertsFp4(mk.FusedMoEExpertsModular):
    """CUTLASS FP4 fused MoE expert implementation."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Fuse activation scales into w_scale_2 in-place so that
        # g1/g2_alphas (which reference the same tensor) stay in sync
        # when EPLB rearranges the parameter.
        layer.w13_weight_scale_2.data.mul_(layer.w13_input_scale)
        layer.w2_weight_scale_2.data.mul_(layer.w2_input_scale)

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return p.is_cuda() and (
            p.is_device_capability_family(100)
            or p.is_device_capability_family(110)
            or p.is_device_capability_family(120)
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kNvfp4Static, kNvfp4Dynamic)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # SILU uses a fused silu+mul+fp4_quant kernel path.
        # Other gated activations use the generic apply_moe_activation()
        # fallback + separate fp4 quantization in run_cutlass_moe_fp4().
        # Non-gated activations (_NO_MUL) are also supported for models
        # like Nemotron-Nano that don't use gated MLP.
        return apply_moe_activation_supported(activation)

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # CutlassExpertsFp4 does not support expert map, which is
        # needed for STANDARD activation format kernels in EP mode.
        return moe_parallel_config.ep_size == 1

    @staticmethod
    def _supports_batch_invariance() -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        return act_dtype

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace1 = (M * topk, max(2 * N, K))
        workspace2 = (M * topk, N)
        output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,  # unused
        a2_scale: torch.Tensor | None,  # unused
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        e, m, n, k, _ = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        n = w2.shape[2] * 2

        run_cutlass_moe_fp4(
            output=output,
            a=hidden_states,
            a1_gscale=self.a1_gscale,
            w1_fp4=w1,
            w1_blockscale=self.w1_scale,
            w1_alphas=self.g1_alphas,
            a2_gscale=self.a2_gscale,
            w2_fp4=w2,
            w2_blockscale=self.w2_scale,
            w2_alphas=self.g2_alphas,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            workspace13=workspace13,
            workspace2=workspace2,
            m=m,
            n=n,
            k=k,
            e=e,
            device=hidden_states.device,
            apply_router_weight_on_input=apply_router_weight_on_input,
            activation_config=self.activation_config,
        )


def run_cutlass_moe_mxfp4(
    output: torch.Tensor,
    a: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    m: int,
    n: int,
    k: int,
    e: int,
    device: torch.device,
    apply_router_weight_on_input: bool = False,
    *,
    activation_config: ApplyMoEActivationConfig | None = None,
) -> None:
    """MXFP4 x MXFP4 MoE implementation using CUTLASS grouped GEMM."""
    is_gated = activation.is_gated
    w1_n = n * 2 if is_gated else n

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1_fp4.dtype == torch.uint8, "weight 1 must be uint8"
    assert w2_fp4.dtype == torch.uint8, "weight 2 must be uint8"
    assert (
        w1_fp4.ndim == 3
        and w2_fp4.ndim == 3
        and w1_blockscale.ndim == 3
        and w2_blockscale.ndim == 3
    ), "All Weights must be of rank 3 for cutlass_moe_mxfp4"
    m_a, k_a = a.shape
    e_w1, w1_n_actual, half_k_w1 = w1_fp4.shape
    e_w2, k_w2, half_n_w2 = w2_fp4.shape

    assert e_w1 == e_w2 and e_w1 == e
    assert k_a == half_k_w1 * 2 and k == k_w2
    assert w1_n_actual == w1_n and half_n_w2 * 2 == n
    assert m == m_a
    assert 2 * half_k_w1 == k_w2
    assert a.dtype in [torch.half, torch.bfloat16], "Invalid input dtype"
    assert topk_weights.size(0) == m and topk_ids.size(0) == m

    topk = topk_ids.size(1)
    out_dtype = a.dtype
    num_topk = topk_ids.size(1)

    expert_offsets = torch.empty((e + 1), dtype=torch.int32, device=device)
    blockscale_offsets = torch.empty((e + 1), dtype=torch.int32, device=device)
    problem_sizes1 = torch.empty((e, 3), dtype=torch.int32, device=device)
    problem_sizes2 = torch.empty((e, 3), dtype=torch.int32, device=device)

    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)

    if apply_router_weight_on_input:
        assert num_topk == 1, (
            "apply_router_weight_on_input is only implemented for topk=1"
        )
        a.mul_(topk_weights.to(out_dtype))

    ops.get_cutlass_moe_mm_data(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        e,
        n,
        k,
        blockscale_offsets,
        is_gated=is_gated,
    )

    a = ops.shuffle_rows(a, a_map)
    rep_a_fp4, rep_a_blockscale = ops.mxfp4_experts_quant(
        a,
        expert_offsets,
        blockscale_offsets,
        e,
        num_topk,
    )
    c1 = _resize_cache(workspace13, (m * topk, w1_n))
    c2 = _resize_cache(workspace2, (m * topk, n))
    c3 = _resize_cache(workspace13, (m * topk, k))

    ops.cutlass_mxfp4_moe_mm(
        c1,
        rep_a_fp4,
        w1_fp4,
        rep_a_blockscale,
        w1_blockscale,
        problem_sizes1,
        expert_offsets[:-1],
        blockscale_offsets[:-1],
    )
    del rep_a_fp4, rep_a_blockscale
    if activation == MoEActivation.SILU and (
        activation_config is None or activation_config.clamp_limit is None
    ):
        int_fp4, int_blockscale = ops.silu_and_mul_mxfp4_experts_quant(
            c1, expert_offsets, blockscale_offsets, e, num_topk
        )
    else:
        apply_moe_activation(
            activation,
            c2,
            c1,
            activation_config=activation_config,
        )
        int_fp4, int_blockscale = ops.mxfp4_experts_quant(
            c2, expert_offsets, blockscale_offsets, e, num_topk
        )

    ops.cutlass_mxfp4_moe_mm(
        c3,
        int_fp4,
        w2_fp4,
        int_blockscale,
        w2_blockscale,
        problem_sizes2,
        expert_offsets[:-1],
        blockscale_offsets[:-1],
    )
    del int_fp4, int_blockscale

    c3 = ops.shuffle_rows(c3, c_map)

    assert output.dtype == out_dtype
    if not apply_router_weight_on_input:
        output.copy_(
            (
                c3.view(m, num_topk, k)
                * topk_weights.view(m, num_topk, 1).to(out_dtype)
            ).sum(dim=1),
            non_blocking=True,
        )
    else:
        output.copy_(c3.view(m, num_topk, k).sum(dim=1), non_blocking=True)
    return


def swizzle_mxfp4_scales(
    scales: torch.Tensor,
    N: int,
    K: int,
) -> torch.Tensor:
    """Swizzle flat [N, K//32] E8M0 scales to CUTLASS tiled layout.

    CUTLASS expects MX scale factors in a tiled layout:
        [numMTiles, numKTiles, 32, 4, 4]
    where numMTiles = ceil(N/128), numKTiles = ceil(K/128),
    and the inner dimensions correspond to the swizzle pattern:
        mTileIdx = mIdx / 128
        outerMIdx = mIdx % 32
        innerMIdx = (mIdx / 32) % 4
        kTileIdx = kIdx / 4
        innerKIdx = kIdx % 4
    with kIdx = col_in_scale_space (i.e., index into K//32).
    """
    assert scales.dtype == torch.uint8
    num_scale_cols = K // 32  # number of E8M0 scale values per row

    num_m_tiles = (N + 127) // 128
    num_k_tiles = (num_scale_cols + 3) // 4

    # Pad N to multiple of 128 and scale_cols to multiple of 4
    padded_N = num_m_tiles * 128
    padded_scale_cols = num_k_tiles * 4

    # Start with flat scales, pad if needed
    padded = torch.zeros(
        padded_N, padded_scale_cols, dtype=torch.uint8, device=scales.device
    )
    padded[:N, :num_scale_cols] = scales

    # Reshape to tile structure:
    # [numMTiles, 4, 32, numKTiles, 4]
    #  mTileIdx, innerMIdx, outerMIdx, kTileIdx, innerKIdx
    tiled = padded.reshape(num_m_tiles, 4, 32, num_k_tiles, 4)
    # Permute to [numMTiles, numKTiles, 32, 4, 4]
    #            (outerMIdx, innerMIdx, innerKIdx)
    tiled = tiled.permute(0, 3, 2, 1, 4).contiguous()
    return tiled.reshape(-1)


class CutlassExpertsMxfp4(mk.FusedMoEExpertsModular):
    """CUTLASS MXFP4 x MXFP4 fused MoE expert implementation."""

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        capability = p.get_device_capability()
        return (
            p.is_cuda()
            and capability is not None
            and ops.mxfp4_experts_quant_supported(capability.to_int())
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kMxfp4Static, kMxfp4Dynamic)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return apply_moe_activation_supported(activation)

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return moe_parallel_config.ep_size == 1

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        return act_dtype

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace1 = (M * topk, max(2 * N, K))
        workspace2 = (M * topk, N)
        output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        e, m, n, k, _ = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        n = w2.shape[2] * 2

        run_cutlass_moe_mxfp4(
            output=output,
            a=hidden_states,
            w1_fp4=w1,
            w1_blockscale=self.w1_scale,
            w2_fp4=w2,
            w2_blockscale=self.w2_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            workspace13=workspace13,
            workspace2=workspace2,
            m=m,
            n=n,
            k=k,
            e=e,
            device=hidden_states.device,
            apply_router_weight_on_input=apply_router_weight_on_input,
            activation_config=self.activation_config,
        )


# W4A8
def run_cutlass_moe_w4a8_fp8(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    a1q_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    w1_chan_scale: torch.Tensor,
    w2_chan_scale: torch.Tensor,
    a_strides1: torch.Tensor,
    a_strides2: torch.Tensor,
    b_strides1: torch.Tensor,
    b_strides2: torch.Tensor,
    c_strides1: torch.Tensor,
    c_strides2: torch.Tensor,
    s_strides1: torch.Tensor,
    s_strides2: torch.Tensor,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    expert_num_tokens: torch.Tensor | None,
    out_dtype: torch.dtype,
    per_act_token: bool,
    per_out_ch: bool,
    use_batched_format: bool,
    topk_weights: torch.Tensor | None,
    group_size: int,
    permute_scratch: MoEPermuteScratch | None,
    *,
    activation_config: ApplyMoEActivationConfig | None = None,
):
    a1q = hidden_states
    M = a1q.size(0)
    local_E = w1.size(0)
    device = a1q.device
    _, K, N_packed = w2.shape
    N = N_packed * 8  # logical N, pack 8 int4 into 1 int32

    assert per_act_token, "W4A8 must use per-token scales"
    assert per_out_ch, "W4A8 must use per-channel scales"
    assert w1_scale is not None
    assert w2_scale is not None
    assert w1_scale.dtype == torch.float8_e4m3fn
    assert w2_scale.dtype == torch.float8_e4m3fn
    assert w1.dtype == torch.int32
    assert w2.dtype == torch.int32
    assert w1_chan_scale.dtype == torch.float32
    assert w2_chan_scale.dtype == torch.float32
    assert w1.size(0) == w2.size(0), "Weights expert number mismatch"
    assert a1q_scale is not None
    assert a2_scale is None
    assert out_dtype in [torch.bfloat16], f"Invalid output dtype: {out_dtype}"
    if expert_map is not None:
        assert expert_num_tokens is None
    assert not use_batched_format, "batched format not supported yet"
    assert group_size == 128, f"Only group size 128 supported but got {group_size=}"

    assert global_num_experts != -1
    assert w1.size(2) * 8 == K, (
        f"w1 hidden size mismatch: got {w1.size(2) * 8}, expected {K=}"
    )

    topk = topk_ids.size(1)
    a1q_perm = _resize_cache(workspace2.view(dtype=torch.float8_e4m3fn), (M * topk, K))
    mm1_out = _resize_cache(workspace13, (M * topk, N * 2))
    act_out = _resize_cache(workspace2, (M * topk, N))
    # original workspace are based on input hidden_states dtype (bf16)
    quant_out = _resize_cache(
        workspace13.view(dtype=torch.float8_e4m3fn), (M * topk, N)
    )
    mm2_out = _resize_cache(workspace2, (M * topk, K))

    problem_sizes1 = torch.empty((local_E, 3), dtype=torch.int32, device=device)
    problem_sizes2 = torch.empty((local_E, 3), dtype=torch.int32, device=device)

    num_expert = global_num_experts if expert_map is None else expert_map.size(0)
    # permuted a1q reuses workspace2
    a1q, a1q_scale, expert_first_token_offset, inv_perm, _ = moe_permute(
        a1q,
        a1q_scale,
        topk_ids,
        num_expert,
        local_E,
        expert_map,
        permuted_hidden_states=a1q_perm,
        scratch=permute_scratch,
    )
    # for RS gemm SwapAB is always enabled (swap logical M, N in the problem shape).
    ops.get_cutlass_moe_mm_problem_sizes_from_expert_offsets(
        expert_first_token_offset, problem_sizes1, problem_sizes2, N, K, True
    )
    expert_offsets = expert_first_token_offset[:-1]

    ops.cutlass_w4a8_moe_mm(
        mm1_out,
        a1q,
        w1,
        a1q_scale,
        w1_chan_scale,
        w1_scale,
        group_size,
        expert_offsets,
        problem_sizes1,
        a_strides1,
        b_strides1,
        c_strides1,
        s_strides1,
    )

    apply_moe_activation(
        activation,
        act_out,
        mm1_out,
        activation_config=activation_config,
    )

    a2q, a2q_scale = ops.scaled_fp8_quant(
        act_out, a2_scale, use_per_token_if_dynamic=per_act_token, output=quant_out
    )

    ops.cutlass_w4a8_moe_mm(
        mm2_out,
        a2q,
        w2,
        a2q_scale,
        w2_chan_scale,
        w2_scale,
        group_size,
        expert_offsets,
        problem_sizes2,
        a_strides2,
        b_strides2,
        c_strides2,
        s_strides2,
    )

    # for non-chunking mode the output is resized from workspace13
    # so we need to make sure mm2_out uses workspace2.
    moe_unpermute(
        out=output,
        permuted_hidden_states=mm2_out,
        topk_weights=topk_weights,
        inv_permuted_idx=inv_perm,
        expert_first_token_offset=expert_first_token_offset,
    )


class CutlassExpertsW4A8Fp8(mk.FusedMoEExpertsModular):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        b_strides1: torch.Tensor,
        b_strides2: torch.Tensor,
        group_size: int,
    ):
        super().__init__(moe_config=moe_config, quant_config=quant_config)

        e = moe_config.num_local_experts
        n = moe_config.intermediate_size_per_partition
        k = moe_config.hidden_dim
        device = moe_config.device

        self.out_dtype = moe_config.in_dtype

        a_strides1_c_strides2 = torch.full((e,), k, device=device, dtype=torch.int64)
        self.a_strides1 = a_strides1_c_strides2
        self.a_strides2 = torch.full((e,), n, device=device, dtype=torch.int64)
        self.c_strides1 = torch.full((e,), 2 * n, device=device, dtype=torch.int64)
        self.c_strides2 = a_strides1_c_strides2

        self.b_strides1 = b_strides1
        self.b_strides2 = b_strides2

        # sizeof(StrideS) = 16 bytes, encoded as 2xint64.
        self.s_strides1 = torch.zeros((e, 2), device=device, dtype=torch.int64)
        self.s_strides1[:, 0] = 2 * n
        self.s_strides2 = torch.zeros((e, 2), device=device, dtype=torch.int64)
        self.s_strides2[:, 0] = k

        self.group_size = group_size
        self._permute_scratch: MoEPermuteScratch | None = None

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        if moe_config.in_dtype != torch.bfloat16:
            return (
                False,
                f"kernel does not support {moe_config.in_dtype} input/output dtype",
            )

        return mk.FusedMoEExperts.is_supported_config(
            cls,
            moe_config,
            weight_key,
            activation_key,
            activation_format,
        )

    @staticmethod
    def _supports_current_device() -> bool:
        return cutlass_group_gemm_supported()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kInt4Static, kFp8DynamicTokenSym)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation.is_gated and apply_moe_activation_supported(activation)

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # topk weights and reduction are fused in moe_unpermute cuda kernel
        return TopKWeightAndReduceNoOP()

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        return self.out_dtype if self.out_dtype is not None else act_dtype

    def _get_permute_scratch(self) -> MoEPermuteScratch | None:
        if self._permute_scratch is None and moe_permute_unpermute_supported():
            self._permute_scratch = MoEPermuteScratch(
                max_num_tokens=self.moe_config.max_num_tokens,
                topk=self.moe_config.experts_per_token,
                num_experts=self.moe_config.num_experts,
                num_local_experts=self.moe_config.num_local_experts,
                device=torch.device(self.moe_config.device),
            )
        return self._permute_scratch

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M * topk, max(N, K))
        workspace2 = (M * topk, max(activation_out_dim, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert self.w1_zp is None, "w1_zp is not supported in CUTLASS MoE"
        assert self.w2_zp is None, "w2_zp is not supported in CUTLASS MoE"

        expert_num_tokens = None

        use_batched_format = (
            self.activation_format() == mk.FusedMoEActivationFormat.BatchedExperts
        )
        assert not use_batched_format, "batched format not supported"

        in_dtype = hidden_states.dtype

        run_cutlass_moe_w4a8_fp8(
            output,
            hidden_states,
            w1,
            w2,
            topk_ids,
            activation,
            global_num_experts,
            expert_map,
            self.w1_scale,
            self.w2_scale,
            a1q_scale,
            a2_scale,
            self.g1_alphas,  # per-channel scales
            self.g2_alphas,  # per-channel scales
            self.a_strides1,
            self.a_strides2,
            self.b_strides1,
            self.b_strides2,
            self.c_strides1,
            self.c_strides2,
            self.s_strides1,
            self.s_strides2,
            workspace13,
            workspace2,
            expert_num_tokens,
            self.out_dtype if self.out_dtype is not None else in_dtype,
            self.per_act_token_quant,
            self.per_out_ch_quant,
            use_batched_format,
            topk_weights,
            self.group_size,
            self._get_permute_scratch(),
            activation_config=self.activation_config,
        )


# ---------------------------------------------------------------- engv2: fused unpermute + topk-weighted sum (moe_unperm.py)
import os as _reax_os

import triton as _reax_triton
import triton.language as _reax_tl


@_reax_triton.jit
def _reax_unperm_p(c3, cmap, w, t, j, K, cols, mask):
    """bf16(float(c3[c_map[t * 8 + j], cols]) * float(bf16(w[t, j]))) widened to fp32 (ATen's bf16 mul)."""
    src = _reax_tl.load(cmap + t * 8 + j).to(_reax_tl.int64)
    x = _reax_tl.load(c3 + src * K + cols, mask=mask, other=0.0).to(_reax_tl.float32)
    wj = _reax_tl.load(w + t * 8 + j).to(_reax_tl.bfloat16).to(_reax_tl.float32)
    return (x * wj).to(_reax_tl.bfloat16).to(_reax_tl.float32)


@_reax_triton.jit
def _reax_unperm_kernel(c3, cmap, w, out, K, BLOCK: _reax_tl.constexpr, ORDER: _reax_tl.constexpr):
    t = _reax_tl.program_id(0)
    cols = _reax_tl.program_id(1) * BLOCK + _reax_tl.arange(0, BLOCK)
    mask = cols < K
    p0 = _reax_unperm_p(c3, cmap, w, t, 0, K, cols, mask)
    p1 = _reax_unperm_p(c3, cmap, w, t, 1, K, cols, mask)
    p2 = _reax_unperm_p(c3, cmap, w, t, 2, K, cols, mask)
    p3 = _reax_unperm_p(c3, cmap, w, t, 3, K, cols, mask)
    p4 = _reax_unperm_p(c3, cmap, w, t, 4, K, cols, mask)
    p5 = _reax_unperm_p(c3, cmap, w, t, 5, K, cols, mask)
    p6 = _reax_unperm_p(c3, cmap, w, t, 6, K, cols, mask)
    p7 = _reax_unperm_p(c3, cmap, w, t, 7, K, cols, mask)
    z = _reax_tl.zeros([BLOCK], dtype=_reax_tl.float32)  # ATen starts every accumulator at the identity 0.0
    if ORDER == 0:  # one accumulator, j = 0 .. 7
        acc = (((((((z + p0) + p1) + p2) + p3) + p4) + p5) + p6) + p7
    elif ORDER == 1:  # 4 accumulators (j mod 4), combined ((a0 + a1) + a2) + a3 (ATen thread_reduce, vt0 = 4)
        acc = ((((z + p0) + p4) + ((z + p1) + p5)) + ((z + p2) + p6)) + ((z + p3) + p7)
    elif ORDER == 2:  # 4 accumulators combined (a0 + a1) + (a2 + a3)
        acc = (((z + p0) + p4) + ((z + p1) + p5)) + (((z + p2) + p6) + ((z + p3) + p7))
    elif ORDER == 3:  # block.y = 2: thread y sums j = y, y+2, y+4, y+6 (one accumulator each), then y0 + y1
        acc = ((((z + p0) + p2) + p4) + p6) + ((((z + p1) + p3) + p5) + p7)
    elif ORDER == 4:  # block.y = 4 or 8: (j, j+4) per thread, shared-memory tree (y0 + y2) + (y1 + y3)
        acc = (((z + p0) + p4) + ((z + p2) + p6)) + (((z + p1) + p5) + ((z + p3) + p7))
    else:  # pairwise tree
        acc = (((z + p0) + p1) + ((z + p2) + p3)) + (((z + p4) + p5) + ((z + p6) + p7))
    _reax_tl.store(out + t * K + cols, acc.to(_reax_tl.bfloat16), mask=mask)


_REAX_UNPERM = {"order": None}


def _reax_unperm_launch(output, c3, c_map, topk_weights, m, topk, k, order):
    BLOCK = 512
    _reax_unperm_kernel[(m, _reax_triton.cdiv(k, BLOCK))](c3, c_map, topk_weights, output, k, BLOCK=BLOCK, ORDER=order,
                                                          num_warps=4)


def _reax_unperm_eager(c3, c_map, topk_weights, m, topk, k):
    c3s = ops.shuffle_rows(c3, c_map)
    return (c3s.view(m, topk, k) * topk_weights.view(m, topk, 1).to(c3.dtype)).sum(dim=1)


def _reax_unperm_data(m, topk, k, g):
    """c3 with exponents spread over 2^-24 .. 2^24 (fp32 sums of the products are then rarely exact, so any change of
    summation order shows), a random row permutation and normalised routing weights."""
    e = torch.randint(-24, 25, (m * topk, k), device="cuda", generator=g, dtype=torch.int16).to(torch.bfloat16)
    c3 = torch.randn(m * topk, k, device="cuda", generator=g, dtype=torch.bfloat16) * torch.exp2(e)  # exact in bf16
    del e
    perm = torch.randperm(m * topk, device="cuda", generator=g).to(torch.int32)
    w = torch.rand(m, topk, device="cuda", generator=g).float()
    return c3, perm, w / w.sum(-1, keepdim=True)


_REAX_UNPERM_M = (1, 2, 3, 5, 8, 13, 16, 31, 64, 100, 128, 255, 256, 512, 1000, 1024, 2048, 4096, 8192)


def _reax_unperm_check(topk, k, dtype):
    """First order that reproduces the eager ops bit for bit (sign of zero included) at every m probed."""
    g = torch.Generator(device="cuda").manual_seed(5150)
    for order in (1, 0, 2, 3, 4, 5):
        ok = True
        for m in _REAX_UNPERM_M:
            c3, perm, w = _reax_unperm_data(m, topk, k, g)
            ref = _reax_unperm_eager(c3, perm, w, m, topk, k)
            out = torch.empty(m, k, device="cuda", dtype=dtype)
            _reax_unperm_launch(out, c3, perm, w, m, topk, k, order)
            if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):  # bit patterns: -0.0 != +0.0
                ok = False
                break
        if ok:
            return order
    return -1


def _reax_unperm(output, c3, c_map, topk_weights, m, topk, k):
    if _reax_os.environ.get("REAX_MOE_UNPERM", "1") != "1" or c3.dtype != torch.bfloat16 or topk != 8:
        return False
    if not (c3.is_contiguous() and c_map.is_contiguous() and topk_weights.is_contiguous() and output.is_contiguous()):
        return False
    st = _REAX_UNPERM
    if st["order"] is None:
        if torch.cuda.is_current_stream_capturing():
            return False
        path = _reax_os.environ.get("REAX_MOE_UNPERM_FILE", "")
        if path and _reax_os.path.exists(path):
            st["order"] = int(open(path).read().split()[0])
            how = f"loaded from {path}"
        else:
            st["order"] = _reax_unperm_check(topk, k, c3.dtype)
            how = "checked in-engine"
        print(f"reax moe_unperm: order {st['order']} {how} "
              f"({'bit-exact vs shuffle_rows + mul + sum' if st['order'] >= 0 else 'no order matches: eager path'})", flush=True)
    if st["order"] < 0:
        return False
    _reax_unperm_launch(output, c3[: m * topk], c_map, topk_weights, m, topk, k, st["order"])
    return True


# ---------------------------------------------------------------- engv2: NVFP4 experts quant reading rows through a_map (moe_gquant.py)
import ctypes as _reax_ctypes
import os as _reax_gq_os

_REAX_GQ = {"on": None, "lib": None, "sms": 0}


def _reax_gq_lib():
    st = _REAX_GQ
    if st["lib"] is None:
        lib = (_reax_ctypes.PyDLL if _reax_gq_os.environ.get("REAX_PYDLL", "0") == "1" else _reax_ctypes.CDLL)(_reax_gq_os.environ.get("REAX_MOE_GQUANT_LIB", "/opt/engv2/libreax_gquant.so"))
        lib.reax_gquant_launch.restype = _reax_ctypes.c_int
        lib.reax_gquant_launch.argtypes = [_reax_ctypes.c_int, _reax_ctypes.c_int] + [_reax_ctypes.c_void_p] * 7 + [
            _reax_ctypes.c_int, _reax_ctypes.c_int, _reax_ctypes.c_void_p]
        st["lib"] = lib
        st["sms"] = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return st["lib"]


def _reax_gquant_raw(a, a_map, a1_gscale, expert_offsets, blockscale_offsets, out, sf):
    rc = _reax_gq_lib().reax_gquant_launch(
        a_map.shape[0], a.shape[1], a.data_ptr(), a_map.data_ptr(), a1_gscale.data_ptr(), out.data_ptr(), sf.data_ptr(),
        expert_offsets.data_ptr(), blockscale_offsets.data_ptr(), a1_gscale.shape[0], _REAX_GQ["sms"],
        torch.cuda.current_stream().cuda_stream)
    if rc != 0:
        raise RuntimeError(f"reax_gquant_launch: CUDA error {rc}")


def _reax_gquant(a, a_map, a1_gscale, expert_offsets, blockscale_offsets, topk):
    """Same outputs (and allocation) as ops.scaled_fp4_experts_quant(ops.shuffle_rows(a, a_map), ...), or None."""
    st = _REAX_GQ
    if st["on"] is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        st["on"] = False
        if _reax_gq_os.environ.get("REAX_MOE_GQUANT", "1") == "1":
            path = _reax_gq_os.environ.get("REAX_MOE_GQUANT_FILE", "")
            try:
                if path and _reax_gq_os.path.exists(path):
                    st["on"] = open(path).read().split()[0] == "1"
                    how = f"check result loaded from {path}"
                else:
                    st["on"] = _reax_gquant_check(small=True)
                    how = "checked in-engine"
            except Exception as e:  # noqa: BLE001 - keep the eager ops
                how = f"disabled ({e!r})"
            print(f"reax moe_gquant: {'on' if st['on'] else 'off'} ({how})", flush=True)
    if not st["on"]:
        return None
    if (a.dtype != torch.bfloat16 or not a.is_contiguous() or a_map.dtype != torch.int32 or a.shape[1] % 16 != 0
            or a1_gscale.dtype != torch.float32):
        return None
    m_numtopk, k = a_map.shape[0], a.shape[1]
    if m_numtopk > envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE * topk:
        return None
    padded_k = (k // 16 + 3) // 4
    out = torch.empty(m_numtopk, k // 2, device=a.device, dtype=torch.uint8)
    sf = torch.empty(envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE * topk, padded_k, dtype=torch.int32, device=a.device)
    _reax_gquant_raw(a, a_map, a1_gscale, expert_offsets, blockscale_offsets, out, sf)
    return out, sf.view(torch.float8_e4m3fn)


def _reax_gq_case(m, k, e, topk, g, kind):
    """Hidden states, routing and per-expert global scales; `kind` picks the value distribution."""
    if kind == 0:  # model-like
        a = (torch.randn(m, k, device="cuda", generator=g) * 0.5).bfloat16()
    elif kind == 1:  # exponents over 2^-133 .. 2^30: bf16 subnormals, zeros (underflow), huge blocks
        ex = torch.randint(-133, 31, (m, k), device="cuda", generator=g, dtype=torch.int32).float()
        a = (torch.randn(m, k, device="cuda", generator=g) * torch.exp2(ex)).bfloat16()
    else:  # small integers and halves: exact e2m1 ties, all-zero and constant 16-blocks
        a = (torch.randint(-12, 13, (m, k), device="cuda", generator=g).float() * 0.5).bfloat16()
        blk = a.view(m, k // 16, 16)
        z = torch.rand(m, k // 16, device="cuda", generator=g) < 0.1
        blk[z] = 0
        c = torch.rand(m, k // 16, device="cuda", generator=g) < 0.05
        blk[c] = blk[c][:, :1].expand(-1, 16).clone()
    logits = torch.randn(m, e, device="cuda", generator=g)
    if kind == 2:
        logits[:, : e // 8] += 3.0  # crowded experts: > 128 rows in some (several SF row tiles)
    topk_ids = logits.topk(topk, dim=-1).indices.to(torch.int32)
    gscale = torch.exp2(torch.randint(-6, 13, (e,), device="cuda", generator=g).float()) * (
        1.0 + torch.rand(e, device="cuda", generator=g))
    return a, topk_ids, gscale.float()


def _reax_gquant_check(small=False):
    """Byte-for-byte: packed FP4 output and the whole (pre-zeroed) E4M3 scale buffer vs shuffle_rows + vLLM quant."""
    g = torch.Generator(device="cuda").manual_seed(5151)
    k, e, topk = 2816, 128, 8
    ms = (1, 3, 8, 31, 64, 200) if small else (1, 2, 3, 7, 8, 16, 31, 64, 100, 128, 200, 333, 512, 1000, 1024, 2048)
    n = 0
    for m in ms:
        for kind in (0, 1, 2):
            a, topk_ids, gscale = _reax_gq_case(m, k, e, topk, g, kind)
            expert_offsets = torch.empty(e + 1, dtype=torch.int32, device="cuda")
            blockscale_offsets = torch.empty(e + 1, dtype=torch.int32, device="cuda")
            ps1 = torch.empty((e, 3), dtype=torch.int32, device="cuda")
            ps2 = torch.empty((e, 3), dtype=torch.int32, device="cuda")
            a_map = torch.empty(topk_ids.numel(), dtype=torch.int32, device="cuda")
            c_map = torch.empty(topk_ids.numel(), dtype=torch.int32, device="cuda")
            ops.get_cutlass_moe_mm_data(topk_ids, expert_offsets, ps1, ps2, a_map, c_map, e, 704, k, blockscale_offsets,
                                        is_gated=True)
            padded_k = (k // 16 + 3) // 4
            rows = envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE * topk
            ref_o = torch.empty(m * topk, k // 2, device="cuda", dtype=torch.uint8)
            ref_s = torch.zeros(rows, padded_k, dtype=torch.int32, device="cuda")
            torch.ops._C.scaled_fp4_experts_quant(ref_o, ref_s, ops.shuffle_rows(a, a_map), gscale, expert_offsets,
                                                  blockscale_offsets)
            out_o = torch.empty_like(ref_o)
            out_s = torch.zeros_like(ref_s)
            _reax_gquant_raw(a, a_map, gscale, expert_offsets, blockscale_offsets, out_o, out_s)
            torch.cuda.synchronize()
            if not (torch.equal(out_o, ref_o) and torch.equal(out_s, ref_s)):
                print(f"reax moe_gquant: MISMATCH at m {m} kind {kind}: fp4 bytes "
                      f"{int((out_o != ref_o).sum())}, scale words {int((out_s != ref_s).sum())}", flush=True)
                return False
            n += 1
    print(f"reax moe_gquant: byte-identical in {n} cases (m up to {ms[-1]}, k {k}, {e} experts, top-{topk})", flush=True)
    return True

import vllm.envs as envs


# ---------------------------------------------------------------- engv2: gelu_tanh_and_mul + NVFP4 experts quant (moe_gact.py)
import ctypes as _reax_ga_ctypes
import os as _reax_ga_os

_REAX_GA = {"variant": None, "lib": None, "sms": 0}


def _reax_ga_lib():
    st = _REAX_GA
    if st["lib"] is None:
        c = _reax_ga_ctypes
        lib = (c.PyDLL if _reax_ga_os.environ.get("REAX_PYDLL", "0") == "1" else c.CDLL)(_reax_ga_os.environ.get("REAX_MOE_GACT_LIB", "/opt/engv2/libreax_gact.so"))
        lib.reax_gact_launch.restype = c.c_int
        lib.reax_gact_launch.argtypes = [c.c_int, c.c_int, c.c_int] + [c.c_void_p] * 6 + [c.c_int, c.c_int, c.c_float,
                                                                                          c.c_void_p]
        lib.reax_gact_configs.restype = c.c_int
        st["lib"] = lib
        st["sms"] = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return st["lib"]


def _reax_gact_raw(variant, c1, a2_gscale, expert_offsets, blockscale_offsets, out, sf):
    rc = _reax_ga_lib().reax_gact_launch(
        variant, c1.shape[0], c1.shape[1] // 2, c1.data_ptr(), a2_gscale.data_ptr(), out.data_ptr(), sf.data_ptr(),
        expert_offsets.data_ptr(), blockscale_offsets.data_ptr(), a2_gscale.shape[0], _REAX_GA["sms"], 0.0,
        torch.cuda.current_stream().cuda_stream)
    if rc != 0:
        raise RuntimeError(f"reax_gact_launch: CUDA error {rc}")


def _reax_gact(activation, activation_config, c1, a2_gscale, expert_offsets, blockscale_offsets, topk):
    """Same outputs (and allocation) as gelu_tanh_and_mul + ops.scaled_fp4_experts_quant, or None."""
    st = _REAX_GA
    if st["variant"] is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        st["variant"] = -1
        if _reax_ga_os.environ.get("REAX_MOE_GACT", "1") == "1":
            path = _reax_ga_os.environ.get("REAX_MOE_GACT_FILE", "")
            try:
                if path and _reax_ga_os.path.exists(path):
                    st["variant"] = int(open(path).read().split()[0])
                    how = f"check result loaded from {path}"
                else:
                    st["variant"] = _reax_gact_pick(small=True)
                    how = "checked in-engine"
            except Exception as e:  # noqa: BLE001 - keep the eager ops
                how = f"disabled ({e!r})"
            print(f"reax moe_gact: variant {st['variant']} ({how})", flush=True)
    if st["variant"] < 0 or activation != MoEActivation.GELU_TANH:
        return None
    if activation_config is not None and activation_config.clamp_limit is not None:
        return None
    if (c1.dtype != torch.bfloat16 or not c1.is_contiguous() or c1.dim() != 2 or c1.shape[1] % 32 != 0
            or a2_gscale.dtype != torch.float32):
        return None
    m_numtopk, k = c1.shape[0], c1.shape[1] // 2
    if m_numtopk > envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE * topk:
        return None
    padded_k = (k // 16 + 3) // 4
    out = torch.empty(m_numtopk, k // 2, device=c1.device, dtype=torch.uint8)
    sf = torch.empty(envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE * topk, padded_k, dtype=torch.int32, device=c1.device)
    _reax_gact_raw(st["variant"], c1, a2_gscale, expert_offsets, blockscale_offsets, out, sf)
    return out, sf.view(torch.float8_e4m3fn)


def _reax_ga_case(m, k, e, topk, g, kind):
    """c1 = [gate | up] rows of the gate|up GEMM output (m * topk rows, 2k wide), routing, per-expert global scales."""
    rows = m * topk
    if kind == 0:  # model-like
        c1 = (torch.randn(rows, 2 * k, device="cuda", generator=g) * 1.5).bfloat16()
    elif kind == 1:  # exponents 2^-133 .. 2^20: bf16 subnormals, zeros, gelu saturation, huge products
        ex = torch.randint(-133, 21, (rows, 2 * k), device="cuda", generator=g, dtype=torch.int32).float()
        c1 = (torch.randn(rows, 2 * k, device="cuda", generator=g) * torch.exp2(ex)).bfloat16()
    else:  # small integers and halves, signed zeros in up, constant and all-zero 16-blocks
        c1 = (torch.randint(-12, 13, (rows, 2 * k), device="cuda", generator=g).float() * 0.5).bfloat16()
        up = c1[:, k:]
        z = torch.rand(rows, k, device="cuda", generator=g) < 0.15
        up[z] = torch.where(torch.rand(rows, k, device="cuda", generator=g) < 0.5, 0.0, -0.0).bfloat16()[z]
        gate = c1[:, :k].reshape(rows, k // 16, 16)
        c = torch.rand(rows, k // 16, device="cuda", generator=g) < 0.05
        gate[c] = gate[c][:, :1].expand(-1, 16).clone()
    logits = torch.randn(m, e, device="cuda", generator=g)
    if kind == 2:
        logits[:, : e // 8] += 3.0  # crowded experts: > 128 rows in some
    topk_ids = logits.topk(topk, dim=1).indices.int()
    gscale = torch.exp2(torch.randint(-6, 13, (e,), device="cuda", generator=g).float()) * (
        1.0 + torch.rand(e, device="cuda", generator=g))
    return c1, topk_ids, gscale.float()


def _reax_gact_check(variant, small=False):
    """Byte-for-byte: packed FP4 output and the whole (pre-zeroed) E4M3 scale buffer vs gelu_tanh_and_mul + vLLM quant."""
    g = torch.Generator(device="cuda").manual_seed(6161)
    k, e, topk = 704, 128, 8
    ms = (1, 3, 8, 31, 64, 200) if small else (1, 2, 3, 7, 8, 16, 31, 64, 100, 128, 200, 333, 512, 887, 1000, 1024, 2048)
    n = 0
    for m in ms:
        for kind in (0, 1, 2):
            c1, topk_ids, gscale = _reax_ga_case(m, k, e, topk, g, kind)
            expert_offsets = torch.empty(e + 1, dtype=torch.int32, device="cuda")
            blockscale_offsets = torch.empty(e + 1, dtype=torch.int32, device="cuda")
            ps1 = torch.empty((e, 3), dtype=torch.int32, device="cuda")
            ps2 = torch.empty((e, 3), dtype=torch.int32, device="cuda")
            a_map = torch.empty(topk_ids.numel(), dtype=torch.int32, device="cuda")
            c_map = torch.empty(topk_ids.numel(), dtype=torch.int32, device="cuda")
            ops.get_cutlass_moe_mm_data(topk_ids, expert_offsets, ps1, ps2, a_map, c_map, e, k, 2816, blockscale_offsets,
                                        is_gated=True)
            padded_k = (k // 16 + 3) // 4
            rows = envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE * topk
            c2 = torch.empty(m * topk, k, device="cuda", dtype=torch.bfloat16)
            torch.ops._C.gelu_tanh_and_mul(c2, c1)
            ref_o = torch.empty(m * topk, k // 2, device="cuda", dtype=torch.uint8)
            ref_s = torch.zeros(rows, padded_k, dtype=torch.int32, device="cuda")
            torch.ops._C.scaled_fp4_experts_quant(ref_o, ref_s, c2, gscale, expert_offsets, blockscale_offsets)
            out_o = torch.empty_like(ref_o)
            out_s = torch.zeros_like(ref_s)
            _reax_gact_raw(variant, c1, gscale, expert_offsets, blockscale_offsets, out_o, out_s)
            torch.cuda.synchronize()
            if not (torch.equal(out_o, ref_o) and torch.equal(out_s, ref_s)):
                print(f"reax moe_gact: variant {variant} MISMATCH at m {m} kind {kind}: fp4 bytes "
                      f"{int((out_o != ref_o).sum())}, scale words {int((out_s != ref_s).sum())}", flush=True)
                return False
            n += 1
    print(f"reax moe_gact: variant {variant} byte-identical in {n} cases (m up to {ms[-1]}, k {k}, {e} experts, "
          f"top-{topk})", flush=True)
    return True


def _reax_gact_bufs(m, g):
    k, e, topk = 704, 128, 8
    c1, topk_ids, gscale = _reax_ga_case(m, k, e, topk, g, 0)
    eo = torch.empty(e + 1, dtype=torch.int32, device="cuda")
    bo = torch.empty(e + 1, dtype=torch.int32, device="cuda")
    p1 = torch.empty((e, 3), dtype=torch.int32, device="cuda")
    p2 = torch.empty((e, 3), dtype=torch.int32, device="cuda")
    am = torch.empty(m * topk, dtype=torch.int32, device="cuda")
    cmap = torch.empty(m * topk, dtype=torch.int32, device="cuda")
    ops.get_cutlass_moe_mm_data(topk_ids, eo, p1, p2, am, cmap, e, k, 2816, bo, is_gated=True)
    out = torch.empty(m * topk, k // 2, device="cuda", dtype=torch.uint8)
    sf = torch.zeros(envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE * topk, (k // 16 + 3) // 4, dtype=torch.int32, device="cuda")
    return c1, gscale, eo, bo, out, sf


def _reax_gact_tune(numerics, verbose=False):
    """Byte-check every launch shape with this numerics (small set), time the exact ones (best of 2) at a step's m."""
    from triton.testing import do_bench
    g = torch.Generator(device="cuda").manual_seed(17)
    bufs = [_reax_gact_bufs(m, g) for m in (440, 887, 1000)]
    best, best_t = numerics, None
    for c in range(_reax_ga_lib().reax_gact_configs()):
        v = numerics + 2 * c
        if c and not _reax_gact_check(v, small=True):
            continue
        tv = [min(do_bench(lambda: _reax_gact_raw(v, *b), warmup=25, rep=100) for _ in range(2)) for b in bufs]
        if verbose:
            print(f"  shape {c}: " + "  ".join(f"m {m} {t * 1e3:5.1f} us" for m, t in zip((440, 887, 1000), tv)),
                  flush=True)
        if best_t is None or sum(tv) < sum(best_t):
            best, best_t = v, tv
    return best


def _reax_gact_pick(small=False, tune=False):
    for v in (0, 1):
        if _reax_gact_check(v, small):
            return _reax_gact_tune(v) if tune else v
    return -1


# ---------------------------------------------------------------- engv2: MoE down GEMM through moe_dn.cu (moe_dn.py)
import ctypes as _reax_dn_ctypes
import os as _reax_dn_os

_REAX_DN = {"on": None, "var": 0, "lib": None}


def _reax_dn_lib():
    st = _REAX_DN
    if st["lib"] is None:
        c = _reax_dn_ctypes
        lib = (c.PyDLL if _reax_dn_os.environ.get("REAX_PYDLL", "0") == "1" else c.CDLL)(_reax_dn_os.environ.get("REAX_MOE_DN_LIB", "/opt/engv2/libreax_moedn.so"))
        lib.reax_moedn_mm_v.restype = c.c_int
        lib.reax_moedn_mm_v.argtypes = [c.c_int] + [c.c_void_p] * 9 + [c.c_int] * 3 + [c.c_void_p]
        lib.reax_moedn_variants.restype = c.c_int
        lib.reax_moedn_variant_name.restype = c.c_char_p
        lib.reax_moedn_variant_name.argtypes = [c.c_int]
        st["lib"] = lib
    return st["lib"]


def _reax_dn_variants():
    lib = _reax_dn_lib()
    return [lib.reax_moedn_variant_name(i).decode() for i in range(lib.reax_moedn_variants())]


def _reax_dn_raw(out, a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets, var=None):
    rc = _reax_dn_lib().reax_moedn_mm_v(
        _REAX_DN["var"] if var is None else var,
        out.data_ptr(), a.data_ptr(), b.data_ptr(), a_sf.data_ptr(), b_sf.data_ptr(), alphas.data_ptr(),
        problem_sizes.data_ptr(), expert_offsets.data_ptr(), sf_offsets.data_ptr(), b.shape[0], b.shape[1],
        b.shape[2] * 2, torch.cuda.current_stream().cuda_stream)
    if rc != 0:
        raise RuntimeError(f"reax_moedn_mm: error {rc}")


def _reax_moe_dn(out, a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets):
    """ops.cutlass_fp4_moe_mm for the down projection; moe_dn.cu when the start-up check said so."""
    st = _REAX_DN
    if st["on"] is None and not torch.cuda.is_current_stream_capturing():
        st["on"] = False
        path = _reax_dn_os.environ.get("REAX_MOE_DN_FILE", "")
        if _reax_dn_os.environ.get("REAX_MOE_DN", "1") == "1" and path and _reax_dn_os.path.exists(path):
            flags = open(path).read().split()
            st["on"] = flags[:1] == ["1"]
            st["var"] = int(flags[1]) if len(flags) > 1 else 0
        print(f"reax moe_dn: {'on' if st['on'] else 'off'}, variant {st['var']} (check file {path or '-'})", flush=True)
    # up to 1280 tokens (10240 rows): a c16 step is <= ~1014, an image step ~400; at m 1760 the kernel is +4.8 % on the
    # PRO 6000 (-3 % on the 5090: more rows per expert -> more 64-row chunks, the weight stream no longer dominates)
    if (st["on"] and a.shape[0] <= 10240 and b.dim() == 3 and b.shape[1:] == (2816, 352) and out.dtype == torch.bfloat16
            and a.is_contiguous() and b.is_contiguous() and a_sf.is_contiguous() and b_sf.is_contiguous()
            and out.is_contiguous() and alphas.dtype == torch.float32 and problem_sizes.shape[0] == b.shape[0]):
        return _reax_dn_raw(out, a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets)
    return ops.cutlass_fp4_moe_mm(out, a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets)


def _reax_dn_case(m, g, kind, n=2816, k=704, e=128, topk=8):
    """Inputs of the down GEMM as run_cutlass_moe_fp4 builds them: m tokens routed top-k over e experts (as moe_tile)."""
    if kind == 0:
        topk_ids = torch.stack([torch.randperm(e, device="cuda", generator=g)[:topk] for _ in range(m)]).int()
    else:  # skewed: a few hot experts
        w = torch.ones(e, device="cuda")
        w[:8] = 40.0 if kind == 1 else 400.0
        topk_ids = torch.stack([torch.multinomial(w, topk, replacement=False, generator=g) for _ in range(m)]).int()
    eo = torch.empty(e + 1, dtype=torch.int32, device="cuda")
    bo = torch.empty(e + 1, dtype=torch.int32, device="cuda")
    p1 = torch.empty((e, 3), dtype=torch.int32, device="cuda")
    p2 = torch.empty((e, 3), dtype=torch.int32, device="cuda")
    am = torch.empty(m * topk, dtype=torch.int32, device="cuda")
    cm_ = torch.empty(m * topk, dtype=torch.int32, device="cuda")
    ops.get_cutlass_moe_mm_data(topk_ids, eo, p1, p2, am, cm_, e, k, n, bo, is_gated=True)
    a = torch.randint(0, 256, (m * topk, k // 2), dtype=torch.uint8, device="cuda", generator=g)
    b = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, device="cuda", generator=g)
    srows = int(bo[-1].item()) + 128
    # E4M3 scales without NaN (0x7f): positive, wide range
    a_sf = torch.randint(0x18, 0x50, (srows, k // 16), dtype=torch.uint8, device="cuda", generator=g)
    b_sf = torch.randint(0x18, 0x50, (e, n, k // 16), dtype=torch.uint8, device="cuda", generator=g)
    alphas = torch.exp2(torch.randint(-14, -4, (e,), device="cuda", generator=g).float()) * (
        1.0 + torch.rand(e, device="cuda", generator=g))
    return (a, b, a_sf.view(torch.float8_e4m3fn), b_sf.view(torch.float8_e4m3fn), alphas.float(), p2, eo[:-1], bo[:-1])


def _reax_dn_check(small=False, variants=None):
    """Byte-for-byte vs ops.cutlass_fp4_moe_mm (both outputs pre-filled with the same pattern) at many m and routings,
    for every variant; returns the exact ones."""
    g = torch.Generator(device="cuda").manual_seed(4242)
    ms = (1, 3, 16, 200) if small else (1, 2, 3, 7, 8, 16, 31, 64, 100, 128, 200, 333, 512, 887, 1000, 1024, 1500, 2048)
    names = _reax_dn_variants()
    exact = set(range(len(names)) if variants is None else variants)
    for m in ms:
        for kind in (0, 1, 2):
            a, b, a_sf, b_sf, alphas, ps, eo, bo = _reax_dn_case(m, g, kind)
            ref = torch.full((m * 8, 2816), -7.0, dtype=torch.bfloat16, device="cuda")
            ops.cutlass_fp4_moe_mm(ref, a, b, a_sf, b_sf, alphas, ps, eo, bo)
            for v in sorted(exact):
                out = torch.full_like(ref, -7.0)
                _reax_dn_raw(out, a, b, a_sf, b_sf, alphas, ps, eo, bo, var=v)
                torch.cuda.synchronize()
                bad = out.view(torch.int16) != ref.view(torch.int16)
                if bool(bad.any()):
                    exact.discard(v)
                    print(f"reax moe_dn: MISMATCH {names[v]} m {m} kind {kind}: {int(bad.sum())} of {out.numel()} "
                          f"values, first at {tuple(int(x) for x in bad.nonzero()[0])}", flush=True)
    return sorted(exact)


def _reax_dn_time(mine, m=887, var=None, rep=300):
    from triton.testing import do_bench
    g = torch.Generator(device="cuda").manual_seed(7)
    a, b, a_sf, b_sf, alphas, ps, eo, bo = _reax_dn_case(m, g, 0)
    out = torch.empty((m * 8, 2816), dtype=torch.bfloat16, device="cuda")
    if mine:
        return do_bench(lambda: _reax_dn_raw(out, a, b, a_sf, b_sf, alphas, ps, eo, bo, var=var), warmup=25, rep=rep)
    return do_bench(lambda: ops.cutlass_fp4_moe_mm(out, a, b, a_sf, b_sf, alphas, ps, eo, bo), warmup=25, rep=rep)


_REAX_DN_TUNE_M = (440, 887, 1000)  # a step's token counts (image steps ~440, c16 text steps 645-1014)


def _reax_dn_tune(exact, verbose=False):
    """Time every exact variant at the tuning m (best of 2); returns (best variant, its times, vLLM's times)."""
    names = _reax_dn_variants()
    t_ref = [min(_reax_dn_time(False, m, rep=100) for _ in range(2)) for m in _REAX_DN_TUNE_M]
    best, best_t = None, None
    for v in exact:
        tv = [min(_reax_dn_time(True, m, v, rep=100) for _ in range(2)) for m in _REAX_DN_TUNE_M]
        if verbose:
            print(f"  {names[v]:14s} " + "  ".join(f"m {m} {t * 1e3:6.1f} us ({100 * (t / r - 1):+5.1f} %)"
                                                 for m, t, r in zip(_REAX_DN_TUNE_M, tv, t_ref)), flush=True)
        if best_t is None or sum(tv) < sum(best_t):
            best, best_t = v, tv
    return best, best_t, t_ref
