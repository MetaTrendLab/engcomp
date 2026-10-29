import sys

ATT_OLD = '''            attn_output = self.attn(q, k, v)

        output, _ = self.o_proj(attn_output)

        return output
'''
ATT_NEW = '''            attn_output = self.attn(q, k, v)

        if _REAX_LAST_ROWS and getattr(self, "_reax_last", False):
            _rows = torch.ops.reax.last_rows(attn_output)
            _o, _ = self.o_proj(attn_output.index_select(0, _rows))
            return _o.new_zeros((attn_output.shape[0], _o.shape[1])).index_copy(0, _rows, _o)
        output, _ = self.o_proj(attn_output)

        return output
'''
MLP_OLD = '''        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
'''
MLP_NEW = '''        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        if _REAX_LAST_ROWS and getattr(self, "_reax_last", False):
            _rows = torch.ops.reax.last_rows(hidden_states)
            _m = self.mlp(hidden_states.index_select(0, _rows))
            hidden_states = _m.new_zeros(hidden_states.shape).index_copy(0, _rows, _m)
        else:
            hidden_states = self.mlp(hidden_states)
'''
MOE_OLD = '''            hidden_states_2 = self.moe(hidden_states_2, router_logits)
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)
'''
MOE_NEW = '''            if _REAX_LAST_ROWS and getattr(self, "_reax_last", False):
                _rows = torch.ops.reax.last_rows(hidden_states_2)
                _e = self.moe(hidden_states_2.index_select(0, _rows), router_logits.index_select(0, _rows))
                hidden_states_2 = _e.new_zeros(hidden_states_2.shape).index_copy(0, _rows, _e)
            else:
                hidden_states_2 = self.moe(hidden_states_2, router_logits)
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)
'''
INIT_OLD = '''        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
'''
INIT_NEW = INIT_OLD + '''        if _REAX_LAST_ROWS:
            _reax_last_rows_init(self, vllm_config)
'''
ADD = '''

# ---------------------------------------------------------------- engv2 Blink: last layer on the sampled rows (last_rows.py)
# unset = auto: Blink's text model only (hidden 2816, 128 experts: what was measured); "1" forces it on other Gemma-4 models
_REAX_LAST_ROWS_ENV = _os.environ.get("REAX_LAST_ROWS", "auto")
_REAX_LAST_ROWS = _REAX_LAST_ROWS_ENV in ("1", "auto")
_REAX_ROWS = {"buf": None, "S": 0}


def _reax_last_rows_init(model, vllm_config):
    global _REAX_LAST_ROWS
    layers = [layer for layer in model.layers if isinstance(layer, Gemma4DecoderLayer)]
    why = None
    cfg = model.config
    if _REAX_LAST_ROWS_ENV == "auto" and (cfg.hidden_size != 2816 or getattr(cfg, "num_experts", 0) != 128):
        why = f"auto: not Blink (hidden {cfg.hidden_size}, experts {getattr(cfg, 'num_experts', None)})"
    elif model.end_layer != model.config.num_hidden_layers or not layers:
        why = "pipeline-parallel rank without the last layer"
    elif vllm_config.speculative_config is not None:
        why = "speculative decoding (several sampled rows per request)"
    elif vllm_config.cache_config.kv_sharing_fast_prefill:
        why = "KV-sharing fast prefill"
    if why is not None:
        logger.warning("reax last_rows: off (%s)", why)
        _REAX_LAST_ROWS = False
        return
    S = vllm_config.scheduler_config.max_num_seqs
    _REAX_ROWS["S"] = S
    _REAX_ROWS["buf"] = torch.zeros(S, dtype=torch.int64, device=torch.device("cuda", torch.cuda.current_device()))
    layers[-1]._reax_last = True
    layers[-1].self_attn._reax_last = True
    logger.info("reax last_rows: layer %d tail (o_proj, MLP, MoE) on <= %d sampled rows", len(model.layers) - 1, S)


def reax_rows_set(logits_indices):
    buf = _REAX_ROWS["buf"]
    if buf is None:
        return
    n = logits_indices.shape[0]
    if n > buf.shape[0] or n == 0:
        raise RuntimeError(f"reax last_rows: {n} sampled rows, buffer {buf.shape[0]}")
    buf[:n].copy_(logits_indices, non_blocking=True)
    if n < buf.shape[0]:
        buf[n:] = logits_indices[-1]


@torch.library.custom_op("reax::last_rows", mutates_args=())
def _reax_last_rows(x: torch.Tensor) -> torch.Tensor:
    # a copy of the runner-filled buffer (graph replays read its current content), clamped to this call's rows
    return _REAX_ROWS["buf"].clamp(0, max(x.shape[0] - 1, 0))


@_reax_last_rows.register_fake
def _reax_last_rows_fake(x):
    return torch.empty(_REAX_ROWS["S"], dtype=torch.int64, device=x.device)
'''
RUN_OLD = '''        # Run model.
        if batch_desc.cg_mode == CUDAGraphMode.FULL:
'''
RUN_NEW = '''        if _REAX_LAST_ROWS_RUN:
            _reax_g4 = _reax_lr_sys.modules.get("vllm.model_executor.models.gemma4")
            if _reax_g4 is not None:
                _reax_g4.reax_rows_set(input_batch.logits_indices)
''' + RUN_OLD
RUN_ADD = '''

import os as _reax_lr_os  # noqa: E402
import sys as _reax_lr_sys  # noqa: E402

_REAX_LAST_ROWS_RUN = _reax_lr_os.environ.get("REAX_LAST_ROWS", "auto") in ("1", "auto")  # no-op unless gemma4 armed it
'''


def patch(path, pairs, add):
    src = open(path).read()
    for old, new in pairs:
        assert src.count(old) == 1, f"{path}: anchor found {src.count(old)} times: {old[:70]!r}"
        src = src.replace(old, new)
    open(path, "w").write(src + add)


if __name__ == "__main__":
    patch(sys.argv[1], [(ATT_OLD, ATT_NEW), (MLP_OLD, MLP_NEW), (MOE_OLD, MOE_NEW), (INIT_OLD, INIT_NEW)], ADD)
    patch(sys.argv[2], [(RUN_OLD, RUN_NEW)], RUN_ADD)
    print("last_rows: installed")
