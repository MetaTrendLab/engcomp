#!/bin/bash
# engv2 dm-all entrypoint: one image, the model from $REAX_MODEL (v2 slots mammoth / blink + vision / reax1 / jeffan).
#   mammoth-1, reax-1 : the q35 engine (engv2/q35-engine) on the reference's own stack (/opt/venv: torch 2.14, fla 0.5.2),
#                       in_ba / readout in the reference's own class per step ("ref" mode, validator concurrency 8), HOLD batching
#   blink (text+image): the 5090's m-line (tools/engv2/blink, entry_m1.sh: derivations, then text compiled + images eager)
#   jeffan            : the reference recipe without VLLM_BATCH_INVARIANT (fp8-kv-colo); the Blink-only knobs are unset
set -u
B_ENV="REAX_STREAM REAX_HALF_CAP REAX_HALF_SPACE_MS REAX_ATT_TILES REAX_VLLM_EXTRA REAX_PINS_FILE REAX_HIDDEN_PINS"
case "${REAX_MODEL:-}" in
  mammoth|mammoth-1|reax1|reax-1)
    for v in $B_ENV; do unset $v; done
    # in_ba / readout at the step's own M, as the reference computes them ("ref" mode): at the validator's concurrency 8 the
    # reference runs the client's 8-request waves as one step and so does HOLD, so both share the cuBLAS class item by item;
    # at c1 (latency items, anti-cache probes) the class is the reference's c1 class. Paired c8 sessions (seed 808): mammoth
    # 512/512 answers byte-identical to the reference, REAX 1 435/512, c1 exact on both; the price is the determinism re-run
    # (c1 class vs c8 class): 0.459 / 0.292 against the 1.1 / 1.15 gates. The c8 pins (pad1536 / pad1280 + readout pad8) gave
    # det 0 but c1 mean 0.102 / 0.205 (REAX 1 gate 0.19); the c16 pins (pad3040 / pad2448 / pad16) matched the c8 class for 0 %.
    case "$REAX_MODEL" in mammoth*) export Q35_INBA=${Q35_INBA_MAMMOTH:-ref} ;; *) export Q35_INBA=${Q35_INBA_REAX1:-ref} ;; esac
    export Q35_READOUT=${Q35_READOUT_DM:-ref} REAX_LANES=${REAX_LANES_DM:-1} Q35_HOLD_MS=${Q35_HOLD_MS:-15} Q35_IDX=1 Q35_IDX_MIN_T=1024 \
           Q35_WARM_MAXT=${Q35_WARM_MAXT:-6144} Q35_WARM_STEPS=${Q35_WARM_STEPS:-60} CIE_LAZY_LOAD=1 REAX_MAX_WAIT_MS=4 \
           REAX_PROFILE=reference PYTHONPATH=/opt/q35 PATH=/opt/venv/bin:$PATH
    # f8g6 per-(N, K, M) FP8 GEMM tactics, byte-identical to the reference's GEMMs (kern/f8g6.cu, table built on the PRO 6000)
    # + the gate|up GEMM with the _smqs epilogue (g7gu, interleaved weight rows; Q35_G6GU= turns it off)
    export Q35_G6_TABLE=${Q35_G6_TABLE-/opt/q35/engine/kern/g6_table_g7gu_pro6000.json} Q35_G6GU=${Q35_G6GU-table}
    # fewer copy kernels in shared-state steps (5090 R4: s_par views, in_ba last chunk without pad + cat, two-pointer
    # DeltaNet out), byte-identical; +1.5 % mammoth offline
    export Q35_COPIES=${Q35_COPIES-1} Q35_ATT1=${Q35_ATT1-1} Q35_FLA_EMPTY=${Q35_FLA_EMPTY-1}  # j_flae
    # gc.collect + gc.freeze after warm-up (as vLLM): a gen-2 collection walked ~730k long-lived objects in 650-700 ms,
    # one per ~half of the REAX 1 sessions (run 70 -> 58 dps); frozen, serving collections stay < 20 ms
    export Q35_GC_FREEZE=${Q35_GC_FREEZE-1}
    # fla fwd_h + fwd_o as one recurrence kernel, byte-equal to fla's BK 64 fwd_o mode (the reference's autotune pick):
    # mammoth +1.4 %, identical answers 512 / 512 on both models (gdn_ho.py, engv2/gdn_ho_test.py / gdn_ho_epi.py)
    export Q35_GDN_HO=${Q35_GDN_HO-1}
    # fewer kernels per step (5090, each byte-checked; dm:16t paired runsc c8 sessions: answers identical 512/512 on both
    # models, mammoth +3.9 % / REAX 1 +2.4 % dps, c1 latency 83 -> 61 / 62 -> 39 ms; lab B=8 GPU ops 2242 -> 1137 per step):
    # host-copy metadata (NOSYNC), DeltaNet prologue in two kernels with the gates (GDN_PRE, GDN_GATES) over prefix + suffix
    # (GDN1), q|k|v conv for prefix + suffix in one launch (CONV1=2), attention output gate read in place (GATE3)
    export Q35_NOSYNC=${Q35_NOSYNC-1} Q35_GDN_PRE=${Q35_GDN_PRE-1} Q35_CONV1=${Q35_CONV1-2} Q35_GDN_GATES=${Q35_GDN_GATES-1} \
           Q35_GDN1=${Q35_GDN1-1} Q35_GATE3=${Q35_GATE3-1} Q35_PARBA=${Q35_PARBA-0}
    # gdn_ho tiles for 188 SMs (gdn_ho_bench.py on this card: BV only tiles V, every grid point byte-equal): c8 shared call
    # 210.0 -> 185.6 us, c16 438.8 -> 387.0 us (BV 128, 8 warps, 2 stages); PARBA / FA1 / GDN2 measured neutral here (j_parba)
    export Q35_GDN_HO_BV=${Q35_GDN_HO_BV-128} Q35_GDN_HO_WARPS=${Q35_GDN_HO_WARPS-8} Q35_GDN_HO_STAGES=${Q35_GDN_HO_STAGES-2}
    # no REAX_MAX_WAIT_MS deadline once a steady c8 wave is complete (5090 serve.Scheduler._full; burst starts, partial waves
    # and c1 keep max_wait + HOLD): paired runsc c8, answers identical 512/512 on both models, the reference-identical count
    # unchanged (same waves); mammoth +2-5 %, REAX 1 neutral
    export Q35_HOLD_EARLY=${Q35_HOLD_EARLY-1}
    # encode threads that submit straight to the scheduler queue (no executor future / loop round trip before submit): paired
    # runsc c8, answers identical: REAX 1 2 threads +1.3 %, mammoth 4 threads +1.5 % (2 = 4 within 1 %; j_reqlog2 / j_encthr)
    export Q35_ENC_THREAD=${Q35_ENC_THREAD-2}
    case "$REAX_MODEL" in reax1|reax-1) export PPLX_SPLIT_CONV_MIN_T=${PPLX_SPLIT_CONV_MIN_T-1} ;; esac  # j_splitconv
    cd /opt/q35 && exec /opt/venv/bin/python -m engine.serve "$@" ;;
  jeffan|jeffan-31b-1710)
    for v in $B_ENV; do unset $v; done
    export REAX_PRESET=${REAX_PRESET_JEFFAN:-fp8-kv-colo}
    export REAX_JEFFAN_F8=${REAX_JEFFAN_F8-1}  # f8g6 g9 FP8 linears, byte-identical to cutlass_scaled_mm (engv2/jeffan/f8)
    exec /opt/reax/entrypoint.sh "$@" ;;
  *)
    # native Blink kernels (moe_*, vis_pack, bf16g) launched through ctypes.PyDLL: vision +0.8 % (3/4), bit-exact on the PRO 6000
    export REAX_PYDLL=${REAX_PYDLL-1}
    export REAX_BF16G7=${REAX_BF16G7_DM-1}  # b7 dense tactics (j_b7ab: text +2.4 %); also the m-line default since 7d56e736
    # + b7s (serial stream-K tilings of bf16g7, byte-equal) in the lt_small4 / lt_vis derive, one flag for both instances (one
    # derive table): paired runsc text +0.7 %, vision +4.4 %, answers identical
    export REAX_BF16SK=${REAX_BF16SK-1}
    # MoE gate|up NVFP4 grouped GEMM with gelu*up + FP4 quant in the epilogue (5090 moe_gu, byte-identical; the start-up derive
    # keeps it only if it beats cutlass + gact by >= 2 % on real routing): PRO 6000 derive -21 % at m 887; paired runsc text
    # +3.6 %, vision +7.4 %, answers identical
    export REAX_MOE_GU=${REAX_MOE_GU-1}
    # MoE / sampler host-side cuts (5090, each byte-checked in-engine): sampler tail on many CTAs (samp_fast), one-CTA
    # get_cutlass_moe_mm_data (moe_data), router GEMM through rgemm where byte-equal (rgemm), NVFP4 MoE input quantised once
    # per token (gquant1), gate|up reading A through a_map (moe_gu gather; the derive keeps it only if > 1 % faster): paired
    # runsc text on dm:20t +2.0 / +0.5 / +0.5 / +3.0 % = +6.0 %, answers identical in all arms (j_tail)
    export REAX_SAMP_FAST=${REAX_SAMP_FAST-1} REAX_MOE_DATA=${REAX_MOE_DATA-1} REAX_RGEMM=${REAX_RGEMM-1} REAX_GQ1=${REAX_GQ1-1} \
           REAX_MOE_GU_GATHER=${REAX_MOE_GU_GATHER-1}
    # the eager image instance follows them too (entry_dual *_IMAGE unset): runsc vision ABAB, one reference, five on vs off
    # = 37.45 vs 36.86 dps (+1.6 %), bit-exact (j_visab); the earlier unpaired -5 % was per-start spread
    # vision encoder (m30, 5090, bit-exact): vis_attn2 variants (derive picks w4_b1_o2 / o7, -7 % attention on this card),
    # one device-to-host read per encoder call (vis_pack1) and the pooler counts from it (vis_pool1): 5090 +3.7 % vision;
    # read by the image path only (the text instance never loads the lib)
    # the Blink text vLLM instance inside the REAX process (5090 inproc.py: vLLM's own completion handler called directly, no
    # loopback HTTP / JSON per half wave; the image instance stays on HTTP :8001): paired runsc text +8.4 % on top of the frozen
    # lt tables (359.0 vs 331.5 dps), cpu 5.8 -> 3.9 ms/dec, c1 18 -> 16 ms, answers identical (j_ltab)
    export REAX_INPROC=${REAX_INPROC-1}
    export REAX_VIS_ATTN_LIB=${REAX_VIS_ATTN_LIB-/opt/engv2/libreax_vattn2.so} REAX_VIS_PACK_SYNC1=${REAX_VIS_PACK_SYNC1-1} \
           REAX_VIS_POOL1=${REAX_VIS_POOL1-1}
    # image instance only (entry_dual *_IMAGE): no EngineCore gap waits (timed futex waits trap to the gVisor sentry):
    # runsc vision 2.254 -> 2.315 on top of the stack above, bit-exact; the text instance keeps its gather / poll loop
    export REAX_CORE_POLL_US_IMAGE=${REAX_CORE_POLL_US_IMAGE-0} REAX_CORE_GATHER_MAX_MS_IMAGE=${REAX_CORE_GATHER_MAX_MS_IMAGE-0} \
           REAX_CORE_POLL_GATHER_MAX_MS_IMAGE=${REAX_CORE_POLL_GATHER_MAX_MS_IMAGE-0}
    exec /opt/engv2/entry_m1.sh "$@" ;;
esac
