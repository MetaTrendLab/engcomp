#!/bin/bash
set -u
B_ENV="REAX_STREAM REAX_HALF_CAP REAX_HALF_SPACE_MS REAX_ATT_TILES REAX_VLLM_EXTRA REAX_PINS_FILE REAX_HIDDEN_PINS"
case "${REAX_MODEL:-}" in
  mammoth|mammoth-1|reax1|reax-1)
    for v in $B_ENV; do unset $v; done
    case "$REAX_MODEL" in mammoth*) export Q35_INBA=${Q35_INBA_MAMMOTH:-ref} ;; *) export Q35_INBA=${Q35_INBA_REAX1:-ref} ;; esac
    export Q35_READOUT=${Q35_READOUT_DM:-ref} REAX_LANES=${REAX_LANES_DM:-1} Q35_HOLD_MS=${Q35_HOLD_MS:-15} Q35_IDX=1 Q35_IDX_MIN_T=1024 \
           Q35_WARM_MAXT=${Q35_WARM_MAXT:-6144} Q35_WARM_STEPS=${Q35_WARM_STEPS:-60} CIE_LAZY_LOAD=1 REAX_MAX_WAIT_MS=4 \
           REAX_PROFILE=reference PYTHONPATH=/opt/q35 PATH=/opt/venv/bin:$PATH
    export Q35_G6_TABLE=${Q35_G6_TABLE-/opt/q35/engine/kern/g6_table_g7gu_pro6000.json} Q35_G6GU=${Q35_G6GU-table}
    export Q35_COPIES=${Q35_COPIES-1} Q35_ATT1=${Q35_ATT1-1} Q35_FLA_EMPTY=${Q35_FLA_EMPTY-1}  # j_flae
    export Q35_GC_FREEZE=${Q35_GC_FREEZE-1}
    export Q35_GDN_HO=${Q35_GDN_HO-1}
    export Q35_NOSYNC=${Q35_NOSYNC-1} Q35_GDN_PRE=${Q35_GDN_PRE-1} Q35_CONV1=${Q35_CONV1-2} Q35_GDN_GATES=${Q35_GDN_GATES-1} \
           Q35_GDN1=${Q35_GDN1-1} Q35_GATE3=${Q35_GATE3-1} Q35_PARBA=${Q35_PARBA-0}
    export Q35_GDN_HO_BV=${Q35_GDN_HO_BV-128} Q35_GDN_HO_WARPS=${Q35_GDN_HO_WARPS-8} Q35_GDN_HO_STAGES=${Q35_GDN_HO_STAGES-2}
    export Q35_HOLD_EARLY=${Q35_HOLD_EARLY-1}
    export Q35_ENC_THREAD=${Q35_ENC_THREAD-2}
    case "$REAX_MODEL" in reax1|reax-1) export PPLX_SPLIT_CONV_MIN_T=${PPLX_SPLIT_CONV_MIN_T-1} ;; esac  # j_splitconv
    cd /opt/q35 && exec /opt/venv/bin/python -m engine.serve "$@" ;;
  jeffan|jeffan-31b-1710)
    for v in $B_ENV; do unset $v; done
    export REAX_PRESET=${REAX_PRESET_JEFFAN:-fp8-kv-colo}
    export REAX_JEFFAN_F8=${REAX_JEFFAN_F8-1}  # f8g6 g9 FP8 linears, byte-identical to cutlass_scaled_mm (engv2/jeffan/f8)
    exec /opt/reax/entrypoint.sh "$@" ;;
  *)
    export REAX_PYDLL=${REAX_PYDLL-1}
    export REAX_BF16G7=${REAX_BF16G7_DM-1}  # b7 dense tactics (j_b7ab: text +2.4 %); also the m-line default since 7d56e736
    export REAX_BF16SK=${REAX_BF16SK-1}
    export REAX_MOE_GU=${REAX_MOE_GU-1}
    export REAX_SAMP_FAST=${REAX_SAMP_FAST-1} REAX_MOE_DATA=${REAX_MOE_DATA-1} REAX_RGEMM=${REAX_RGEMM-1} REAX_GQ1=${REAX_GQ1-1} \
           REAX_MOE_GU_GATHER=${REAX_MOE_GU_GATHER-1}
    export REAX_INPROC=${REAX_INPROC-1}
    export REAX_NOSTATS=${REAX_NOSTATS-1} REAX_NODETOK=${REAX_NODETOK-1} REAX_ENC_INLINE=${REAX_ENC_INLINE-1} \
           REAX_ONE_REQ=${REAX_ONE_REQ-1} REAX_ENC_FAST=${REAX_ENC_FAST-1}
    export REAX_STEP_NEW_MAX_IMAGE=${REAX_STEP_NEW_MAX_IMAGE-4}
    export REAX_VIS_ATTN_LIB=${REAX_VIS_ATTN_LIB-/opt/engv2/libreax_vattn2.so} REAX_VIS_PACK_SYNC1=${REAX_VIS_PACK_SYNC1-1} \
           REAX_VIS_POOL1=${REAX_VIS_POOL1-1}
    export REAX_CORE_POLL_US_IMAGE=${REAX_CORE_POLL_US_IMAGE-0} REAX_CORE_GATHER_MAX_MS_IMAGE=${REAX_CORE_GATHER_MAX_MS_IMAGE-0} \
           REAX_CORE_POLL_GATHER_MAX_MS_IMAGE=${REAX_CORE_POLL_GATHER_MAX_MS_IMAGE-0}
    exec /opt/engv2/entry_m1.sh "$@" ;;
esac
