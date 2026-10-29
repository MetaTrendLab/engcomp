#!/bin/bash
if [ "${REAX_BF16G7:-0}" = 1 ]; then
  export REAX_BF16G7_CUBIN=${REAX_BF16G7_CUBIN:-/opt/engv2/bf16g7.cubin}
  export REAX_BF16G7_ONLY=${REAX_BF16G7_ONLY-$(cat /opt/engv2/bf16g7_allow.txt)}
fi
if [ "${REAX_BF16SK:-0}" = 1 ] && [ -f /opt/engv2/b7sk.cubin ]; then
  export REAX_BF16SK_CUBIN=${REAX_BF16SK_CUBIN:-/opt/engv2/b7sk.cubin}
fi
python3 /opt/engv2/vis_mlp.py derive "$REAX_VIS_MLP_FILE" || rm -f "$REAX_VIS_MLP_FILE"
python3 /opt/engv2/vis_rope.py derive "$REAX_VIS_ROPE_FILE" || rm -f "$REAX_VIS_ROPE_FILE"
python3 /opt/engv2/vis_norm.py derive "$REAX_VIS_NORM_FILE" || rm -f "$REAX_VIS_NORM_FILE"
python3 /opt/engv2/vis_pack.py derive "$REAX_VIS_PACK_FILE" || rm -f "$REAX_VIS_PACK_FILE"
GQ1_ANY=$([ "${REAX_GQ1:-0}" = 1 ] || [ "${REAX_GQ1_IMAGE:-0}" = 1 ] && echo 1 || echo 0)
GATHER_ANY=$([ "${REAX_MOE_GU_GATHER:-0}" = 1 ] || [ "${REAX_MOE_GU_GATHER_IMAGE:-0}" = 1 ] && echo 1 || echo 0)
REAX_GQ1=$GQ1_ANY python3 /opt/engv2/moe_gquant.py derive "$REAX_MOE_GQUANT_FILE" || rm -f "$REAX_MOE_GQUANT_FILE"
python3 /opt/engv2/moe_unperm.py derive "$REAX_MOE_UNPERM_FILE" || rm -f "$REAX_MOE_UNPERM_FILE"
python3 /opt/engv2/moe_gact.py derive "$REAX_MOE_GACT_FILE" || rm -f "$REAX_MOE_GACT_FILE"
python3 /opt/engv2/moe_dn.py derive "$REAX_MOE_DN_FILE" || rm -f "$REAX_MOE_DN_FILE"
if [ -n "${REAX_MOE_DN_SPLIT_IMAGE:-}" ] || [ -n "${REAX_MOE_DN_ONLY_IMAGE:-}" ]; then
  export REAX_MOE_DN_FILE_IMAGE=${REAX_MOE_DN_FILE_IMAGE:-/tmp/reax_moe_dn_image.txt}
  REAX_MOE_DN_SPLIT=${REAX_MOE_DN_SPLIT_IMAGE:-${REAX_MOE_DN_SPLIT:-1}} REAX_MOE_DN_ONLY=${REAX_MOE_DN_ONLY_IMAGE:-${REAX_MOE_DN_ONLY:-}} \
    python3 /opt/engv2/moe_dn.py derive "$REAX_MOE_DN_FILE_IMAGE" || rm -f "$REAX_MOE_DN_FILE_IMAGE"
fi
if [ "${REAX_MOE_GU:-0}" = 1 ]; then
  REAX_MOE_GU_GATHER=$GATHER_ANY python3 /opt/engv2/moe_gu.py derive "$REAX_MOE_GU_FILE" || rm -f "$REAX_MOE_GU_FILE"
fi
if [ -n "${REAX_LT_DERIVE_N:-}" ]; then
  python3 /opt/engv2/lt_frozen.py small derive-n "$REAX_LT_DERIVE_DIR" "$REAX_LT_DERIVE_N" &&
    python3 /opt/engv2/lt_frozen.py vis derive-n "$REAX_LT_DERIVE_DIR" "$REAX_LT_DERIVE_N"
  exit $?
fi
if [ -n "${REAX_LT_SMALL_FROZEN:-}" ] && python3 /opt/engv2/lt_frozen.py small check "$REAX_LT_SMALL_FROZEN" "$REAX_LT_SMALL_FILE"; then :
else python3 /opt/engv2/lt_small4.py derive "$REAX_LT_SMALL_FILE" || rm -f "$REAX_LT_SMALL_FILE"; fi
if [ -n "${REAX_LT_VIS_FROZEN:-}" ] && python3 /opt/engv2/lt_frozen.py vis check "$REAX_LT_VIS_FROZEN" "$REAX_LT_VIS_FILE"; then :
else python3 /opt/engv2/lt_vis.py derive "$REAX_LT_VIS_FILE" || rm -f "$REAX_LT_VIS_FILE"; fi
exec /opt/engv2/entry_dual.sh "$@"
