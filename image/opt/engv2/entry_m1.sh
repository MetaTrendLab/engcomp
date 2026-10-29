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
python3 /opt/engv2/moe_gquant.py derive "$REAX_MOE_GQUANT_FILE" || rm -f "$REAX_MOE_GQUANT_FILE"
python3 /opt/engv2/moe_unperm.py derive "$REAX_MOE_UNPERM_FILE" || rm -f "$REAX_MOE_UNPERM_FILE"
python3 /opt/engv2/moe_gact.py derive "$REAX_MOE_GACT_FILE" || rm -f "$REAX_MOE_GACT_FILE"
python3 /opt/engv2/moe_dn.py derive "$REAX_MOE_DN_FILE" || rm -f "$REAX_MOE_DN_FILE"
python3 /opt/engv2/lt_small4.py derive "$REAX_LT_SMALL_FILE" || rm -f "$REAX_LT_SMALL_FILE"
python3 /opt/engv2/lt_vis.py derive "$REAX_LT_VIS_FILE" || rm -f "$REAX_LT_VIS_FILE"
exec /opt/engv2/entry_dual.sh "$@"
