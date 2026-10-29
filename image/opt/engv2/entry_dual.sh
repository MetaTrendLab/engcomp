#!/bin/bash
set -u
python3 -I /opt/engv2/engv2_pins.py
DIR=${REAX_MODEL_DIR:-/model}
VPORT=${REAX_VLLM_PORT:-8000}
IPORT=${REAX_VLLM_IMAGE_PORT:-8001}
export REAX_PRESET=${REAX_PRESET:-ref-prod}
export REAX_VISION_ENCODER_BUDGET=${REAX_VISION_ENCODER_BUDGET:-total10}
wait_health() {  # port pid
  for _ in $(seq 1 450); do
    curl -sf -m 3 "http://127.0.0.1:$1/health" >/dev/null && return 0
    kill -0 "$2" 2>/dev/null || { echo "reax: vLLM on :$1 died during startup"; return 1; }
    sleep 2
  done
  echo "reax: vLLM on :$1 not healthy"; return 1
}
REAX_BLINK_OVERLAY=${REAX_BLINK_OVERLAY_TEXT:-1} python3 -m engine.backends.vllm_readout blink ref-prod "$DIR" --port "$VPORT" --exec ${REAX_VLLM_EXTRA:-} &
P1=$!
wait_health "$VPORT" $P1 || exit 3
python3 -I /opt/engv2/engv2_pins.py check
PIDS="$P1"
if [ "${REAX_DUAL:-1}" = 1 ]; then
  REAX_STEP_NEW_MAX=${REAX_STEP_NEW_MAX_IMAGE:-0} REAX_LM_FUSED=${REAX_LM_FUSED_IMAGE:-1} REAX_FP8_DENSE=0 \
    REAX_CORE_POLL_US=${REAX_CORE_POLL_US_IMAGE-${REAX_CORE_POLL_US-200}} \
    REAX_CORE_GATHER_MAX_MS=${REAX_CORE_GATHER_MAX_MS_IMAGE-${REAX_CORE_GATHER_MAX_MS-4}} \
    REAX_CORE_POLL_GATHER_MAX_MS=${REAX_CORE_POLL_GATHER_MAX_MS_IMAGE-${REAX_CORE_POLL_GATHER_MAX_MS-4}} \
    REAX_HEAD_SUB=${REAX_HEAD_SUB_IMAGE:-0} REAX_LAST_ROWS=${REAX_LAST_ROWS_IMAGE:-1} \
    REAX_BLINK_OVERLAY=${REAX_BLINK_OVERLAY_IMAGE:-${REAX_LAST_ROWS_IMAGE:-1}} \
    python3 -m engine.backends.vllm_readout blink ref-prod "$DIR" \
    --port "$IPORT" --exec ${REAX_VLLM_EXTRA_IMAGE:---enforce-eager --gpu-memory-utilization 0.22} &
  P2=$!
  wait_health "$IPORT" $P2 || exit 3
  export REAX_VLLM_IMAGE_URL="http://127.0.0.1:$IPORT"
  PIDS="$P1 $P2"
fi
export REAX_VLLM_BASE_URL="http://127.0.0.1:$VPORT"
python3 -m engine.serve --model blink-v0.3-26b-a4b --model-dir "$DIR" --host "${REAX_HOST:-0.0.0.0}" \
  --port "${REAX_PORT:-8741}" --opt "preset=${REAX_PRESET}" &
SRV=$!
trap 'kill $PIDS $SRV 2>/dev/null' TERM INT
wait -n $PIDS $SRV
echo "reax: a child process exited; shutting down"
kill $PIDS $SRV 2>/dev/null
exit 1
