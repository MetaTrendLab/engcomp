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
NOSTATS_TEXT=$([ "${REAX_NOSTATS:-0}" = 1 ] && echo --disable-log-stats)
NOSTATS_IMAGE=$([ "${REAX_NOSTATS_IMAGE:-${REAX_NOSTATS:-0}}" = 1 ] && echo --disable-log-stats)
image_env() {  # exec "$@" in the image instance's environment (one list for start_image and REAX_INPROC_IMAGE)
  REAX_STEP_NEW_MAX=${REAX_STEP_NEW_MAX_IMAGE:-0} REAX_LM_FUSED=${REAX_LM_FUSED_IMAGE:-1} REAX_FP8_DENSE=0 \
    REAX_CORE_POLL_US=${REAX_CORE_POLL_US_IMAGE-${REAX_CORE_POLL_US-200}} \
    REAX_CORE_GATHER_MAX_MS=${REAX_CORE_GATHER_MAX_MS_IMAGE-${REAX_CORE_GATHER_MAX_MS-4}} \
    REAX_CORE_POLL_GATHER_MAX_MS=${REAX_CORE_POLL_GATHER_MAX_MS_IMAGE-${REAX_CORE_POLL_GATHER_MAX_MS-4}} \
    REAX_HEAD_SUB=${REAX_HEAD_SUB_IMAGE:-0} REAX_LAST_ROWS=${REAX_LAST_ROWS_IMAGE:-1} \
    REAX_MOE_DN_FILE=${REAX_MOE_DN_FILE_IMAGE:-${REAX_MOE_DN_FILE:-}} \
    REAX_MOE_GU_GATHER=${REAX_MOE_GU_GATHER_IMAGE-${REAX_MOE_GU_GATHER-}} \
    REAX_SAMP_FAST=${REAX_SAMP_FAST_IMAGE-${REAX_SAMP_FAST-}} REAX_MOE_DATA=${REAX_MOE_DATA_IMAGE-${REAX_MOE_DATA-}} \
    REAX_RGEMM=${REAX_RGEMM_IMAGE-${REAX_RGEMM-}} REAX_GQ1=${REAX_GQ1_IMAGE-${REAX_GQ1-}} \
    REAX_BLINK_OVERLAY=${REAX_BLINK_OVERLAY_IMAGE:-${REAX_LAST_ROWS_IMAGE:-1}} \
    exec "$@"
}
IMAGE_EXTRA="${REAX_VLLM_EXTRA_IMAGE:---enforce-eager --gpu-memory-utilization 0.22} $NOSTATS_IMAGE"
start_image() {  # the image instance (eager, numerics of the vision reference); sets P2 (image_env execs: $! is vLLM)
  image_env python3 -m engine.backends.vllm_readout blink ref-prod "$DIR" --port "$IPORT" --exec $IMAGE_EXTRA &
  P2=$!
  wait_health "$IPORT" $P2
}
if [ "${REAX_INPROC:-0}" = 1 ]; then
  export REAX_INPROC_READY=${REAX_INPROC_READY:-/tmp/reax_inproc_ready} REAX_VLLM_PORT=$VPORT
  rm -f "$REAX_INPROC_READY"
  export REAX_VLLM_BASE_URL="http://127.0.0.1:$VPORT"
  [ "${REAX_DUAL:-1}" = 1 ] && export REAX_VLLM_IMAGE_URL="http://127.0.0.1:$IPORT"
  IMG_INPROC=0
  if [ "${REAX_DUAL:-1}" = 1 ] && [ "${REAX_INPROC_IMAGE:-0}" = 1 ]; then
    IMG_INPROC=1
    export REAX_INPROC_IMAGE_ENV=${REAX_INPROC_IMAGE_ENV:-/tmp/reax_inproc_image_env.json} \
      REAX_INPROC_IMAGE_GO=${REAX_INPROC_IMAGE_GO:-/tmp/reax_inproc_image_go} REAX_INPROC_IMAGE_EXTRA="$IMAGE_EXTRA" \
      REAX_VLLM_IMAGE_PORT=$IPORT
    rm -f "$REAX_INPROC_IMAGE_GO"
    (image_env python3 -I -S -c 'import json, os, sys; json.dump(dict(os.environ), open(sys.argv[1], "w"))' \
      "$REAX_INPROC_IMAGE_ENV") || exit 3
  fi
  REAX_BLINK_OVERLAY=${REAX_BLINK_OVERLAY_TEXT:-1} python3 -m engine.serve --model blink-v0.3-26b-a4b --model-dir "$DIR" \
    --host "${REAX_HOST:-0.0.0.0}" --port "${REAX_PORT:-8741}" --opt "preset=${REAX_PRESET}" &
  SRV=$!
  for _ in $(seq 1 900); do
    [ -f "$REAX_INPROC_READY" ] && break
    kill -0 $SRV 2>/dev/null || { echo "reax: REAX server died during startup"; exit 3; }
    sleep 2
  done
  [ -f "$REAX_INPROC_READY" ] || { echo "reax: in-process text vLLM not ready"; kill $SRV; exit 3; }
  python3 -I /opt/engv2/engv2_pins.py check
  [ $IMG_INPROC = 1 ] && touch "$REAX_INPROC_IMAGE_GO"
  PIDS=""
  if [ "${REAX_DUAL:-1}" = 1 ] && [ $IMG_INPROC = 0 ]; then
    start_image || { kill $SRV; exit 3; }
    PIDS="$P2"
  fi
  trap 'kill $PIDS $SRV 2>/dev/null' TERM INT
  wait -n $PIDS $SRV
  echo "reax: a child process exited; shutting down"
  kill $PIDS $SRV 2>/dev/null
  exit 1
fi
REAX_BLINK_OVERLAY=${REAX_BLINK_OVERLAY_TEXT:-1} python3 -m engine.backends.vllm_readout blink ref-prod "$DIR" --port "$VPORT" --exec ${REAX_VLLM_EXTRA:-} $NOSTATS_TEXT &
P1=$!
wait_health "$VPORT" $P1 || exit 3
python3 -I /opt/engv2/engv2_pins.py check
PIDS="$P1"
if [ "${REAX_DUAL:-1}" = 1 ]; then
  start_image || exit 3
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
