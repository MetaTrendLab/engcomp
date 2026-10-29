# engcomp v2-dm18n-v2-dm18n: uid 208 (engine competition v2.0, track dm-all)

- Image: `ghcr.io/elninolab/v2-dm18n@sha256:356b770cc7d7eb3c5f0728c01a8b005eb7597dc4bdde8f07637d3a6d5bfbd009`
- Committed by uid 208, hotkey `5HEmARQrC2Qrp3KciXngp28MoBCoCF2bnBfwGsvcgxUozfUg`
- Declared models (`ai.reax.engine.models`): mammoth,blink,jeffan,vision
- Candidate image before the wrap: `engv2-dm:18x-ns` (id `de7c8de13afe`); `Dockerfile` here is the wrap
  (labels, `USER 10002:10002`, SOURCES.md + KERNELS.md), nothing else changes.

## What is here

- `image/`: every file of our own layers exactly as shipped in the image (sources, Python, patches, pins, the compiled
  kernels). Paths mirror the image. Nothing of the REAX Engine reference images is copied here; they are the base images.
- `build/`: the build recipes from our repository at `cf6e171a` (engv2/docker/Dockerfile.dm1, engv2/docker/entry_dm.sh, tools/engv2/blink/Dockerfile.m1, tools/sidecar/reax_lt/reax_lt.cpp, tools/sidecar/reax_lt/setup.py): the Blink m-line
  (`Dockerfile.m1`, built on top of `ghcr.io/reaxlabs/reax-engine-ref-blink`), the one-image `dm-all` entry (`Dockerfile.dm1`,
  `entry_dm.sh`: the model from `REAX_MODEL`), and the `reax_lt` extension built in its own stage.
- `KERNELS.md`: every native of our layers with sha256, source sha256 and the exact build line, and our rebuild result.

## Natives

| native | rebuild inside this image |
|---|---|
| `/opt/engv2/b7sk.cubin` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/engv2/bf16g7.cubin` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/engv2/bf16gemm.cubin` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/engv2/f8gemm.cubin` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/engv2/jf8/f8g6.cubin` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/engv2/jf8/f8sk.cubin` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/engv2/libreax_gact.so` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/engv2/libreax_gquant.so` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/engv2/libreax_moedn.so` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/engv2/libreax_vattn.so` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/engv2/libreax_vattn2.so` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/q35/engine/kern/f8g6.cubin` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/q35/engine/kern/f8gemm.cubin` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/q35/engine/kern/f8sk.cubin` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/usr/local/lib/python3.12/dist-packages/reax_lt.cpython-312-x86_64-linux-gnu.so` | not rebuilt (fast submission: byte-for-byte verification pending) |

## Rewritten third-party files

Files of the base image that our build rewrote in place (patched by the scripts and `.patch` files under `image/opt/engv2`,
as the recipes in `build/` show) or added next to it. They are in `image/` exactly as shipped, so they can be compared with the
image byte for byte; their licence is the original one (`NOTICE`).

- `/opt/reax/engine/backends/vllm_readout.py`
- `/opt/reax/engine/serve.py`
- `/usr/local/lib/python3.12/dist-packages/reax_lt.cpython-312-x86_64-linux-gnu.so`
- `/usr/local/lib/python3.12/dist-packages/transformers/models/gemma4/modeling_gemma4.py`
- `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/determinism/batch_invariant.py`
- `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py`
- `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/linear.py`
- `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py`
- `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4_mm.py`
- `/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_unified_attention.py`
- `/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py`
- `/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py`

## Rebuild

- Base: `ghcr.io/reaxlabs/reax-engine-ref-blink@sha256:97d06a91453979d6756176aba4a1cb1d78b38390643c37ed1b83f79a4ff9f606`; `/opt/venv` in the image is the mammoth reference's venv copied in whole
  (`COPY --from=<ref-mammoth> /opt/venv /opt/venv`), so it is not repeated here.
- Natives: see `KERNELS.md` (compiler version, working directory, exact line per file).
- Every file of our layers is in `image/`.

## Licence

Our files: MIT (`LICENSE`). Files that come from the REAX Engine reference images keep the REAX Engine License, rewritten
vLLM / transformers files Apache-2.0 (`NOTICE`).
