# engcomp v2-dm22: uid 166 (engine competition v2.0, track dm-all)

- Image: `ghcr.io/elninolab/dm22@sha256:47a211d942b9e1e7afa8b5bfac6ba774dfa363898a83442cc8d09eb1bae35dfe`
- Committed by uid 166, hotkey `5EPFQfgtzZVpj3pVtGkEr4ffPRWLsKcQtau6bVpBFW9e3pRd`
- Declared models (`ai.reax.engine.models`): mammoth,blink,reax1,jeffan,vision
- Built on the host from the submit package `engcomp-v2-dm22` with `build/build.sh` (image id `b44b2ab29812`); `Dockerfile` here is that package's `Dockerfile.dm1` (the `USER 10002:10002` and the hotkey label are set there).

## What is here

- `image/`: every file of our own layers exactly as shipped in the image (sources, Python, patches, pins, the compiled
  kernels). Paths mirror the image. Nothing of the REAX Engine reference images is copied here; they are the base images.
- `build/`: the build recipes from our repository at `submit package `engcomp-v2-dm22` (MANIFEST sha256 417fea5ba059)` (MANIFEST, build.sh, src/engv2/docker/Dockerfile.dm1, src/engv2/docker/entry_dm.sh, src/tools/engv2/blink/Dockerfile.m1, src/tools/sidecar/reax_lt/setup.py): the Blink m-line
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
| `/opt/engv2/libreax_moedata.so` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/engv2/libreax_moedn.so` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/engv2/libreax_moegu.so` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/engv2/libreax_vattn.so` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/engv2/libreax_vattn2.so` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/engv2/rgemm.cubin` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/q35/engine/kern/f8g6.cubin` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/q35/engine/kern/f8gemm.cubin` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/q35/engine/kern/f8sk.cubin` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/usr/local/lib/python3.12/dist-packages/reax_lt.cpython-312-x86_64-linux-gnu.so` | not rebuilt (fast submission: byte-for-byte verification pending) |

## Rewritten third-party files

Files of the base image that our build rewrote in place (patched by the scripts and `.patch` files under `image/opt/engv2`,
as the recipes in `build/` show) or added next to it. They are in `image/` exactly as shipped, so they can be compared with the
image byte for byte; their licence is the original one (`NOTICE`).

- `/opt/reax/engine/backends/vllm_readout.py`
- `/opt/reax/engine/enc_fast.py`
- `/opt/reax/engine/fn_cpu.py`
- `/opt/reax/engine/inproc.py`
- `/opt/reax/engine/serve.py`
- `/opt/reax/engine/spin_keep.py`
- `/opt/reax/engine/thread_cpu.py`
- `/usr/local/lib/python3.12/dist-packages/reax_lt.cpython-312-x86_64-linux-gnu.so`
- `/usr/local/lib/python3.12/dist-packages/transformers/models/gemma4/modeling_gemma4.py`
- `/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/completion/serving.py`
- `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/determinism/batch_invariant.py`
- `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py`
- `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/router/gate_linear.py`
- `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/linear.py`
- `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py`
- `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4_mm.py`
- `/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_unified_attention.py`
- `/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py`
- `/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py`
- `/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/sample/gumbel.py`
- `/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/sample/logit_bias.py`
- `/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/sample/logprob.py`

## Rebuild

- Base: `ghcr.io/reaxlabs/reax-engine-ref-blink@sha256:97d06a91453979d6756176aba4a1cb1d78b38390643c37ed1b83f79a4ff9f606`; `/opt/venv` in the image is the mammoth reference's venv copied in whole
  (`COPY --from=<ref-mammoth> /opt/venv /opt/venv`), so it is not repeated here.
- Natives: see `KERNELS.md` (compiler version, working directory, exact line per file).
- Every file of our layers is in `image/`.

## Licence

Our files: MIT (`LICENSE`). Files that come from the REAX Engine reference images keep the REAX Engine License, rewritten
vLLM / transformers files Apache-2.0 (`NOTICE`).
