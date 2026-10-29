# Compiled code in this image (engine competition v2.0, track dm-all)

Entry committed by uid 166, hotkey `5EPFQfgtzZVpj3pVtGkEr4ffPRWLsKcQtau6bVpBFW9e3pRd`. Every native file (.so / .cubin) of our layers, its sha256, the source
it is built from (sha256 of the source and of every header it includes) and the exact build line. "byte-identical" = our own
rebuild inside this very image (its nvcc) reproduces the shipped bytes. Natives of the reference images are not listed:
they are the base images' files, unchanged. Sources: https://github.com/MetaTrendLab/engcomp/tree/v2-dm22

## Toolchain

Rebuild status: this entry was committed without the byte-for-byte rebuild pass (fast submission); the build lines below are the recorded ones and the verification follows in an update of this branch.

- Compiler: `Cuda compilation tools, release 13.0, V13.0.88 Build cuda_13.0.r13.0/compiler.36424714_0` (the image's own `nvcc`; the image is built on the reference image
  `ghcr.io/reaxlabs/reax-engine-ref-blink@sha256:97d06a91453979d6756176aba4a1cb1d78b38390643c37ed1b83f79a4ff9f606` (slot vision), whose CUDA toolkit it uses).
- Target: as in each build line (`sm_120a` / `sm_120f`, RTX PRO 6000).
- Working directory: the directory of the native; the source is given by its absolute path in the image, exactly as in the
  build line (nvcc hashes that path into the module id, so the bytes depend on it). Run the build line inside the image,
  or inside the reference image above with the `image/` tree of this branch copied to the same paths.
- A build line that ends in `>/dev/null` in the image's RUN layer is shown without it; `-Xptxas -v` is kept when it was used.

| native | sha256 | source(s) | build line | rebuild |
|---|---|---|---|---|
| `/opt/engv2/b7sk.cubin` | `69b373d80004d4b4…` | `b7sk.cu` 2b675270040d, `bf16g7.cu` 64bb6672aeef | `nvcc -arch=sm_120a -O3 -cubin -Xptxas -v -o /opt/engv2/b7sk.cubin /opt/engv2/b7sk.cu` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/engv2/bf16g7.cubin` | `fa55c8dc7e713482…` | `bf16g7.cu` 64bb6672aeef | `nvcc -arch=sm_120a -O3 -cubin -Xptxas -v -o /opt/engv2/bf16g7.cubin /opt/engv2/bf16g7.cu` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/engv2/bf16gemm.cubin` | `6c3b4fff9150b441…` | `bf16gemm.cu` b7ab83855415 | `nvcc -O3 -std=c++17 -cubin -gencode arch=compute_120a,code=sm_120a -o /opt/engv2/bf16gemm.cubin /opt/engv2/bf16gemm.cu` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/engv2/f8gemm.cubin` | `18f97e21feb2a290…` | `f8gemm.cu` 032921f68800 | `nvcc -O3 -std=c++17 -cubin -gencode arch=compute_120a,code=sm_120a -o /opt/engv2/f8gemm.cubin /opt/engv2/f8gemm.cu` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/engv2/jf8/f8g6.cubin` | `b253f8c44fcbc080…` | `f8g6.cu` dc2e9d2c8b24 | `nvcc -O3 -std=c++17 -cubin -gencode arch=compute_120a,code=sm_120a -o /opt/engv2/jf8/f8g6.cubin /opt/q35/engine/kern/f8g6.cu` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/engv2/jf8/f8sk.cubin` | `e6330f0710b81217…` | `f8sk.cu` 575854236479, `f8g6.cu` dc2e9d2c8b24 | `nvcc -O3 -std=c++17 -cubin -gencode arch=compute_120a,code=sm_120a -o /opt/engv2/jf8/f8sk.cubin /opt/q35/engine/kern/f8sk.cu` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/engv2/libreax_gact.so` | `6aa93e422ee6697b…` | `moe_gact.cu` 2698444d86f3 | `nvcc -O3 -shared -Xcompiler -fPIC -Xlinker --strip-all -gencode arch=compute_120f,code=sm_120f -o /opt/engv2/libreax_gact.so /opt/engv2/moe_gact.cu` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/engv2/libreax_gquant.so` | `7782e58fd0be9c0e…` | `moe_gquant.cu` 40230dd7ceab | `nvcc -O3 -shared -Xcompiler -fPIC -Xlinker --strip-all -gencode arch=compute_120f,code=sm_120f -o /opt/engv2/libreax_gquant.so /opt/engv2/moe_gquant.cu` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/engv2/libreax_moedata.so` | `ced00001e097a349…` | `moe_data.cu` bfd1ff3323cd | `nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC -Xlinker --strip-all -gencode arch=compute_120a,code=sm_120a -o /opt/engv2/libreax_moedata.so /opt/engv2/moe_data.cu` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/engv2/libreax_moedn.so` | `8222c391c6ca6aeb…` | `moe_dn.cu` 80c99b4d13ad | `nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC -Xlinker --strip-all -gencode arch=compute_120a,code=sm_120a -o /opt/engv2/libreax_moedn.so /opt/engv2/moe_dn.cu` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/engv2/libreax_moegu.so` | `203d12a1e8b84cd3…` | `moe_gu.cu` 81123db53718 | `nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC -Xlinker --strip-all -gencode arch=compute_120a,code=sm_120a -o /opt/engv2/libreax_moegu.so /opt/engv2/moe_gu.cu` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/engv2/libreax_vattn.so` | `718a1d36d69f3e41…` | `vis_attn.cu` a56839b64f4a | `nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC -Xlinker --strip-all -gencode arch=compute_120a,code=sm_120a -o /opt/engv2/libreax_vattn.so /opt/engv2/vis_attn.cu` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/engv2/libreax_vattn2.so` | `ebc2d83e3729dc47…` | `vis_attn2.cu` 4a213a02cb27 | `nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC -Xlinker --strip-all -gencode arch=compute_120a,code=sm_120a -o /opt/engv2/libreax_vattn2.so /opt/engv2/vis_attn2.cu` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/engv2/rgemm.cubin` | `abe44e03cc7e0ae6…` | `rgemm.cu` bdae14346f05, `bf16g7.cu` 64bb6672aeef | `nvcc -arch=sm_120a -O3 -cubin -o /opt/engv2/rgemm.cubin /opt/engv2/rgemm.cu` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/q35/engine/kern/f8g6.cubin` | `b253f8c44fcbc080…` | `f8g6.cu` dc2e9d2c8b24 | `nvcc -O3 -std=c++17 -cubin -gencode arch=compute_120a,code=sm_120a -o /opt/q35/engine/kern/f8g6.cubin /opt/q35/engine/kern/f8g6.cu` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/q35/engine/kern/f8gemm.cubin` | `fd182b4d4827e6cc…` | `f8gemm.cu` 032921f68800 | `nvcc -O3 -std=c++17 -cubin -gencode arch=compute_120a,code=sm_120a -o /opt/q35/engine/kern/f8gemm.cubin /opt/q35/engine/kern/f8gemm.cu` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/opt/q35/engine/kern/f8sk.cubin` | `e6330f0710b81217…` | `f8sk.cu` 575854236479, `f8g6.cu` dc2e9d2c8b24 | `nvcc -O3 -std=c++17 -cubin -gencode arch=compute_120a,code=sm_120a -o /opt/q35/engine/kern/f8sk.cubin /opt/q35/engine/kern/f8sk.cu` | not rebuilt (fast submission: byte-for-byte verification pending) |
| `/usr/local/lib/python3.12/dist-packages/reax_lt.cpython-312-x86_64-linux-gnu.so` | `555d910c825c2b4d…` | — | `cd /build && TORCH_CUDA_ARCH_LIST=12.0 python3 setup.py build_ext --inplace >/build/build.log 2>&1` | not rebuilt (fast submission: byte-for-byte verification pending) |

Declared models (`ai.reax.engine.models`): mammoth,blink,reax1,jeffan,vision. The engine reads `REAX_MODEL` and the weights at `/model`.
