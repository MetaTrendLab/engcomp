# engcomp

Sources and exact build steps of the compiled kernels in our REAX (Bittensor subnet 92) engine-competition entries. One branch per submission; each branch has the kernels, a README naming the UID and the image digest, and `build.sh`, which rebuilds every shipped cubin byte for byte.

| branch | uid | image | natives | published |
|---|---|---|---|---|
| [`g5-u14`](../../tree/g5-u14) | 14 | `ghcr.io/elninolab/g5@sha256:61d0c90c4937376a9422aa55a91668d910917d55ad1332b8d74ded8a22359a4e` | ecep.cubin, ecrin.cubin, onormpv.cubin | 2026-10-26 16:25 UTC |
| [`v1f`](../../tree/v1f) | 14 | `ghcr.io/elninolab/v1f@sha256:8aef3a88da9fe6a6c8464bb217d9a64051ff0706a5018057e7eacf5a9b889982` | — | 2026-10-27 13:19 UTC |
| [`v1g`](../../tree/v1g) | 243 | `ghcr.io/elninolab/v1g@sha256:36b37b0b58070239dda0c19386834bf7c0fc69389f5ab428241d29a0e248ceb7` | — | 2026-10-27 13:28 UTC |
| [`v2-dm18n-v2-dm18n`](../../tree/v2-dm18n-v2-dm18n) | 208 | `ghcr.io/elninolab/v2-dm18n@sha256:356b770cc7d7eb3c5f0728c01a8b005eb7597dc4bdde8f07637d3a6d5bfbd009` | b7sk.cubin, bf16g7.cubin, bf16gemm.cubin, f8gemm.cubin, f8g6.cubin, f8sk.cubin, libreax_gact.so, libreax_gquant.so, libreax_moedn.so, libreax_vattn.so, libreax_vattn2.so, f8g6.cubin, f8gemm.cubin, f8sk.cubin, reax_lt.cpython-312-x86_64-linux-gnu.so | 2026-10-29 09:16 UTC |
| [`v2-dm22`](../../tree/v2-dm22) | 166 | `ghcr.io/elninolab/dm22@sha256:47a211d942b9e1e7afa8b5bfac6ba774dfa363898a83442cc8d09eb1bae35dfe` | b7sk.cubin, bf16g7.cubin, bf16gemm.cubin, f8gemm.cubin, f8g6.cubin, f8sk.cubin, libreax_gact.so, libreax_gquant.so, libreax_moedata.so, libreax_moedn.so, libreax_moegu.so, libreax_vattn.so, libreax_vattn2.so, rgemm.cubin, f8g6.cubin, f8gemm.cubin, f8sk.cubin, reax_lt.cpython-312-x86_64-linux-gnu.so | 2026-10-29 16:16 UTC |
