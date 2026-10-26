# g5 — kernel sources of the REAX engine entry `g5` (uid 14)

Public source of the compiled kernels in the REAX (Bittensor subnet 92) engine-competition entry **engcomp-g5**:

| | |
|---|---|
| Image | `ghcr.io/elninolab/g5@sha256:61d0c90c4937376a9422aa55a91668d910917d55ad1332b8d74ded8a22359a4e` |
| Commitment | `reax-eng/1:s1-fast:ghcr.io/elninolab/g5@sha256:61d0c90c4937376a9422aa55a91668d910917d55ad1332b8d74ded8a22359a4e` (uid 14, hotkey `5Gn8txQhUg1NKqu89dmTT4Lxebd6jFBNrov3nj3HZsG7sq83`) |
| Base image | `ghcr.io/johnkell9801-hub/s1q9@sha256:f1422b06f6098ad7b8939c1e05a48b82493d2ccc672912b92e21491256ca0898` |

The image ships these compiled kernels of its own; every other native is the base image's, byte for byte.

| native (in the image) | source (this branch) | built in | sha256 | our rebuild |
|---|---|---|---|---|
| `/opt/s1eng/s1eng/ecep.cubin` | `kernels/ecep.cu` | `/opt/s1eng/s1eng` | `c197d31304b7256d174f5e55be31d89ceceae2e5bafd13dd087295be14079ad9` | byte-identical |
| `/opt/s1eng/s1eng/ecrin.cubin` | `kernels/ecrin.cu` | `/opt/s1eng/s1eng` | `796a4c85f1371ca251e2c153c8e3ac6995c7aba7e62be8d33be143b363040cad` | byte-identical |
| `/opt/s1eng/s1eng/onormpv.cubin` | `kernels/onormpv.cu` | `/w` | `cfd1ae2f616c1b3f6f6c4a446183c11678f03652212d41183b7da09896134773` | byte-identical |

The files in `kernels/` are the same files the image carries next to the kernels (sha256 in `SHA256SUMS`).

## Rebuild

Toolchain: `nvidia/cuda:13.4.3-devel-ubuntu24.04` (nvcc 13.4). No GPU needed.

```bash
./build.sh            # rebuilds every cubin and checks it against SHA256SUMS
```

or by hand (each cubin is written next to the sources):

```bash
# ecep.cubin
docker run --rm --network none -v "$PWD/kernels:/opt/s1eng/s1eng" -w /opt/s1eng/s1eng \
  nvidia/cuda:13.4.3-devel-ubuntu24.04 \
  nvcc -O3 -arch=sm_120a -cubin -std=c++17 -Xptxas -v -o ecep.cubin ecep.cu

# ecrin.cubin
docker run --rm --network none -v "$PWD/kernels:/opt/s1eng/s1eng" -w /opt/s1eng/s1eng \
  nvidia/cuda:13.4.3-devel-ubuntu24.04 \
  nvcc -O3 -arch=sm_120a -cubin -std=c++17 -o ecrin.cubin ecrin.cu

# onormpv.cubin
docker run --rm --network none -v "$PWD/kernels:/w" -w /w \
  nvidia/cuda:13.4.3-devel-ubuntu24.04 \
  nvcc -O3 -arch=sm_120a -cubin -std=c++17 -Xptxas -v -DFFS_SHIP -DONORM_V=5 -DONORM_PF2 -DONORM_SILU_FIN -DONORM_FINROLL -o onormpv.cubin onormpv.cu
```

**Why `/w` for `onormpv.cubin`:** nvcc names anonymous-namespace device functions after its module id, the CRC32 of the source file's absolute path, so the bytes depend on the directory it is compiled in. The shipped file carries module id `473d68c0` = CRC32 of `/w/onormpv.cu`; built in `/opt/s1eng/s1eng` it differs.

**Build line of `ecrin.cubin`:** the comment in the source says `nvcc -O3 -arch=sm_120a -cubin -std=c++17 -Xptxas -v -o ecrin.cubin ecrin.cu`; the shipped file was built with `nvcc -O3 -arch=sm_120a -cubin -std=c++17 -o ecrin.cubin ecrin.cu` (the extra flag changes the output of this kernel).

## Compare with the image

```bash
docker run --rm --network none --entrypoint sha256sum \
  ghcr.io/elninolab/g5@sha256:61d0c90c4937376a9422aa55a91668d910917d55ad1332b8d74ded8a22359a4e \
  /opt/s1eng/s1eng/ecep.cu /opt/s1eng/s1eng/ecrin.cu /opt/s1eng/s1eng/onorm.cu /opt/s1eng/s1eng/onormpv.cu /opt/s1eng/s1eng/q8.cu /opt/s1eng/s1eng/rotf_consts.inc /opt/s1eng/s1eng/ecep.cubin /opt/s1eng/s1eng/ecrin.cubin /opt/s1eng/s1eng/onormpv.cubin
```

## Declared parents (`ai.reax.engine.parents`)

| uid | digest |
|---|---|
| 176 | `01e6e113da15bbc5f992779bb099a55d99b936ab088b64fb4967ef557f72db6d` |
| 56 | `23dcdecc1519edf813699ab48adadd1df07a269016da3816bee37cba84e8a89d` |
| 133 | `bb2890dcd66d03013b3140ea183d8b18da9f7bef34079a3490a4d87020792558` |
| 161 | `d32e69ebc30ef65257b37ba9b6f16d8389a08b46d655be39e3e1c380551d0e70` |
| 164 | `944fa7aee414607c4b620de5600224fb7b844f0f9d844031b07f591b8b7455f7` |
| 223 | `74242a1cebff468026cbc5d128623b2e8f65994df46992064cc3e6e6ed024970` |
| 220 | `f1422b06f6098ad7b8939c1e05a48b82493d2ccc672912b92e21491256ca0898` |
| 240 | `244f02c4a855082df3e4afcef4de39bb1ed7003350bcdf3457ce99f640dca7b6` |

What we changed is listed in `ENGCOMP-NOTICE` (the same file is in the image).

## Licence

MIT (`LICENSE`). Origins of the derived files and credits: `NOTICE` and `ENGCOMP-NOTICE`.
