#!/bin/bash
# Rebuild the cubins shipped in ghcr.io/elninolab/g5@sha256:61d0c90c4937376a9422aa55a91668d910917d55ad1332b8d74ded8a22359a4e
# from the sources in kernels/ and compare them byte for byte with the shipped ones (SHA256SUMS).
# Needs docker; no GPU, no network inside the container.
set -euo pipefail
cd "$(dirname "$0")"
CUDA=${CUDA_IMAGE:-nvidia/cuda:13.4.3-devel-ubuntu24.04}
out=$(mktemp -d)
trap 'rm -rf -- "${out:?}"' EXIT
cp kernels/* "$out/"
build() {  # <build dir inside the container> <command>
  local dir=$1; shift
  docker run --rm --network none --user "$(id -u):$(id -g)" -v "$out:$dir" -w "$dir" "$CUDA" bash -c "$*" >/dev/null
}
build /opt/s1eng/s1eng 'nvcc -O3 -arch=sm_120a -cubin -std=c++17 -Xptxas -v -o ecep.cubin ecep.cu'
build /opt/s1eng/s1eng 'nvcc -O3 -arch=sm_120a -cubin -std=c++17 -o ecrin.cubin ecrin.cu'
build /w 'nvcc -O3 -arch=sm_120a -cubin -std=c++17 -Xptxas -v -DFFS_SHIP -DONORM_V=5 -DONORM_PF2 -DONORM_SILU_FIN -DONORM_FINROLL -o onormpv.cubin onormpv.cu'
sha256sum kernels/*
(cd "$out" && sha256sum *.cubin)
grep -E 'cubin$' SHA256SUMS | (cd "$out" && sha256sum -c -)
echo "all cubins rebuilt byte for byte"
