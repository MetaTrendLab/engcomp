#!/bin/bash
set -euo pipefail
TAG=${1:?usage: ENGINE_HOTKEY=<ss58> ./build.sh <registry/image:tag> [with-reax1]}
HK=${ENGINE_HOTKEY:?ENGINE_HOTKEY=<ss58 of the committing hotkey>}
M=mammoth,blink,jeffan,vision
[ "${2:-}" = with-reax1 ] && M=mammoth,blink,reax1,jeffan,vision
cd "$(dirname "$0")/src"
docker build -q -f tools/engv2/blink/Dockerfile.m1 -t engv2-blink:dm22-sub tools >/dev/null
docker build -q -f engv2/docker/Dockerfile.dm1 --build-arg BLINK=engv2-blink:dm22-sub --build-arg MODELS=$M \
  --build-arg ENGINE_HOTKEY="$HK" -t "$TAG" engv2 >/dev/null
docker push -q "$TAG" >/dev/null
docker inspect --format '{{index .RepoDigests 0}}' "$TAG"
