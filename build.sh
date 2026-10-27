#!/bin/bash
# ENGINE_HOTKEY=<ss58 of the hotkey that will commit> ./build.sh <image:tag> [--no-push]
#   -> builds (labels: hotkey, parents.json), pushes, prints <image>@sha256:<digest>
set -euo pipefail
TAG=${1:?image:tag (e.g. ghcr.io/<user>/<name>:e1 or 127.0.0.1:5000/engcomp-q8s:dev)}
cd "$(dirname "$0")"
true
[ -n "${ENGINE_HOTKEY:-}" ] || echo "warning: ENGINE_HOTKEY empty (fine for a local dev build, not for a submission)" >&2
docker build --network host -q --build-arg ENGINE_HOTKEY="${ENGINE_HOTKEY:-}" \
  --build-arg ENGINE_PARENTS="$(cat parents.json)" -t "$TAG" . >/dev/null
[ "${2:-}" = "--no-push" ] && { docker image inspect "$TAG" --format '{{.Id}}'; exit 0; }
docker push -q "$TAG" >/dev/null
repo=${TAG%:*}
docker image inspect "$TAG" --format '{{range .RepoDigests}}{{println .}}{{end}}' | grep "^$repo@" | head -1
