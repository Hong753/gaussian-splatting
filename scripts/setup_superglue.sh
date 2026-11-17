#!/usr/bin/env bash
# Cache the official pretrained PyTorch models, without modifying the environment.
set -euo pipefail
if (($#!=1)); then echo 'Usage: setup_superglue.sh CACHE_DIRECTORY' >&2; exit 2; fi
revision=ddcf11f42e7e0732a0c4607648f9448ea8d73590
repository="$1"
mkdir -p "$(dirname "$repository")"
exec 8>"$repository.lock"
flock 8
if [[ ! -d "$repository" ]]; then
  stage="$(mktemp -d "$repository.download_XXXXXX")"
  trap 'rm -rf "$stage"' EXIT
  echo 'Downloading official SuperPoint/SuperGlue code and pretrained weights (one-time cache).'
  git clone --quiet --no-checkout https://github.com/magicleap/SuperGluePretrainedNetwork.git "$stage"
  git -C "$stage" checkout --quiet --detach "$revision"
  mv "$stage" "$repository"
  trap - EXIT
fi
if [[ "$(git -C "$repository" rev-parse HEAD)" != "$revision" ]]; then
  echo "SuperGlue cache must be at commit $revision: $repository" >&2; exit 1
fi
git -C "$repository" diff --exit-code HEAD -- models/superpoint.py models/superglue.py \
  models/weights/superpoint_v1.pth models/weights/superglue_indoor.pth >/dev/null
echo "SuperPoint/SuperGlue ready: $repository"
