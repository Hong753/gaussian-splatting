#!/usr/bin/env bash
# Usage: bash scripts/run_orange4f_upstream_scaffold.sh GPU fixed|adaptive [SOURCE]
set -euo pipefail
gpu="${1:?Supply a GPU number and fixed or adaptive}"
variant="${2:?Supply fixed or adaptive}"
[[ "$gpu" =~ ^[0-9]+$ && "$variant" =~ ^(fixed|adaptive)$ ]] || exit 2
version="${SCAFFOLD_RUN_VERSION:-v1}"
[[ "$version" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
source "${RECONSTRUCTION_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
conda activate 3dgs
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="$gpu"
suite="${ORANGE4F_OUTPUT_ROOT:-/data/3dgs_outputs/orange4f/3dgs_finetuning}/stride3_noaa_v1"
model="$suite/scaffold_upstream_${variant}_${version}"
mkdir -p "$(dirname "$suite")"
exec 9>"$suite.lock"
flock 9
if [[ -n "${3:-}" ]]; then
  source_scene="$3"
elif [[ -f "$suite/suite.json" ]]; then
  source_scene="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$suite/suite.json")"
else
  source_scene=/workspace/colmap_scenes/orange4f
fi
cloud_args=()
if [[ -n "${ORANGE4F_POINT_CLOUD:-}" ]]; then cloud_args=(--point-cloud "$ORANGE4F_POINT_CLOUD"); fi
python -u scripts/prepare_orange4f.py --source "$source_scene" --output "$suite" "${cloud_args[@]}"
flock -u 9
exec 8>"$model.lock"
flock -n 8 || { echo "Another process is running $model" >&2; exit 1; }

# Keep the authors' CUDA rasterizer private. Both renderers have the same Python import name.
revision="$(python -c 'from scripts.upstream_scaffold import REVISION; print(REVISION)')"
cache="${SCAFFOLD_UPSTREAM_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/orange4f-scaffold}/$revision"
mkdir -p "$cache"
exec 7>"$cache/setup.lock"
flock 7
if [[ ! -d "$cache/source" ]]; then
  checkout="$(mktemp -d "$cache/checkout.XXXXXX")"
  git -C "$checkout" init -q
  git -C "$checkout" fetch -q --depth 1 https://github.com/city-super/Scaffold-GS.git "$revision"
  git -C "$checkout" checkout -q --detach FETCH_HEAD
  mv "$checkout" "$cache/source"
fi
[[ "$(git -C "$cache/source" rev-parse HEAD)" == "$revision" ]]
git -C "$cache/source" diff --quiet HEAD
abi="$(python -c 'import sys, torch, simple_knn._C; print(f"py{sys.version_info.major}{sys.version_info.minor}-torch{torch.__version__}-cuda{torch.version.cuda}-sm{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}")')"
deps="$cache/$abi"
if [[ ! -f "$deps/complete" ]]; then
  build_dir="$(mktemp -d "$cache/build.XXXXXX")"
  trap 'rm -rf -- "$build_dir"' EXIT
  stage="$build_dir/python"
  export MAX_JOBS="${MAX_JOBS:-4}"
  # Older upstream CUDA code uses standard fixed-width integer types.
  build_src="$build_dir/raster"
  mkdir -p "$build_src"
  cp -a "$cache/source/submodules/diff-gaussian-rasterization/." "$build_src/"
  sed -i '/^#include <vector>/i #include <cstdint>' "$build_src/cuda_rasterizer/rasterizer.h"
  python -m pip install --no-build-isolation --no-deps --target "$stage" "$build_src"
  for dep in einops colorama jaxtyping; do
    python -c "import $dep" 2>/dev/null || python -m pip install --target "$stage" "$dep"
  done
  if ! python -c 'from torch_scatter import scatter_max' 2>/dev/null; then
    python -m pip install --no-build-isolation --no-deps --target "$stage" torch-scatter==2.1.2
  fi
  PYTHONPATH="$stage${PYTHONPATH:+:$PYTHONPATH}" python -c 'import torch, simple_knn._C; from torch_scatter import scatter_max; from diff_gaussian_rasterization import GaussianRasterizer; assert hasattr(GaussianRasterizer, "visible_filter")'
  touch "$stage/complete"
  mv "$stage" "$deps"
  rm -rf -- "$build_dir"
  trap - EXIT
fi
flock -u 7
python -u scripts/upstream_scaffold.py prepare --suite "$suite" --model "$model" --variant "$variant" \
  --source-code "$cache/source" --dependencies "$deps" --voxel-size "${SCAFFOLD_VOXEL_SIZE:-0.02}"
python -u scripts/upstream_scaffold.py train --model "$model" 2>&1 | tee -a "$model/training.log"
python -u scripts/upstream_scaffold.py score --model "$model"
exec python -u scripts/render.py --model "$model" --iteration 30000 --split both --eval-set unselected \
  --max-views 32 --no-antialiasing --gpu "$gpu" --output "$model/comparisons"
