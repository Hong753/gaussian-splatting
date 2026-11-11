#!/usr/bin/env bash
# Fresh LiDAR initialization; soft regularization only; default GPU 0.
set -euo pipefail
source "${SOFT_SURFACE_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
conda activate 3dgs
cd "$(dirname "${BASH_SOURCE[0]}")/.."
soft_gpu="${1:-0}"
if (($# > 0)); then shift; fi
soft_root="${SOFT_SURFACE_OUTPUT_ROOT:-/data/3dgs_outputs/orange_4F/1_full}"
soft_source="${SOFT_SURFACE_SOURCE:-/data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full}"
soft_baseline="${SOFT_SURFACE_BASELINE:-}"
if [[ -z "$soft_baseline" ]]; then
  soft_baseline="$(python -c 'from scripts.render import resolve_models; import sys; print(resolve_models(None, sys.argv[1], ["sharp_keyframes_stride3_aa"])[0])' "$soft_root")"
fi
soft_model="${SOFT_SURFACE_MODEL:-$soft_root/3dgs_stride3_aa_soft_surface_warm7k_lr50_v1}"
python -u scripts/run_soft_surface.py \
  --source "$soft_source" --model "$soft_model" --gpu "$soft_gpu" \
  --scene-geometry "${SOFT_SURFACE_SCENE:-$soft_root/lidar_scene_surface_v1/scene.json}" \
  --baseline "$soft_baseline" --keyframes "$soft_baseline/keyframes.json" "$@"
