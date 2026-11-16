#!/usr/bin/env bash
# Large-scene appearance training at fixed measured XYZ, with pruning retained.
set -euo pipefail
if (($#>1)); then echo 'Usage: bash scripts/run_native_lidar_fixed_200k.sh [GPU]' >&2; exit 2; fi
source "${RECONSTRUCTION_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
conda activate 3dgs
cd "$(dirname "${BASH_SOURCE[0]}")/.."
fixed_gpu="${1:-1}"
if [[ -n "$(nvidia-smi -i "$fixed_gpu" --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "GPU $fixed_gpu is occupied; the new run was not started." >&2; exit 1
fi
fixed_root="${NATIVE_LIDAR_OUTPUT_ROOT:-/data/3dgs_outputs/orange_4F/1_full}"
fixed_experiments="${NATIVE_LIDAR_FINETUNING_ROOT:-$fixed_root/3dgs_finetuning}"
fixed_source="${NATIVE_LIDAR_SOURCE:-/data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full}"
fixed_baseline="${NATIVE_LIDAR_BASELINE:-}"
if [[ -z "$fixed_baseline" ]]; then
  fixed_baseline="$(python -c 'from scripts.render import resolve_models; import sys; print(resolve_models(None, sys.argv[1], ["sharp_keyframes_stride3_aa"])[0])' "$fixed_root")"
fi
fixed_model="${NATIVE_LIDAR_MODEL:-$fixed_experiments/3dgs_stride3_aa_fixed_lidar_large_scene_200k_lateprune_v2}"
python -u scripts/run_native_lidar.py prepare \
  --source "$fixed_source" --baseline "$fixed_baseline" \
  --point-cloud "${NATIVE_LIDAR_POINTS:-$fixed_baseline/input.ply}" \
  --model "$fixed_model" --gpu "$fixed_gpu" --resolution 1 \
  --iterations 200000 --freeze-lidar-positions --scaling-lr 0.001 \
  --densify-from-iter 100000 --densify-until-iter 180000 --densification-interval 2000 \
  --pruning --prune-world-scale-ratio 0.01 --prune-screen-size 0 \
  --prune-size-from-iter 100000 --preserve-densification-radii \
  --checkpoint-every 10000 --expected-frames 5009 \
  --debug-preview-every 500 --debug-preview-train 2 --debug-preview-test 2
exec bash scripts/run_prepared_reconstruction.sh "$fixed_model"
