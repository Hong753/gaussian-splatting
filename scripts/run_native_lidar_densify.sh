#!/usr/bin/env bash
# GPU 1 counterpart to run_fixed_lidar_unconstrained.sh, with moving XYZ and growth.
set -euo pipefail
if (($#>1)); then echo 'Usage: bash scripts/run_native_lidar_densify.sh [GPU]' >&2; exit 2; fi
source "${RECONSTRUCTION_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
conda activate 3dgs
cd "$(dirname "${BASH_SOURCE[0]}")/.."
native_gpu="${1:-1}"
if [[ -n "$(nvidia-smi -i "$native_gpu" --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "GPU $native_gpu is occupied; the new run was not started." >&2; exit 1
fi
native_root="${NATIVE_LIDAR_OUTPUT_ROOT:-/data/3dgs_outputs/orange_4F/1_full}"
native_source="${NATIVE_LIDAR_SOURCE:-/data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full}"
native_baseline="${NATIVE_LIDAR_BASELINE:-}"
if [[ -z "$native_baseline" ]]; then
  native_baseline="$(python -c 'from scripts.render import resolve_models; import sys; print(resolve_models(None, sys.argv[1], ["sharp_keyframes_stride3_aa"])[0])' "$native_root")"
fi
native_model="${NATIVE_LIDAR_MODEL:-$native_root/3dgs_stride3_aa_lidar_unconstrained_lr50_densify_noresets_noprune_30000_v1}"
python -u scripts/run_native_lidar.py prepare \
  --source "$native_source" --baseline "$native_baseline" \
  --point-cloud "${NATIVE_LIDAR_POINTS:-$native_baseline/input.ply}" \
  --model "$native_model" --gpu "$native_gpu" \
  --iterations 30000 --position-lr-divisor 50 \
  --checkpoint-every 10000 --expected-frames 5009 \
  --debug-preview-every 500 --debug-preview-train 2 --debug-preview-test 2
exec bash scripts/run_prepared_reconstruction.sh "$native_model"
