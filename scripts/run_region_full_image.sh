#!/usr/bin/env bash
# Fresh GPU 0 regional restart. Retain previous results and stop only its named run.
set -euo pipefail
if (($# < 2 || $# > 4)); then
  echo 'Usage: bash scripts/run_region_full_image.sh GPU REGION [VERSION=v1] [PREVIOUS_MODEL=crop_v2]' >&2; exit 2
fi
source "${RECONSTRUCTION_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
conda activate 3dgs
cd "$(dirname "${BASH_SOURCE[0]}")/.."
full_gpu="$1"
full_region="$(realpath -e "$2")"
full_version="${3:-v1}"
if [[ "$full_version" =~ [^a-zA-Z0-9_-] ]]; then echo 'Invalid version' >&2; exit 2; fi
full_model="$full_region/runs/stride3_fullimage_fixed_lidar_native_lr_resets_$full_version"
full_previous="${4:-$full_region/runs/crop_v2}"
if [[ -e "$full_model" ]]; then echo "Model exists: $full_model; use a fresh VERSION." >&2; exit 1; fi
# Validate and score the selected regional images before stopping the old run.
python -u scripts/run_region_full_image.py --region "$full_region" --model "$full_model" --gpu "$full_gpu"
if [[ -f "$full_previous/periodic_evaluation/plan.json" ]]; then
  python scripts/stop_reconstruction_run.py --model "$full_previous" --gpu "$full_gpu" --stop
fi
echo "Prepared $full_model. Original images/poses and previous outputs are retained."
exec bash scripts/run_prepared_reconstruction.sh "$full_model"
