#!/usr/bin/env bash
# Export automatic regional candidates, then train 3DGS + AA with adaptive geometry.
set -euo pipefail
if (($#>3)); then
  echo 'Usage: bash scripts/run_region_auto.sh [GPU] [REGION] [VERSION]' >&2; exit 2
fi
source "${RECONSTRUCTION_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
conda activate 3dgs
cd "$(dirname "${BASH_SOURCE[0]}")/.."
region_gpu="${1:-0}"
region_root="${REGION_OUTPUT_ROOT:-/data/3dgs_outputs/orange_4F/1_full}"
region_dir="${2:-$region_root/3dgs_finetuning/regions/wall_06663_auto_v1}"
region_version="${3:-v1}"
if [[ ! "$region_version" =~ ^[a-zA-Z0-9_-]+$ ]]; then echo 'Invalid version label' >&2; exit 2; fi
region_model="$region_dir/runs/crop_adaptive_lr05_${region_version}"
if [[ -e "$region_model" ]]; then
  echo "Model directory already exists: $region_model. Supply a new VERSION (e.g. v2)." >&2; exit 1
fi
region_active="$(nvidia-smi -i "$region_gpu" --query-compute-apps=pid --format=csv,noheader)"
if [[ -n "$region_active" ]]; then
  echo "GPU $region_gpu is occupied. No existing process was stopped." >&2; exit 1
fi
region_source="${REGION_SOURCE:-/data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full}"
region_baseline="${REGION_BASELINE:-}"
if [[ -z "$region_baseline" ]]; then
  region_baseline="$(python -c 'from scripts.render import resolve_models; import sys; print(resolve_models(None, sys.argv[1], ["sharp_keyframes_stride3_aa"])[0])' "$region_root")"
fi
python -u scripts/select_region.py --source "$region_source" --baseline "$region_baseline" \
  --output "$region_dir" --anchor 06663.jpg --size 4 4 4 --halo 1 \
  --automatic --export --reuse
python -u scripts/run_region_ablation.py --region "$region_dir" --context crop \
  --model "$region_model" --gpu "$region_gpu" --geometry adaptive --position-lr-multiplier .05
exec bash scripts/run_prepared_reconstruction.sh "$region_model"
