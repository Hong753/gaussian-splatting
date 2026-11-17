#!/usr/bin/env bash
# GPU REGION [crop|full] [VERSION]; native training plus independent evaluator.
set -euo pipefail
if (($#<2 || $#>4)); then
  echo 'Usage: bash scripts/run_region_ablation.sh GPU REGION [crop|full] [VERSION]' >&2; exit 2
fi
source "${RECONSTRUCTION_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
conda activate 3dgs
cd "$(dirname "${BASH_SOURCE[0]}")/.."
region_gpu="$1"
region_dir="$(realpath "$2")"
region_context="${3:-crop}"
region_version="${4:-v1}"
if [[ "$region_version" =~ [^a-zA-Z0-9_-] ]]; then echo 'Invalid version label' >&2; exit 2; fi
region_model="$region_dir/runs/${region_context}_${region_version}"
if [[ -n "$(nvidia-smi -i "$region_gpu" --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "GPU $region_gpu is occupied. No existing process was stopped." >&2; exit 1
fi
python -u scripts/run_region_ablation.py --region "$region_dir" --context "$region_context" \
  --model "$region_model" --gpu "$region_gpu"
exec bash scripts/run_prepared_reconstruction.sh "$region_model"
