#!/usr/bin/env bash
# Fifth run: original poses + adaptive geometry + a LiDAR SCP prior.
set -euo pipefail
if (($# < 2 || $# > 3)); then
  echo 'Usage: bash scripts/run_region_scp.sh GPU REGION [VERSION=v1]' >&2; exit 2
fi
source "${RECONSTRUCTION_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
conda activate 3dgs
cd "$(dirname "${BASH_SOURCE[0]}")/.."
scp_gpu="$1"
scp_region="$(realpath -e "$2")"
scp_version="${3:-v1}"
if [[ "$scp_version" =~ [^a-zA-Z0-9_-] ]]; then echo 'Invalid version' >&2; exit 2; fi
scp_model="$scp_region/runs/crop_original_adaptive_lr005_lidar_scp_$scp_version"
scp_cache="$scp_region/lidar_scp_v1"
if [[ -e "$scp_model" ]]; then echo "Model already exists: $scp_model; use a new VERSION." >&2; exit 1; fi
if [[ -n "$(nvidia-smi -i "$scp_gpu" --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "GPU $scp_gpu is occupied; no existing process was stopped." >&2; exit 1
fi
python -u scripts/prepare_lidar_scp.py --point-cloud "$scp_region/input.ply" --output "$scp_cache" --workers 8
python -u scripts/run_region_ablation.py --region "$scp_region" --context crop \
  --geometry adaptive --position-lr-multiplier 0.05 --iterations 90000 \
  --lidar-scp-cache "$scp_cache" --model "$scp_model" --gpu "$scp_gpu"
exec bash scripts/run_prepared_reconstruction.sh "$scp_model"
