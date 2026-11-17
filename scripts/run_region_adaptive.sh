#!/usr/bin/env bash
# Additional geometry ablation: GPU REGION [original|colmap] [VERSION].
set -euo pipefail
if (($# < 2 || $# > 4)); then
  echo 'Usage: bash scripts/run_region_adaptive.sh GPU REGION [original|colmap] [VERSION=v1]' >&2
  exit 2
fi
source "${RECONSTRUCTION_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
conda activate 3dgs
cd "$(dirname "${BASH_SOURCE[0]}")/.."
region_gpu="$1"
region_dir="$(realpath -e "$2")"
pose_source="${3:-original}"
region_version="${4:-v1}"
colmap_version="${COLMAP_VERSION:-v1}"
colmap_overlap="${COLMAP_SEQUENTIAL_OVERLAP:-20}"
if [[ "$region_version" =~ [^a-zA-Z0-9_-] || "$colmap_version" =~ [^a-zA-Z0-9_-] || ! "$colmap_overlap" =~ ^[1-9][0-9]{0,3}$ ]]; then
  echo 'Invalid version or COLMAP overlap.' >&2; exit 2
fi
if [[ "$pose_source" == original ]]; then
  region_source="$region_dir"
  region_model="$region_dir/runs/crop_original_adaptive_lr005_$region_version"
elif [[ "$pose_source" == colmap ]]; then
  colmap_workspace="$region_dir/colmap_superpoint_superglue_seq${colmap_overlap}_$colmap_version"
  region_source="$colmap_workspace/dataset"
  region_model="$region_dir/runs/crop_colmap_superglue_seq${colmap_overlap}_adaptive_lr005_$region_version"
  if [[ ! -f "$region_source/colmap_rerun.json" ]]; then
    echo "COLMAP dataset is not ready: $region_source" >&2
    echo 'Finish/resume run_region_superglue.sh on GPU 1, then rerun this command. No training folder was created.' >&2
    exit 1
  fi
  # Recheck accepted alignment and exported files; do not reuse a partial export.
  python -u scripts/region_colmap.py export --workspace "$colmap_workspace"
else
  echo 'Pose source must be original or colmap.' >&2; exit 2
fi
if [[ -e "$region_model" ]]; then
  echo "Model already exists: $region_model; use a new VERSION for a fresh run." >&2; exit 1
fi
if [[ -n "$(nvidia-smi -i "$region_gpu" --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "GPU $region_gpu is occupied; no existing process was stopped." >&2; exit 1
fi
python -u scripts/run_region_ablation.py --region "$region_source" --context crop \
  --geometry adaptive --position-lr-multiplier 0.05 --iterations 90000 \
  --model "$region_model" --gpu "$region_gpu"
exec bash scripts/run_prepared_reconstruction.sh "$region_model"
