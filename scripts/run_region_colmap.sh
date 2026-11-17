#!/usr/bin/env bash
# Fresh train-only COLMAP, align to LiDAR, then the matching regional GS control.
set -euo pipefail
source "${RECONSTRUCTION_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
conda activate 3dgs
cd "$(dirname "${BASH_SOURCE[0]}")/.."
gpu="${1:-1}"
region="${2:-/data/3dgs_outputs/orange_4F/1_full/3dgs_finetuning/regions/wall_06663_v1}"
version="${3:-v1}"
if (($#>3)) || [[ ! "$version" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo 'Usage: bash scripts/run_region_colmap.sh GPU REGION VERSION' >&2; exit 2
fi
region="$(realpath -e "$region")"
job="$region/colmap_train_$version"
model="$region/runs/crop_colmap_fixed_$version"
colmap="${COLMAP_BIN:-colmap}"
command -v "$colmap" >/dev/null || { echo 'COLMAP executable not found. Set COLMAP_BIN to your existing COLMAP binary.' >&2; exit 1; }
if [[ -e "$model" ]]; then
  echo "Model already exists: $model. Use a new VERSION for a fresh experiment." >&2; exit 1
fi
gpu_pids="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)"
if [[ -n "$gpu_pids" ]]; then echo "GPU $gpu is occupied; no existing process was stopped." >&2; exit 1; fi

python -u scripts/region_colmap.py prepare --region "$region" --workspace "$job"
exec 9>"$job/run.lock"
flock -n 9 || { echo "COLMAP workspace is already in use: $job" >&2; exit 1; }
# Limit visibility for COLMAP only. The training plan retains the physical GPU ID.
colmap_gpu() {
  CUDA_VISIBLE_DEVICES="$gpu" QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=8 "$colmap" "$@"
}
run_stage() {
  local log="$1"; shift
  printf 'COLMAP stage: %s\n' "$1"
  colmap_gpu "$@" 2>&1 | tee "$job/logs/$log.log"
}
for command in feature_extractor exhaustive_matcher mapper bundle_adjuster model_aligner; do
  colmap_gpu "$command" -h >"$job/logs/${command}_help.txt" 2>&1
done
# COLMAP renamed these option groups; use the installed executable's help.
feature_help="$job/logs/feature_extractor_help.txt"
matching_help="$job/logs/exhaustive_matcher_help.txt"
if grep -q -- '--FeatureExtraction.use_gpu' "$feature_help"; then extraction=FeatureExtraction; else extraction=SiftExtraction; fi
if grep -q -- '--FeatureMatching.use_gpu' "$matching_help"; then matching=FeatureMatching; else matching=SiftMatching; fi
require_option() {
  grep -q -- "--$2" "$1" || { echo "Installed COLMAP lacks required option $2 (see $1)." >&2; exit 1; }
}
for option in "$extraction.use_gpu" "$extraction.gpu_index" "$extraction.max_image_size" ImageReader.existing_camera_id image_list_path; do
  require_option "$feature_help" "$option"
done
for option in "$matching.use_gpu" "$matching.gpu_index"; do require_option "$matching_help" "$option"; done
for option in focal_length principal_point extra_params; do
  require_option "$job/logs/mapper_help.txt" "Mapper.ba_refine_$option"
  require_option "$job/logs/bundle_adjuster_help.txt" "BundleAdjustment.refine_$option"
done
for option in ref_images_path ref_is_gps alignment_type alignment_max_error transform_path; do
  require_option "$job/logs/model_aligner_help.txt" "$option"
done

if [[ ! -e "$job/database.db" ]]; then run_stage database database_creator --database_path "$job/database.db"; fi
python scripts/region_colmap.py seed-cameras --workspace "$job"
if [[ ! -e "$job/features.done" ]]; then
  for list in "$job"/lists/camera_*.txt; do
    camera="${list##*/camera_}"; camera="${camera%.txt}"
    run_stage "features_$camera" feature_extractor --database_path "$job/database.db" \
      --image_path "$job/images" --image_list_path "$list" --ImageReader.existing_camera_id "$camera" \
      "--$extraction.use_gpu" 1 "--$extraction.gpu_index" 0 "--$extraction.max_image_size" 3200
  done
  touch "$job/features.done"
fi
if [[ ! -e "$job/matching.done" ]]; then
  run_stage matching exhaustive_matcher --database_path "$job/database.db" \
    "--$matching.use_gpu" 1 "--$matching.gpu_index" 0
  touch "$job/matching.done"
fi
if [[ ! -e "$job/mapping.done" ]]; then
  # Each failed attempt stays available for inspection; it is never a new seed.
  mapping_root="$(mktemp -d "$job/sparse_attempt_XXXXXX")"
  run_stage mapping mapper --database_path "$job/database.db" --image_path "$job/images" \
    --output_path "$mapping_root" --Mapper.ba_refine_focal_length 0 \
    --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0
  printf '%s\n' "$mapping_root" >"$job/mapping_root.txt"
  touch "$job/mapping.done"
fi
mapping_root="$(cat "$job/mapping_root.txt")"
python -u scripts/region_colmap.py choose-model --workspace "$job" --mapping-root "$mapping_root"
selected_model="$(cat "$job/selected_model.txt")"
if [[ ! -e "$job/ba.done" ]]; then
  mkdir -p "$job/refined"
  run_stage bundle_adjustment bundle_adjuster --input_path "$selected_model" --output_path "$job/refined" \
    --BundleAdjustment.refine_focal_length 0 --BundleAdjustment.refine_principal_point 0 \
    --BundleAdjustment.refine_extra_params 0
  touch "$job/ba.done"
fi
if [[ ! -e "$job/alignment.done" ]]; then
  tolerance="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["alignment_max_error"])' "$job/job.json")"
  mkdir -p "$job/aligned"
  run_stage alignment model_aligner --input_path "$job/refined" --output_path "$job/aligned" \
    --ref_images_path "$job/reference_centers.txt" --ref_is_gps 0 --alignment_type custom \
    --alignment_max_error "$tolerance" --transform_path "$job/similarity.txt"
  touch "$job/alignment.done"
fi
python -u scripts/region_colmap.py export --workspace "$job"
flock -u 9
exec 9>&-
python -u scripts/run_region_ablation.py --region "$job/dataset" --context crop \
  --geometry fixed --model "$model" --gpu "$gpu"
exec bash scripts/run_prepared_reconstruction.sh "$model"
