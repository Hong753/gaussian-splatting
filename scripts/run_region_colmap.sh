#!/usr/bin/env bash
# Fresh train-only COLMAP, align to LiDAR, then the matching regional GS control.
set -euo pipefail
source "${RECONSTRUCTION_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
conda activate 3dgs
cd "$(dirname "${BASH_SOURCE[0]}")/.."
gpu="${1:-1}"
region="${2:-/data/3dgs_outputs/orange_4F/1_full/3dgs_finetuning/regions/wall_06663_v1}"
version="${3:-v1}"
frontend="${REGION_COLMAP_FRONTEND:-sift}"
overlap="${COLMAP_SEQUENTIAL_OVERLAP:-20}"
bridge_rounds="${SUPERGLUE_BRIDGE_ROUNDS:-3}"
bridge_budget="${SUPERGLUE_BRIDGE_BUDGET:-512}"
if [[ ! "$bridge_rounds" =~ ^[0-9]{1,2}$ || ! "$bridge_budget" =~ ^[1-9][0-9]{0,3}$ ]]; then
  echo 'SUPERGLUE_BRIDGE_ROUNDS must be 0..99; SUPERGLUE_BRIDGE_BUDGET must be 1..9999.' >&2; exit 2
fi
bridge_rounds=$((10#$bridge_rounds))
if [[ "$frontend" != sift && "$frontend" != superglue ]] || [[ ! "$overlap" =~ ^[1-9][0-9]{0,3}$ ]]; then
  echo 'Frontend must be sift or superglue; sequential overlap must be a positive integer.' >&2; exit 2
fi
if (($#>3)) || [[ ! "$version" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo 'Usage: bash scripts/run_region_colmap.sh GPU REGION VERSION' >&2; exit 2
fi
region="$(realpath -e "$region")"
job="$region/colmap_train_$version"
model="$region/runs/crop_colmap_fixed_$version"
if [[ "$frontend" == superglue ]]; then
  job="$region/colmap_superpoint_superglue_seq${overlap}_$version"
  model="$region/runs/crop_colmap_superglue_seq${overlap}_fixed_$version"
fi
colmap="${COLMAP_BIN:-colmap}"
command -v "$colmap" >/dev/null || { echo 'COLMAP executable not found. Set COLMAP_BIN to your existing COLMAP binary.' >&2; exit 1; }
if [[ -e "$model" ]]; then
  echo "Model already exists: $model. Use a new VERSION for a fresh experiment." >&2; exit 1
fi
gpu_pids="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)"
if [[ -n "$gpu_pids" ]]; then echo "GPU $gpu is occupied; no existing process was stopped." >&2; exit 1; fi

python -u scripts/region_colmap.py prepare --region "$region" --workspace "$job" \
  --frontend "$frontend" --sequential-overlap "$overlap"
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
echo 'Checking COLMAP options; reconstruction and GS training have not started yet.'
feature_command=feature_extractor
matching_command=exhaustive_matcher
if [[ "$frontend" == superglue ]]; then feature_command=feature_importer; matching_command=matches_importer; fi
for command in "$feature_command" "$matching_command" mapper bundle_adjuster model_aligner; do
  colmap_gpu "$command" -h >"$job/logs/${command}_help.txt" 2>&1
done
# COLMAP renamed these option groups; use the installed executable's help.
feature_help="$job/logs/${feature_command}_help.txt"
matching_help="$job/logs/${matching_command}_help.txt"
alignment_help="$job/logs/model_aligner_help.txt"
if grep -q -- '--FeatureExtraction.use_gpu' "$feature_help"; then extraction=FeatureExtraction; else extraction=SiftExtraction; fi
if grep -q -- '--FeatureMatching.use_gpu' "$matching_help"; then matching=FeatureMatching; else matching=SiftMatching; fi
require_option() {
  grep -q -- "--$2" "$1" || { echo "Installed COLMAP lacks required option $2 (see $1)." >&2; exit 1; }
}
for option in ImageReader.existing_camera_id image_list_path; do
  require_option "$feature_help" "$option"
done
require_option "$matching_help" "$matching.use_gpu"
if [[ "$frontend" == sift ]]; then
  for option in "$extraction.use_gpu" "$extraction.gpu_index" "$extraction.max_image_size"; do require_option "$feature_help" "$option"; done
  require_option "$matching_help" "$matching.gpu_index"
else
  require_option "$feature_help" import_path
  for option in match_list_path match_type; do require_option "$matching_help" "$option"; done
fi
for option in focal_length principal_point extra_params; do
  require_option "$job/logs/mapper_help.txt" "Mapper.ba_refine_$option"
  require_option "$job/logs/bundle_adjuster_help.txt" "BundleAdjustment.refine_$option"
done
for option in ref_images_path ref_is_gps alignment_type transform_path; do
  require_option "$alignment_help" "$option"
done
alignment_extra=()
if grep -q -- '--alignment_max_error' "$alignment_help"; then
  alignment_error_option=alignment_max_error
elif grep -q -- '--robust_alignment_max_error' "$alignment_help"; then
  alignment_error_option=robust_alignment_max_error
else
  echo "Stopped before reconstruction: COLMAP exposes neither alignment_max_error nor robust_alignment_max_error (see $alignment_help)." >&2
  exit 1
fi
if grep -Eq -- '--robust_alignment([[:space:]=]|$)' "$alignment_help"; then
  alignment_extra+=(--robust_alignment 1)
fi
printf 'COLMAP alignment tolerance option: --%s\n' "$alignment_error_option"

if [[ ! -e "$job/database.db" ]]; then run_stage database database_creator --database_path "$job/database.db"; fi
python scripts/region_colmap.py seed-cameras --workspace "$job"
if [[ "$frontend" == superglue ]]; then
  CUDA_VISIBLE_DEVICES="$gpu" python -c 'import torch; assert torch.cuda.is_available(), "SuperPoint/SuperGlue needs CUDA in the active 3dgs environment"; print("Learned frontend:", torch.cuda.get_device_name(0), flush=True)'
  superglue_repo="${SUPERGLUE_REPO:-${XDG_CACHE_HOME:-$HOME/.cache}/3dgs/SuperGluePretrainedNetwork-ddcf11f}"
  bash scripts/setup_superglue.sh "$superglue_repo"
  python -u scripts/region_superglue.py prepare --workspace "$job" --repository "$superglue_repo" \
    --max-size "${SUPERPOINT_MAX_SIZE:-1600}" --max-keypoints "${SUPERPOINT_MAX_KEYPOINTS:-4096}"
  if [[ ! -e "$job/features.done" ]]; then
    python -u scripts/region_superglue.py extract --workspace "$job" --gpu "$gpu" 2>&1 | tee "$job/logs/superpoint.log"
    for list in "$job"/lists/camera_*.txt; do
      camera="${list##*/camera_}"; camera="${camera%.txt}"
      run_stage "image_import_$camera" feature_importer --database_path "$job/database.db" \
        --image_path "$job/images" --image_list_path "$list" --ImageReader.existing_camera_id "$camera" \
        --import_path "$job/image_registration"
    done
    python -u scripts/region_superglue.py import-keypoints --workspace "$job"
    touch "$job/features.done"
  fi
  if [[ ! -e "$job/matching.done" ]]; then
    verify_pending() {
      local batch="$1"
      python -u scripts/region_superglue.py match --workspace "$job" --gpu "$gpu" --pending-only \
        2>&1 | tee "$job/logs/superglue_$batch.log"
      # Feed only missing geometry rows, including cached matches from an
      # interrupted batch. Completed rejected/accepted pairs are never rerun.
      if [[ -s "$job/superglue_raw_matches.txt" ]]; then
        run_stage "verification_$batch" matches_importer --database_path "$job/database.db" \
          --match_list_path "$job/superglue_raw_matches.txt" --match_type raw "--$matching.use_gpu" 0
      fi
      python -u scripts/region_superglue.py audit-verified --workspace "$job"
    }
    verify_pending resume
    for ((bridge_attempt=0; bridge_attempt<bridge_rounds; bridge_attempt++)); do
      connected="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["connected"])' "$job/verification_report.json")"
      if [[ "$connected" == True ]]; then break; fi
      python -u scripts/region_superglue.py plan-bridges --workspace "$job" --bridge-budget "$bridge_budget"
      bridge_id="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["rounds"]))' "$job/bridge_pairs.json")"
      verify_pending "bridge_$bridge_id"
    done
    python -u scripts/region_superglue.py check-verified --workspace "$job"
    touch "$job/matching.done"
  fi
else
  # Non-CUDA COLMAP builds use OpenGL for use_gpu=1; offscreen does not make
  # that headless-safe. Only select GPU SIFT when the binary advertises CUDA.
  colmap_gpu -h >"$job/logs/build_help.txt" 2>&1
  sift_device="${COLMAP_SIFT_DEVICE:-auto}"
  case "$sift_device" in
    cpu) sift_gpu=0 ;;
    auto) if grep -Eq 'with CUDA' "$job/logs/build_help.txt"; then sift_gpu=1; else sift_gpu=0; fi ;;
    cuda) grep -Eq 'with CUDA' "$job/logs/build_help.txt" || { echo 'COLMAP does not advertise a CUDA build.' >&2; exit 1; }; sift_gpu=1 ;;
    *) echo 'COLMAP_SIFT_DEVICE must be auto, cpu or cuda.' >&2; exit 2 ;;
  esac
  printf 'SIFT extraction/matching use_gpu=%s (GS training remains on GPU %s).\n' "$sift_gpu" "$gpu"
  extraction_threads=(); matching_threads=()
  if grep -q -- "--$extraction.num_threads" "$feature_help"; then extraction_threads=("--$extraction.num_threads" 8); fi
  if grep -q -- "--$matching.num_threads" "$matching_help"; then matching_threads=("--$matching.num_threads" 8); fi
if [[ ! -e "$job/features.done" ]]; then
  for list in "$job"/lists/camera_*.txt; do
    camera="${list##*/camera_}"; camera="${camera%.txt}"
    run_stage "features_$camera" feature_extractor --database_path "$job/database.db" \
      --image_path "$job/images" --image_list_path "$list" --ImageReader.existing_camera_id "$camera" \
      "--$extraction.use_gpu" "$sift_gpu" "--$extraction.gpu_index" 0 "--$extraction.max_image_size" 3200 "${extraction_threads[@]}"
  done
  touch "$job/features.done"
fi
if [[ ! -e "$job/matching.done" ]]; then
  run_stage matching exhaustive_matcher --database_path "$job/database.db" \
    "--$matching.use_gpu" "$sift_gpu" "--$matching.gpu_index" 0 "${matching_threads[@]}"
  touch "$job/matching.done"
fi
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
    "--$alignment_error_option" "$tolerance" "${alignment_extra[@]}" --transform_path "$job/similarity.txt"
  touch "$job/alignment.done"
fi
python -u scripts/region_colmap.py export --workspace "$job"
printf 'COLMAP preprocessing completed; preparing Gaussian training on GPU %s.\n' "$gpu"
flock -u 9
exec 9>&-
python -u scripts/run_region_ablation.py --region "$job/dataset" --context crop \
  --geometry fixed --model "$model" --gpu "$gpu"
exec bash scripts/run_prepared_reconstruction.sh "$model"
