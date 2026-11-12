#!/usr/bin/env bash
# Keep GPU 3's 1x run; replace the old GPU 0 clamp run with a matched cap trial.
set -euo pipefail
if (($# > 1)); then
  echo 'Usage: bash scripts/run_local_scale_tune.sh [GPU]' >&2
  exit 2
fi
source "${LOCAL_SCALE_TUNE_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
conda activate 3dgs
cd "$(dirname "${BASH_SOURCE[0]}")/.."
tune_gpu="${1:-0}"
tune_root="${LOCAL_SCALE_TUNE_OUTPUT_ROOT:-/data/3dgs_outputs/orange_4F/1_full}"
tune_reference="${LOCAL_SCALE_TUNE_REFERENCE:-$tune_root/3dgs_stride3_aa_fixed_lidar_local_knn10_100k_v1}"
tune_previous="${LOCAL_SCALE_TUNE_STOP_MODEL:-$tune_root/3dgs_stride3_aa_fixed_lidar_scale2_noprune_100k_v1}"
tune_ceiling="${LOCAL_SCALE_TUNE_CEILING:-2}"
tune_target="${LOCAL_SCALE_TUNE_TARGET:-1}"
tune_iterations="${LOCAL_SCALE_TUNE_ITERATIONS:-30000}"
tune_schedule="${LOCAL_SCALE_TUNE_SCHEDULE:-off}"
tune_suffix=''
if [[ "$tune_schedule" != off ]]; then tune_suffix="_${tune_schedule}"; fi
if [[ "${LOCAL_SCALE_TUNE_POSES:-0}" == 1 ]]; then tune_suffix+='_poses'; fi
tune_model="${LOCAL_SCALE_TUNE_MODEL:-$tune_root/3dgs_stride3_aa_fixed_lidar_local_cap${tune_ceiling}_target${tune_target}${tune_suffix}_${tune_iterations}_v1}"
tune_schedule_args=(--schedule-mode "$tune_schedule"
  --scale-start-multiplier "${LOCAL_SCALE_TUNE_START_CEILING:-1}"
  --scale-ramp-start "${LOCAL_SCALE_TUNE_RAMP_START:-7000}"
  --scale-ramp-end "${LOCAL_SCALE_TUNE_RAMP_END:-20000}"
  --lr-decay-start "${LOCAL_SCALE_TUNE_LR_START:-7000}"
  --lr-decay-end "${LOCAL_SCALE_TUNE_LR_END:-$tune_iterations}"
  --lr-scaling "${LOCAL_SCALE_TUNE_LR_SCALE:-0.001}"
  --lr-scaling-final "${LOCAL_SCALE_TUNE_LR_SCALE_FINAL:-0.0001}"
  --lr-opacity "${LOCAL_SCALE_TUNE_LR_OPACITY:-0.01}"
  --lr-opacity-final "${LOCAL_SCALE_TUNE_LR_OPACITY_FINAL:-0.001}"
  --lr-rotation "${LOCAL_SCALE_TUNE_LR_ROTATION:-0.0005}"
  --lr-rotation-final "${LOCAL_SCALE_TUNE_LR_ROTATION_FINAL:-0.00005}"
  --lr-f-dc "${LOCAL_SCALE_TUNE_LR_DC:-0.0025}"
  --lr-f-dc-final "${LOCAL_SCALE_TUNE_LR_DC_FINAL:-0.00025}"
  --lr-f-rest "${LOCAL_SCALE_TUNE_LR_REST:-0.000125}"
  --lr-f-rest-final "${LOCAL_SCALE_TUNE_LR_REST_FINAL:-0.0000125}")
if [[ "${LOCAL_SCALE_TUNE_POSES:-0}" == 1 ]]; then
  # A new pose experiment must not stop or share a device with an existing run.
  tune_device_pids="$(nvidia-smi -i "$tune_gpu" --query-compute-apps=pid --format=csv,noheader)"
  if [[ -n "$tune_device_pids" ]]; then
    echo "GPU $tune_gpu already has a compute process; choose a free GPU for this additional run." >&2
    exit 1
  fi
  tune_schedule_args+=(--pose-refinement
    --pose-start "${POSE_REFINE_START:-7000}" --pose-end "${POSE_REFINE_END:-20000}"
    --pose-every "${POSE_REFINE_EVERY:-1}" --pose-max-translation "${POSE_REFINE_MAX_TRANSLATION:-0.03}"
    --pose-max-rotation-degrees "${POSE_REFINE_MAX_ROTATION_DEGREES:-0.5}"
    --pose-translation-lr "${POSE_REFINE_TRANSLATION_LR:-0.001}"
    --pose-rotation-lr "${POSE_REFINE_ROTATION_LR:-0.0002}"
    --pose-prior-weight "${POSE_REFINE_PRIOR_WEIGHT:-0.001}" --pose-max-width "${POSE_REFINE_MAX_WIDTH:-960}")
fi
if [[ "${LOCAL_SCALE_TUNE_OFFSETS:-0}" == 1 ]]; then
  if [[ "${LOCAL_SCALE_TUNE_POSES:-0}" != 1 ]]; then
    echo 'Bounded LiDAR offsets require the alternating-pose experiment.' >&2
    exit 2
  fi
  tune_schedule_args+=(--lidar-offsets
    --offset-max-world "${LIDAR_OFFSET_MAX_WORLD:-0.005}"
    --offset-sigma-ratio "${LIDAR_OFFSET_SIGMA_RATIO:-0.25}"
    --offset-warmup "${LIDAR_OFFSET_WARMUP:-7000}" --offset-ramp "${LIDAR_OFFSET_RAMP:-2000}"
    --offset-lr-init "${LIDAR_OFFSET_LR_INIT:-0.00001}" --offset-lr-final "${LIDAR_OFFSET_LR_FINAL:-0.000001}"
    --offset-prior-weight "${LIDAR_OFFSET_PRIOR_WEIGHT:-0.001}")
fi
python -u scripts/tune_local_scale.py --model "$tune_reference"
python -u scripts/tune_local_scale.py prepare \
  --reference-model "$tune_reference" --model "$tune_model" --gpu "$tune_gpu" \
  --ceiling-multiplier "$tune_ceiling" --target-multiplier "$tune_target" --iterations "$tune_iterations" \
  "${tune_schedule_args[@]}"
if [[ "${LOCAL_SCALE_TUNE_POSES:-0}" != 1 ]]; then
  python -u scripts/stop_reconstruction_run.py --model "$tune_previous" --gpu "$tune_gpu" --stop
fi
tune_eval_pid=''
tune_train_pid=''
tune_cleanup() {
  if [[ -n "$tune_train_pid" ]]; then
    kill "$tune_train_pid" 2>/dev/null || true
    wait "$tune_train_pid" 2>/dev/null || true
  fi
  if [[ -n "$tune_eval_pid" ]]; then
    kill "$tune_eval_pid" 2>/dev/null || true
    wait "$tune_eval_pid" 2>/dev/null || true
  fi
}
trap tune_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python -u scripts/run_local_scale.py watch --model "$tune_model" --parent-pid "$$" \
  >"$tune_model/periodic_evaluation/evaluator.log" 2>&1 &
tune_eval_pid=$!
echo "Evaluator PID: $tune_eval_pid; GPU: $tune_gpu; log: $tune_model/periodic_evaluation/evaluator.log"
python -u scripts/run_local_scale.py train --model "$tune_model" > >(tee "$tune_model/training.log") 2>&1 &
tune_train_pid=$!
if wait "$tune_train_pid"; then
  tune_train_pid=''
  echo 'Training finished. Waiting for all queued all-frame evaluations...'
else
  tune_train_rc=$?
  tune_train_pid=''
  echo 'Training failed; inspect training.log. Saved outputs are retained.' >&2
  exit "$tune_train_rc"
fi
tune_eval_rc=0
wait "$tune_eval_pid" || tune_eval_rc=$?
tune_eval_pid=''
exit "$tune_eval_rc"
