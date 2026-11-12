#!/usr/bin/env bash
# Validate the replacement, stop only the old named run, train + evaluate on GPU 0.
set -euo pipefail
if (($# > 1)); then
  echo 'Usage: bash scripts/run_distance_weighted.sh [GPU]' >&2
  exit 2
fi
source "${DISTANCE_UPDATE_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
conda activate 3dgs
cd "$(dirname "${BASH_SOURCE[0]}")/.."
distance_gpu="${1:-0}"
distance_root="${DISTANCE_UPDATE_OUTPUT_ROOT:-/data/3dgs_outputs/orange_4F/1_full}"
distance_previous="${DISTANCE_UPDATE_PREVIOUS_MODEL:-$distance_root/3dgs_stride3_aa_fixed_lidar_scale2_noprune_100k_v1}"
distance_model="${DISTANCE_UPDATE_MODEL:-$distance_root/3dgs_stride3_aa_fixed_lidar_scale2_distance_100k_v1}"
python -u scripts/run_distance_weighted.py prepare \
  --previous-model "$distance_previous" --model "$distance_model" --gpu "$distance_gpu" \
  --reference "${DISTANCE_UPDATE_REFERENCE:-2.0}" --power "${DISTANCE_UPDATE_POWER:-2.0}" \
  --min-weight "${DISTANCE_UPDATE_MIN_WEIGHT:-0.05}" \
  --chunk-size "${DISTANCE_UPDATE_CHUNK_SIZE:-131072}"
python -u scripts/stop_reconstruction_run.py --model "$distance_previous" --gpu "$distance_gpu" --stop
distance_eval_pid=''
distance_train_pid=''
distance_cleanup() {
  if [[ -n "$distance_train_pid" ]]; then
    kill "$distance_train_pid" 2>/dev/null || true
    wait "$distance_train_pid" 2>/dev/null || true
  fi
  if [[ -n "$distance_eval_pid" ]]; then
    kill "$distance_eval_pid" 2>/dev/null || true
    wait "$distance_eval_pid" 2>/dev/null || true
  fi
}
trap distance_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python -u scripts/run_distance_weighted.py watch --model "$distance_model" --parent-pid "$$" \
  >"$distance_model/periodic_evaluation/evaluator.log" 2>&1 &
distance_eval_pid=$!
echo "Evaluator PID: $distance_eval_pid; GPU: $distance_gpu; log: $distance_model/periodic_evaluation/evaluator.log"
python -u scripts/run_distance_weighted.py train --model "$distance_model" \
  > >(tee "$distance_model/training.log") 2>&1 &
distance_train_pid=$!
if wait "$distance_train_pid"; then
  distance_train_pid=''
  echo 'Training finished. Waiting for all queued all-frame evaluations...'
else
  distance_train_rc=$?
  distance_train_pid=''
  echo 'Training failed; inspect training.log. Saved checkpoints/results are retained.' >&2
  exit "$distance_train_rc"
fi
distance_eval_rc=0
wait "$distance_eval_pid" || distance_eval_rc=$?
distance_eval_pid=''
if ((distance_eval_rc != 0)); then
  echo "Evaluation incomplete; inspect $distance_model/periodic_evaluation/evaluator.log and restart watch." >&2
fi
exit "$distance_eval_rc"
