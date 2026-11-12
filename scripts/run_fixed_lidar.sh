#!/usr/bin/env bash
# Native fixed-center training and a separate queued evaluator, on the same GPU.
set -euo pipefail
if (($# > 1)); then
  echo 'Usage: bash scripts/run_fixed_lidar.sh [GPU]; configure paths/settings with FIXED_LIDAR_* variables.' >&2
  exit 2
fi
source "${FIXED_LIDAR_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
conda activate 3dgs
cd "$(dirname "${BASH_SOURCE[0]}")/.."
lidar_gpu="${1:-0}"
lidar_root="${FIXED_LIDAR_OUTPUT_ROOT:-/data/3dgs_outputs/orange_4F/1_full}"
lidar_source="${FIXED_LIDAR_SOURCE:-/data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full}"
lidar_baseline="${FIXED_LIDAR_BASELINE:-}"
if [[ -z "$lidar_baseline" ]]; then
  lidar_baseline="$(python -c 'from scripts.render import resolve_models; import sys; print(resolve_models(None, sys.argv[1], ["sharp_keyframes_stride3_aa"])[0])' "$lidar_root")"
fi
lidar_model="${FIXED_LIDAR_MODEL:-$lidar_root/3dgs_stride3_aa_fixed_lidar_scale2_noprune_100k_v1}"
python -u scripts/run_fixed_lidar.py prepare \
  --source "$lidar_source" --model "$lidar_model" --gpu "$lidar_gpu" \
  --baseline "$lidar_baseline" --keyframes "$lidar_baseline/keyframes.json" \
  --point-cloud "${FIXED_LIDAR_POINTS:-$lidar_baseline/input.ply}" \
  --max-scale-ratio "${FIXED_LIDAR_MAX_SCALE_RATIO:-2.0}" \
  --max-scale-world "${FIXED_LIDAR_MAX_SCALE_WORLD:-0.0}" \
  --scale-control "${FIXED_LIDAR_SCALE_CONTROL:-clamp}" \
  --scale-regularization-weight "${FIXED_LIDAR_SCALE_REGULARIZATION_WEIGHT:-0.01}" \
  --iterations "${FIXED_LIDAR_ITERATIONS:-100000}" \
  --checkpoint-every "${FIXED_LIDAR_CHECKPOINT_EVERY:-10000}" \
  --expected-frames "${FIXED_LIDAR_EXPECTED_FRAMES:-5009}" \
  --report-max-views "${FIXED_LIDAR_REPORT_MAX_VIEWS:-64}" \
  --eval-memory-fraction "${FIXED_LIDAR_EVAL_MEMORY_FRACTION:-0.20}"
lidar_eval_pid=''
lidar_train_pid=''
lidar_cleanup() {
  if [[ -n "$lidar_train_pid" ]]; then
    kill "$lidar_train_pid" 2>/dev/null || true
    wait "$lidar_train_pid" 2>/dev/null || true
  fi
  if [[ -n "$lidar_eval_pid" ]]; then
    kill "$lidar_eval_pid" 2>/dev/null || true
    wait "$lidar_eval_pid" 2>/dev/null || true
  fi
}
trap lidar_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python -u scripts/run_fixed_lidar.py watch --model "$lidar_model" --parent-pid "$$" \
  >"$lidar_model/periodic_evaluation/evaluator.log" 2>&1 &
lidar_eval_pid=$!
echo "Evaluator PID: $lidar_eval_pid; GPU: $lidar_gpu; log: $lidar_model/periodic_evaluation/evaluator.log"
python -u scripts/run_fixed_lidar.py train --model "$lidar_model" \
  > >(tee "$lidar_model/training.log") 2>&1 &
lidar_train_pid=$!
if wait "$lidar_train_pid"; then
  lidar_train_pid=''
  echo 'Training finished. Waiting for all queued all-frame evaluations...'
else
  lidar_train_rc=$?
  lidar_train_pid=''
  echo 'Training failed; inspect training.log. Published checkpoints/results are retained.' >&2
  exit "$lidar_train_rc"
fi
lidar_eval_rc=0
wait "$lidar_eval_pid" || lidar_eval_rc=$?
lidar_eval_pid=''
if ((lidar_eval_rc != 0)); then
  echo "Evaluation incomplete; inspect $lidar_model/periodic_evaluation/evaluator.log and restart watch." >&2
fi
exit "$lidar_eval_rc"
