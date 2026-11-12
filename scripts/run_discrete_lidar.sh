#!/usr/bin/env bash
# Third ablation: measured candidate pool + adaptive activation + scale pruning.
set -euo pipefail
if (($# > 1)); then
  echo 'Usage: bash scripts/run_discrete_lidar.sh [GPU=2]' >&2
  exit 2
fi
source "${DISCRETE_LIDAR_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
conda activate 3dgs
cd "$(dirname "${BASH_SOURCE[0]}")/.."
discrete_gpu="${1:-2}"
discrete_root="${DISCRETE_LIDAR_OUTPUT_ROOT:-/data/3dgs_outputs/orange_4F/1_full}"
discrete_source="${DISCRETE_LIDAR_SOURCE:-/data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full}"
discrete_baseline="${DISCRETE_LIDAR_BASELINE:-}"
if [[ -z "$discrete_baseline" ]]; then
  discrete_baseline="$(python -c 'from scripts.render import resolve_models; import sys; print(resolve_models(None, sys.argv[1], ["sharp_keyframes_stride3_aa"])[0])' "$discrete_root")"
fi
discrete_model="${DISCRETE_LIDAR_MODEL:-$discrete_root/3dgs_stride3_aa_discrete_lidar_prune5cm_100k_v1}"
python -u scripts/run_discrete_lidar.py prepare \
  --source "$discrete_source" --model "$discrete_model" --gpu "$discrete_gpu" \
  --baseline "$discrete_baseline" --keyframes "$discrete_baseline/keyframes.json" \
  --point-cloud "${DISCRETE_LIDAR_POINTS:-$discrete_baseline/input.ply}" \
  --cache "${DISCRETE_LIDAR_CACHE:-$discrete_root/lidar_discrete_pool_v1}" \
  --prune-sigma "${DISCRETE_LIDAR_PRUNE_SIGMA:-0.05}" \
  --initial-fraction "${DISCRETE_LIDAR_INITIAL_FRACTION:-0.5}" \
  --init-voxel "${DISCRETE_LIDAR_INIT_VOXEL:-0.05}" \
  --growth-radius "${DISCRETE_LIDAR_GROWTH_RADIUS:-0.10}" \
  --max-new "${DISCRETE_LIDAR_MAX_NEW:-50000}" \
  --iterations "${DISCRETE_LIDAR_ITERATIONS:-100000}" \
  --checkpoint-every "${DISCRETE_LIDAR_CHECKPOINT_EVERY:-10000}" \
  --expected-frames "${DISCRETE_LIDAR_EXPECTED_FRAMES:-5009}" \
  --report-max-views "${DISCRETE_LIDAR_REPORT_MAX_VIEWS:-64}" \
  --eval-memory-fraction "${DISCRETE_LIDAR_EVAL_MEMORY_FRACTION:-0.20}"
discrete_train_pid=''
discrete_eval_pid=''
discrete_cleanup() {
  if [[ -n "$discrete_train_pid" ]]; then
    kill "$discrete_train_pid" 2>/dev/null || true
    wait "$discrete_train_pid" 2>/dev/null || true
  fi
  if [[ -n "$discrete_eval_pid" ]]; then
    kill "$discrete_eval_pid" 2>/dev/null || true
    wait "$discrete_eval_pid" 2>/dev/null || true
  fi
}
trap discrete_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python -u scripts/run_discrete_lidar.py watch --model "$discrete_model" --parent-pid "$$" \
  >"$discrete_model/periodic_evaluation/evaluator.log" 2>&1 &
discrete_eval_pid=$!
echo "Evaluator PID: $discrete_eval_pid; GPU: $discrete_gpu; log: $discrete_model/periodic_evaluation/evaluator.log"
python -u scripts/run_discrete_lidar.py train --model "$discrete_model" \
  > >(tee "$discrete_model/training.log") 2>&1 &
discrete_train_pid=$!
if wait "$discrete_train_pid"; then
  discrete_train_pid=''
  echo 'Training finished. Waiting for all queued all-frame evaluations...'
else
  discrete_train_rc=$?
  discrete_train_pid=''
  echo 'Training failed; inspect training.log. Cache and published checkpoints are retained.' >&2
  exit "$discrete_train_rc"
fi
discrete_eval_rc=0
wait "$discrete_eval_pid" || discrete_eval_rc=$?
discrete_eval_pid=''
exit "$discrete_eval_rc"
