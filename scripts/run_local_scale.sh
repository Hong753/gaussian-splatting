#!/usr/bin/env bash
# GPU 3: fixed full LiDAR, local size control, independent all-frame evaluator.
set -euo pipefail
if (($# > 1)); then
  echo 'Usage: bash scripts/run_local_scale.sh [GPU]; configure with LOCAL_SCALE_* variables.' >&2
  exit 2
fi
source "${LOCAL_SCALE_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
conda activate 3dgs
cd "$(dirname "${BASH_SOURCE[0]}")/.."
local_gpu="${1:-3}"
local_root="${LOCAL_SCALE_OUTPUT_ROOT:-/data/3dgs_outputs/orange_4F/1_full}"
local_source="${LOCAL_SCALE_SOURCE:-/data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full}"
local_baseline="${LOCAL_SCALE_BASELINE:-}"
if [[ -z "$local_baseline" ]]; then
  local_baseline="$(python -c 'from scripts.render import resolve_models; import sys; print(resolve_models(None, sys.argv[1], ["sharp_keyframes_stride3_aa"])[0])' "$local_root")"
fi
local_model="${LOCAL_SCALE_MODEL:-$local_root/3dgs_stride3_aa_fixed_lidar_local_knn10_100k_v1}"
python -u scripts/run_local_scale.py prepare \
  --source "$local_source" --model "$local_model" --gpu "$local_gpu" \
  --baseline "$local_baseline" --keyframes "$local_baseline/keyframes.json" \
  --point-cloud "${LOCAL_SCALE_POINTS:-$local_baseline/input.ply}" \
  --local-scale-cache "${LOCAL_SCALE_CACHE:-$local_root/lidar_local_scale_knn10_v1}" \
  --neighbors "${LOCAL_SCALE_NEIGHBORS:-10}" \
  --support-sigma "${LOCAL_SCALE_SUPPORT_SIGMA:-3}" \
  --overlap-factor "${LOCAL_SCALE_OVERLAP_FACTOR:-2}" \
  --max-scale-ratio "${LOCAL_SCALE_MAX_SCALE_RATIO:-1.0}" \
  --max-scale-world "${LOCAL_SCALE_MAX_SCALE_WORLD:-0.015}" \
  --scale-regularization-weight "${LOCAL_SCALE_WEIGHT:-0.01}" \
  --audit-samples "${LOCAL_SCALE_AUDIT_SAMPLES:-2048}" \
  --audit-pair-budget "${LOCAL_SCALE_AUDIT_PAIR_BUDGET:-2000000}" \
  --cache-workers "${LOCAL_SCALE_CACHE_WORKERS:-8}" \
  --resolution "${LOCAL_SCALE_RESOLUTION:--1}" \
  --iterations "${LOCAL_SCALE_ITERATIONS:-100000}" \
  --checkpoint-every "${LOCAL_SCALE_CHECKPOINT_EVERY:-10000}" \
  --expected-frames "${LOCAL_SCALE_EXPECTED_FRAMES:-5009}" \
  --report-max-views "${LOCAL_SCALE_REPORT_MAX_VIEWS:-64}" \
  --eval-memory-fraction "${LOCAL_SCALE_EVAL_MEMORY_FRACTION:-0.20}"
local_eval_pid=''
local_train_pid=''
local_cleanup() {
  if [[ -n "$local_train_pid" ]]; then
    kill "$local_train_pid" 2>/dev/null || true
    wait "$local_train_pid" 2>/dev/null || true
  fi
  if [[ -n "$local_eval_pid" ]]; then
    kill "$local_eval_pid" 2>/dev/null || true
    wait "$local_eval_pid" 2>/dev/null || true
  fi
}
trap local_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python -u scripts/run_local_scale.py watch --model "$local_model" --parent-pid "$$" \
  >"$local_model/periodic_evaluation/evaluator.log" 2>&1 &
local_eval_pid=$!
echo "Evaluator PID: $local_eval_pid; GPU: $local_gpu; log: $local_model/periodic_evaluation/evaluator.log"
python -u scripts/run_local_scale.py train --model "$local_model" \
  > >(tee "$local_model/training.log") 2>&1 &
local_train_pid=$!
if wait "$local_train_pid"; then
  local_train_pid=''
  echo 'Training finished. Waiting for all queued all-frame evaluations...'
else
  local_train_rc=$?
  local_train_pid=''
  echo 'Training failed; inspect training.log. Published checkpoints/results are retained.' >&2
  exit "$local_train_rc"
fi
local_eval_rc=0
wait "$local_eval_pid" || local_eval_rc=$?
local_eval_pid=''
if ((local_eval_rc != 0)); then
  echo "Evaluation incomplete; inspect $local_model/periodic_evaluation/evaluator.log and restart watch." >&2
fi
exit "$local_eval_rc"
