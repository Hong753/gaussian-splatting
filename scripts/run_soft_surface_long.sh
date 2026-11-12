#!/usr/bin/env bash
# Two Python processes: foreground training and one queued evaluator on the same GPU.
set -euo pipefail
if (($# > 1)); then
  echo 'Usage: bash scripts/run_soft_surface_long.sh [GPU]; use SOFT_SURFACE_* variables for paths/settings.' >&2
  exit 2
fi
source "${SOFT_SURFACE_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
conda activate 3dgs
cd "$(dirname "${BASH_SOURCE[0]}")/.."
soft_gpu="${1:-0}"
soft_root="${SOFT_SURFACE_OUTPUT_ROOT:-/data/3dgs_outputs/orange_4F/1_full}"
soft_source="${SOFT_SURFACE_SOURCE:-/data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full}"
soft_baseline="${SOFT_SURFACE_BASELINE:-}"
if [[ -z "$soft_baseline" ]]; then
  soft_baseline="$(python -c 'from scripts.render import resolve_models; import sys; print(resolve_models(None, sys.argv[1], ["sharp_keyframes_stride3_aa"])[0])' "$soft_root")"
fi
soft_model="${SOFT_SURFACE_MODEL:-$soft_root/3dgs_stride3_aa_soft_surface_warm7k_lr50_100k_v1}"
python -u scripts/run_soft_surface_long.py prepare \
  --source "$soft_source" --model "$soft_model" --gpu "$soft_gpu" \
  --scene-geometry "${SOFT_SURFACE_SCENE:-$soft_root/lidar_scene_surface_v1/scene.json}" \
  --baseline "$soft_baseline" --keyframes "$soft_baseline/keyframes.json" \
  --iterations "${SOFT_SURFACE_ITERATIONS:-100000}" \
  --checkpoint-every "${SOFT_SURFACE_CHECKPOINT_EVERY:-10000}" \
  --expected-frames "${SOFT_SURFACE_EXPECTED_FRAMES:-5009}" \
  --report-max-views "${SOFT_SURFACE_REPORT_MAX_VIEWS:-64}" \
  --eval-memory-fraction "${SOFT_SURFACE_EVAL_MEMORY_FRACTION:-0.20}"
soft_eval_pid=''
soft_train_pid=''
soft_cleanup() {
  if [[ -n "$soft_train_pid" ]]; then
    kill "$soft_train_pid" 2>/dev/null || true
    wait "$soft_train_pid" 2>/dev/null || true
  fi
  if [[ -n "$soft_eval_pid" ]]; then
    kill "$soft_eval_pid" 2>/dev/null || true
    wait "$soft_eval_pid" 2>/dev/null || true
  fi
}
trap soft_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python -u scripts/run_soft_surface_long.py watch --model "$soft_model" --parent-pid "$$" \
  >"$soft_model/periodic_evaluation/evaluator.log" 2>&1 &
soft_eval_pid=$!
echo "Evaluator PID: $soft_eval_pid; GPU: $soft_gpu; log: $soft_model/periodic_evaluation/evaluator.log"
python -u scripts/run_soft_surface_long.py train --model "$soft_model" \
  > >(tee "$soft_model/training.log") 2>&1 &
soft_train_pid=$!
if wait "$soft_train_pid"; then
  soft_train_pid=''
  echo 'Training finished. Waiting for all queued all-frame evaluations...'
else
  soft_train_rc=$?
  soft_train_pid=''
  echo 'Training failed; inspect training.log. Published checkpoints and evaluation results are retained.' >&2
  exit "$soft_train_rc"
fi
soft_eval_rc=0
wait "$soft_eval_pid" || soft_eval_rc=$?
soft_eval_pid=''
if ((soft_eval_rc != 0)); then
  echo "Evaluation incomplete; inspect $soft_model/periodic_evaluation/evaluator.log and restart watch." >&2
fi
exit "$soft_eval_rc"
