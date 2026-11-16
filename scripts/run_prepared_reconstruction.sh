#!/usr/bin/env bash
# Run an already validated periodic plan, with its independent all-frame worker.
set -euo pipefail
if (($#!=1)); then echo 'Usage: bash scripts/run_prepared_reconstruction.sh MODEL' >&2; exit 2; fi
source "${RECONSTRUCTION_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
conda activate 3dgs
cd "$(dirname "${BASH_SOURCE[0]}")/.."
prepared_model="$1"
prepared_gpu="$(python -c 'from scripts.run_soft_surface_long import load_plan; import sys; print(load_plan(sys.argv[1])["gpu"])' "$prepared_model")"
if [[ -n "$(nvidia-smi -i "$prepared_gpu" --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "GPU $prepared_gpu is still occupied; prepared run was not started." >&2; exit 1
fi
prepared_train_pid=''
prepared_eval_pid=''
cleanup() {
  if [[ -n "$prepared_train_pid" ]]; then kill "$prepared_train_pid" 2>/dev/null || true; wait "$prepared_train_pid" 2>/dev/null || true; fi
  if [[ -n "$prepared_eval_pid" ]]; then kill "$prepared_eval_pid" 2>/dev/null || true; wait "$prepared_eval_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python -u scripts/run_soft_surface_long.py watch --model "$prepared_model" --parent-pid "$$" \
  >"$prepared_model/periodic_evaluation/evaluator.log" 2>&1 &
prepared_eval_pid=$!
python -u scripts/run_soft_surface_long.py train --model "$prepared_model" > >(tee "$prepared_model/training.log") 2>&1 &
prepared_train_pid=$!
wait "$prepared_train_pid"
prepared_train_pid=''
wait "$prepared_eval_pid"
prepared_eval_pid=''
