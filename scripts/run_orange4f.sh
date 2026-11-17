#!/usr/bin/env bash
# Usage: run_orange4f.sh GPU [SOURCE] [VERSION]; all runs share cached selection.
set -euo pipefail
source "${RECONSTRUCTION_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
conda activate 3dgs
cd "$(dirname "${BASH_SOURCE[0]}")/.."
orange_gpu="${1:?Usage: run_orange4f.sh GPU [SOURCE] [VERSION]}"
case "$orange_gpu" in 0|1|2|3) ;; *) echo 'GPU must be 0, 1, 2 or 3' >&2; exit 2;; esac
orange_source="${2:-/workspace/colmap_scenes/orange4f}"
orange_version="${3:-v1}"
[[ "$orange_version" != *[!a-zA-Z0-9_-]* ]] || { echo 'Invalid version' >&2; exit 2; }
orange_source="$(realpath -e "$orange_source")"
orange_output_root="${ORANGE4F_OUTPUT_ROOT:-/data/3dgs_outputs/orange4f/3dgs_finetuning}"
orange_root="$(realpath -m "$orange_output_root/stride3_noaa_$orange_version")"
echo "Dataset: $orange_source"
echo "Outputs: $orange_root"
mkdir -p "$(dirname "$orange_root")"
# Multiple terminal commands can safely prepare the shared scene concurrently.
exec 9>"$orange_root.lock"
flock 9
orange_extra=()
if [[ -n "${ORANGE4F_POINT_CLOUD:-}" ]]; then orange_extra+=(--point-cloud "$ORANGE4F_POINT_CLOUD"); fi
python -u scripts/prepare_orange4f.py --source "$orange_source" --output "$orange_root" --version "$orange_version" "${orange_extra[@]}"
flock -u 9
orange_names=(3dgs 3dgs_fixed_lidar gsprior gsprior_fixed_lidar)
orange_model="$orange_root/${orange_names[$orange_gpu]}"
if [[ "$orange_gpu" -lt 2 ]]; then
  exec bash scripts/run_prepared_reconstruction.sh "$orange_model"
fi
if [[ -n "$(nvidia-smi -i "$orange_gpu" --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "GPU $orange_gpu is occupied; no existing run was stopped." >&2; exit 1
fi
exec 8>"$HOME/.gsprior_orange4f_setup.lock"
flock 8
CUDA_VISIBLE_DEVICES="$orange_gpu" bash scripts/setup_gsprior.sh
flock -u 8
gsprior_repo="${GSPRIOR_REPO:-$HOME/.cache/3dgs/GSPrior-dcb7c89-orange4f-v1}"
gsprior_python="${GSPRIOR_ENV:-$HOME/.venvs/gsprior-orange4f}/bin/python"
orange_eval_pid='';orange_train_pid=''
cleanup() {
  if [[ -n "$orange_train_pid" ]]; then kill "$orange_train_pid" 2>/dev/null || true; wait "$orange_train_pid" 2>/dev/null || true; fi
  if [[ -n "$orange_eval_pid" ]]; then kill "$orange_eval_pid" 2>/dev/null || true; wait "$orange_eval_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$gsprior_python" -u scripts/gsprior_entry.py watch --repo "$gsprior_repo" --model "$orange_model" --parent-pid "$$" \
  >"$orange_model/evaluator.log" 2>&1 &
orange_eval_pid=$!
"$gsprior_python" -u scripts/gsprior_entry.py train --repo "$gsprior_repo" --model "$orange_model" \
  > >(tee "$orange_model/training.log") 2>&1 &
orange_train_pid=$!
wait "$orange_train_pid"; orange_train_pid=''
wait "$orange_eval_pid"; orange_eval_pid=''
