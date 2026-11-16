#!/usr/bin/env bash
# Fresh restart of the exact GPU 0 culling run, retaining all previous outputs.
set -euo pipefail
if (($#>2)); then echo 'Usage: bash scripts/restart_culling_previews.sh [GPU] [OLD_MODEL]' >&2; exit 2; fi
source "${RECONSTRUCTION_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
conda activate 3dgs
cd "$(dirname "${BASH_SOURCE[0]}")/.."
restart_args=(--gpu "${1:-0}" --root "${RECONSTRUCTION_ROOT:-/data/3dgs_outputs/orange_4F/1_full}" --every "${DEBUG_PREVIEW_EVERY:-500}")
if (($#==2)); then restart_args+=(--model "$2"); fi
restart_receipt="$(mktemp)"
trap 'rm -f "$restart_receipt"' EXIT
python -u scripts/restart_culling_previews.py "${restart_args[@]}" --stop --receipt "$restart_receipt"
restart_model="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"])' "$restart_receipt")"
rm -f "$restart_receipt"
trap - EXIT
exec bash scripts/run_prepared_reconstruction.sh "$restart_model"
