#!/usr/bin/env bash
# Usage: run_orange4f_scaffold.sh GPU learnable|fixed|fixed-static|indexed|sensed|mesh-visible [SOURCE] [SUITE_VERSION]
set -euo pipefail
source "${RECONSTRUCTION_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
conda activate 3dgs
cd "$(dirname "${BASH_SOURCE[0]}")/.."
scaffold_gpu="${1:?Supply GPU and a Scaffold variant}"
scaffold_variant="${2:?Supply a Scaffold variant}"
case "$scaffold_variant" in learnable|fixed|fixed-static|indexed|sensed|mesh-visible) ;; *) echo 'Expected learnable, fixed, fixed-static, indexed, sensed or mesh-visible' >&2; exit 2;; esac
scaffold_suite_version="${4:-v1}"
scaffold_version="${SCAFFOLD_RUN_VERSION:-v1}"
[[ "$scaffold_suite_version" != *[!a-zA-Z0-9_-]* && "$scaffold_version" != *[!a-zA-Z0-9_-]* ]] || exit 2
scaffold_parent="${ORANGE4F_OUTPUT_ROOT:-/data/3dgs_outputs/orange4f/3dgs_finetuning}"
scaffold_root="$(realpath -m "$scaffold_parent/stride3_noaa_$scaffold_suite_version")"
mkdir -p "$(dirname "$scaffold_root")"
exec 9>"$scaffold_root.lock"
flock 9
if [[ -n "${3:-}" ]]; then
  scaffold_source="$(realpath -e "$3")"
elif [[ -f "$scaffold_root/suite.json" ]]; then
  scaffold_source="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$scaffold_root/suite.json")"
else
  scaffold_source=/workspace/colmap_scenes/orange4f
fi
scaffold_extra=()
if [[ -n "${ORANGE4F_POINT_CLOUD:-}" ]]; then scaffold_extra+=(--point-cloud "$ORANGE4F_POINT_CLOUD"); fi
python -u scripts/prepare_orange4f.py --source "$scaffold_source" --output "$scaffold_root" "${scaffold_extra[@]}"
python -u scripts/run_orange4f_scaffold.py --suite "$scaffold_root" --variant "$scaffold_variant" \
  --gpu "$scaffold_gpu" --voxel-size "${SCAFFOLD_VOXEL_SIZE:-0.02}" --run-version "$scaffold_version"
flock -u 9
exec bash scripts/run_prepared_reconstruction.sh "$scaffold_root/scaffold_${scaffold_variant}_anchors_$scaffold_version"
