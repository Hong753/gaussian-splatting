#!/usr/bin/env bash
# Native 3DGS + AA; fixed measured XYZ, no population edits or scale constraints.
set -euo pipefail
if (($#>1)); then echo 'Usage: bash scripts/run_fixed_lidar_unconstrained.sh [GPU]' >&2; exit 2; fi
export FIXED_LIDAR_SCALE_CONTROL=off
export FIXED_LIDAR_MAX_SCALE_RATIO=2
export FIXED_LIDAR_MAX_SCALE_WORLD=0
export FIXED_LIDAR_ITERATIONS="${FIXED_LIDAR_ITERATIONS:-30000}"
export DEBUG_PREVIEW_EVERY="${DEBUG_PREVIEW_EVERY:-500}"
export FIXED_LIDAR_MODEL="${FIXED_LIDAR_MODEL:-${FIXED_LIDAR_OUTPUT_ROOT:-/data/3dgs_outputs/orange_4F/1_full}/3dgs_stride3_aa_fixed_lidar_unconstrained_noresets_${FIXED_LIDAR_ITERATIONS}_v1}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_fixed_lidar.sh" "${1:-2}"
