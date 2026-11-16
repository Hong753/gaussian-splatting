#!/usr/bin/env bash
# Match the GPU 1 native/pruning run; only lower the clone/split boundary 100x.
set -euo pipefail
if (($#>1)); then echo 'Usage: bash scripts/run_native_lidar_fine_split.sh [GPU]' >&2; exit 2; fi
export NATIVE_LIDAR_PERCENT_DENSE=0.0001
export NATIVE_LIDAR_MODEL="${NATIVE_LIDAR_MODEL:-${NATIVE_LIDAR_OUTPUT_ROOT:-/data/3dgs_outputs/orange_4F/1_full}/3dgs_stride3_aa_lidar_unconstrained_lr50_split1e-4_noresets_prune_30000_v1}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_native_lidar_densify.sh" "${1:-0}"
