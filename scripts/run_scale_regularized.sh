#!/usr/bin/env bash
# Second fixed-LiDAR run: same periodic evaluator, independent model folder/GPU.
set -euo pipefail
if (($# > 1)); then
  echo 'Usage: bash scripts/run_scale_regularized.sh [GPU=1]' >&2
  exit 2
fi
reg_root="${FIXED_LIDAR_OUTPUT_ROOT:-/data/3dgs_outputs/orange_4F/1_full}"
export FIXED_LIDAR_MODEL="${SCALE_REG_MODEL:-$reg_root/3dgs_stride3_aa_fixed_lidar_scalereg_100k_v1}"
export FIXED_LIDAR_SCALE_CONTROL=regularize
export FIXED_LIDAR_MAX_SCALE_RATIO="${SCALE_REG_RATIO:-2.0}"
export FIXED_LIDAR_MAX_SCALE_WORLD="${SCALE_REG_TARGET_WORLD:-0.05}"
export FIXED_LIDAR_SCALE_REGULARIZATION_WEIGHT="${SCALE_REG_WEIGHT:-0.01}"
# The delegated launcher activates conda 3dgs itself, validates fresh output,
# then starts training and one evaluator on the requested GPU.
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_fixed_lidar.sh" "${1:-1}"
