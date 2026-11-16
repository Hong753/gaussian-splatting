#!/usr/bin/env bash
# Matched counterpart to the latest depth-decay run: strictly fixed LiDAR XYZ.
set -euo pipefail
if (($#>1)); then echo 'Usage: bash scripts/run_pose_lidar_fixed_positions.sh [GPU]' >&2;exit 2;fi
export LOCAL_SCALE_TUNE_FIXED_POSITIONS=1
export LOCAL_SCALE_TUNE_CEILING="${LOCAL_SCALE_TUNE_CEILING:-15}"
export LOCAL_SCALE_TUNE_TARGET="${LOCAL_SCALE_TUNE_TARGET:-3}"
export DEPTH_UPDATE_MODE="${DEPTH_UPDATE_MODE:-relative}"
fixed_root="${LOCAL_SCALE_TUNE_OUTPUT_ROOT:-/data/3dgs_outputs/orange_4F/1_full}"
fixed_depth="depth_${DEPTH_UPDATE_MODE}_ref${DEPTH_UPDATE_REFERENCE:-2}_q${DEPTH_UPDATE_NEAR_QUANTILE:-0.05}_half${DEPTH_UPDATE_HALF_LIFE:-2}_floor${DEPTH_UPDATE_MIN_WEIGHT:-0.05}"
export LOCAL_SCALE_TUNE_MODEL="${LOCAL_SCALE_TUNE_MODEL:-$fixed_root/3dgs_stride3_aa_fixed_lidar_cap${LOCAL_SCALE_TUNE_CEILING}_target${LOCAL_SCALE_TUNE_TARGET}_poses_culling_adaptive_${fixed_depth}_noresets_${LOCAL_SCALE_TUNE_ITERATIONS:-30000}_v1}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_pose_lidar_depth_decay.sh" "${1:-0}"
