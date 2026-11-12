#!/usr/bin/env bash
# GPU 0: 15x original local caps, 3x targets, anchored growth, poses and culling.
set -euo pipefail
if (($#>1)); then echo 'Usage: bash scripts/run_pose_lidar_adaptive.sh [GPU]' >&2;exit 2;fi
export LOCAL_SCALE_TUNE_ACTIVATION=1
export LOCAL_SCALE_TUNE_CEILING="${LOCAL_SCALE_TUNE_CEILING:-15}"
export LOCAL_SCALE_TUNE_TARGET="${LOCAL_SCALE_TUNE_TARGET:-3}"
adaptive_root="${LOCAL_SCALE_TUNE_OUTPUT_ROOT:-/data/3dgs_outputs/orange_4F/1_full}"
adaptive_suffix=''
if [[ "${LOCAL_SCALE_TUNE_DEPTH_DECAY:-0}" == 1 ]]; then
  adaptive_suffix="_depth_${DEPTH_UPDATE_MODE:-relative}_ref${DEPTH_UPDATE_REFERENCE:-2}_q${DEPTH_UPDATE_NEAR_QUANTILE:-0.05}_half${DEPTH_UPDATE_HALF_LIFE:-2}_floor${DEPTH_UPDATE_MIN_WEIGHT:-0.05}"
fi
export LOCAL_SCALE_TUNE_MODEL="${LOCAL_SCALE_TUNE_MODEL:-$adaptive_root/3dgs_stride3_aa_lidar_offsets_cap${LOCAL_SCALE_TUNE_CEILING}_target${LOCAL_SCALE_TUNE_TARGET}_poses_culling_adaptive${adaptive_suffix}_${LOCAL_SCALE_TUNE_ITERATIONS:-30000}_v1}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_pose_lidar_culling.sh" "${1:-0}"
