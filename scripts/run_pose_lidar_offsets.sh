#!/usr/bin/env bash
# Larger local scales and bounded LiDAR offsets, alternating with camera poses.
set -euo pipefail
if (($#>1)); then echo 'Usage: bash scripts/run_pose_lidar_offsets.sh [GPU]' >&2;exit 2;fi
export LOCAL_SCALE_TUNE_POSES=1
export LOCAL_SCALE_TUNE_OFFSETS=1
export LOCAL_SCALE_TUNE_CEILING="${LOCAL_SCALE_TUNE_CEILING:-5}"
export LOCAL_SCALE_TUNE_TARGET="${LOCAL_SCALE_TUNE_TARGET:-1}"
export LOCAL_SCALE_TUNE_SCHEDULE="${LOCAL_SCALE_TUNE_SCHEDULE:-joint}"
offset_root="${LOCAL_SCALE_TUNE_OUTPUT_ROOT:-/data/3dgs_outputs/orange_4F/1_full}"
export LOCAL_SCALE_TUNE_MODEL="${LOCAL_SCALE_TUNE_MODEL:-$offset_root/3dgs_stride3_aa_lidar_offsets_cap${LOCAL_SCALE_TUNE_CEILING}_poses_${LOCAL_SCALE_TUNE_ITERATIONS:-30000}_v1}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_local_scale_tune.sh" "${1:-2}"
