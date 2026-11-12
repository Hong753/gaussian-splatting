#!/usr/bin/env bash
# GPU 3: the cap5/offset/pose experiment plus training-only CUDA mesh culling.
set -euo pipefail
if (($#>1)); then echo 'Usage: bash scripts/run_pose_lidar_culling.sh [GPU]' >&2;exit 2;fi
export LOCAL_SCALE_TUNE_POSES=1
export LOCAL_SCALE_TUNE_OFFSETS=1
export LOCAL_SCALE_TUNE_CULLING=1
export LOCAL_SCALE_TUNE_CEILING="${LOCAL_SCALE_TUNE_CEILING:-5}"
export LOCAL_SCALE_TUNE_SCHEDULE="${LOCAL_SCALE_TUNE_SCHEDULE:-joint}"
culling_root="${LOCAL_SCALE_TUNE_OUTPUT_ROOT:-/data/3dgs_outputs/orange_4F/1_full}"
export LOCAL_SCALE_TUNE_MODEL="${LOCAL_SCALE_TUNE_MODEL:-$culling_root/3dgs_stride3_aa_lidar_offsets_cap${LOCAL_SCALE_TUNE_CEILING}_poses_culling_${LOCAL_SCALE_TUNE_ITERATIONS:-30000}_v1}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_pose_lidar_offsets.sh" "${1:-3}"
