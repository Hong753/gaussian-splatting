#!/usr/bin/env bash
# Additional cap15 experiment: view-relative depth or normalized-rank decay.
set -euo pipefail
if (($#>1)); then echo 'Usage: bash scripts/run_pose_lidar_depth_decay.sh [GPU]' >&2;exit 2;fi
export LOCAL_SCALE_TUNE_DEPTH_DECAY=1
export DEPTH_UPDATE_MODE="${DEPTH_UPDATE_MODE:-relative}"
export DEPTH_UPDATE_REFERENCE="${DEPTH_UPDATE_REFERENCE:-2}"
export DEPTH_UPDATE_HALF_LIFE="${DEPTH_UPDATE_HALF_LIFE:-2}"
export DEPTH_UPDATE_MIN_WEIGHT="${DEPTH_UPDATE_MIN_WEIGHT:-0.05}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_pose_lidar_adaptive.sh" "${1:-0}"
