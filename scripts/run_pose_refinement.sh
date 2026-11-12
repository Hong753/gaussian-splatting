#!/usr/bin/env bash
# Additional bounded-pose trial; existing experiments are not stopped.
set -euo pipefail
if (($#>1)); then echo 'Usage: bash scripts/run_pose_refinement.sh [GPU]' >&2;exit 2;fi
export LOCAL_SCALE_TUNE_POSES=1
export LOCAL_SCALE_TUNE_SCHEDULE="${LOCAL_SCALE_TUNE_SCHEDULE:-joint}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_local_scale_tune.sh" "${1:-0}"
