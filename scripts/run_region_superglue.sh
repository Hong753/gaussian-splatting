#!/usr/bin/env bash
# The learned frontend is CUDA-only; COLMAP verification/mapping are headless.
set -euo pipefail
export REGION_COLMAP_FRONTEND=superglue
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_region_colmap.sh" "$@"
