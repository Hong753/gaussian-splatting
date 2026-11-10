#!/usr/bin/env bash
# Native 3DGS + AA + visible-surface tangent densification, with fail-fast stages.
set -euo pipefail
source "${DA3_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
conda activate 3dgs
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tangent_gpu="${1:-2}"
if (($# > 0)); then shift; fi
tangent_source_kind="${1:-lidar}"
if (($# > 0)); then shift; fi
case "$tangent_source_kind" in
  lidar|da3) ;;
  *) echo 'Usage: bash scripts/run_view_tangent.sh [GPU=2] [lidar|da3] [workflow options]' >&2; exit 2 ;;
esac
tangent_root="${DA3_OUTPUT_ROOT:-/data/3dgs_outputs/orange_4F/1_full}"
tangent_source="${DA3_SCENE_ROOT:-/data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full}"
tangent_base="${DA3_BASELINE:-}"
if [[ -z "$tangent_base" ]]; then
  tangent_base="$(python -c 'from scripts.render import resolve_models; import sys; print(resolve_models(None, sys.argv[1], ["sharp_keyframes_stride3_aa"])[0])' "$tangent_root")"
fi
tangent_model="${VIEW_TANGENT_MODEL_DIR:-$tangent_root/3dgs_stride3_aa_${tangent_source_kind}_occlusion_viewtangent_warm7k_lr50}"
tangent_command=(scripts/run_occlusion.py)
if [[ "$tangent_source_kind" == da3 ]]; then
  tangent_command=(scripts/run_da3_occlusion.py --depth-dir "${DA3_DEPTH_DIR:-$tangent_root/da3_lidar_depth_stride3}")
fi
python -u "${tangent_command[@]}" --source "$tangent_source" --model "$tangent_model" \
  --baseline "$tangent_base" --keyframes "$tangent_base/keyframes.json" --gpu "$tangent_gpu" \
  --position-warmup-steps 7000 --position-ramp-steps 2000 --position-lr-scale .02 \
  --view-tangent "$@"
