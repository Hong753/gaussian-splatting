#!/usr/bin/env bash
# Native 3DGS + AA + visible-surface tangent densification, with fail-fast stages.
set -euo pipefail
source "${DA3_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
conda activate 3dgs
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tangent_gpu="${1:-1}"
if (($# > 0)); then shift; fi
tangent_source_kind="${1:-mesh}"
if (($# > 0)); then shift; fi
case "$tangent_source_kind" in
  mesh|lidar|da3) ;;
  *) echo 'Usage: bash scripts/run_view_tangent.sh [GPU=1] [mesh|lidar|da3] [workflow options]' >&2; exit 2 ;;
esac
tangent_root="${DA3_OUTPUT_ROOT:-/data/3dgs_outputs/orange_4F/1_full}"
tangent_source="${DA3_SCENE_ROOT:-/data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full}"
tangent_base="${DA3_BASELINE:-}"
if [[ -z "$tangent_base" ]]; then
  tangent_base="$(python -c 'from scripts.render import resolve_models; import sys; print(resolve_models(None, sys.argv[1], ["sharp_keyframes_stride3_aa"])[0])' "$tangent_root")"
fi
tangent_variant="$tangent_source_kind"
if [[ "$tangent_source_kind" == da3 ]]; then tangent_variant=da3_v2; fi
tangent_model="${VIEW_TANGENT_MODEL_DIR:-$tangent_root/3dgs_stride3_aa_${tangent_variant}_occlusion_viewtangent_warm7k_lr50}"
tangent_command=(scripts/run_occlusion.py)
if [[ "$tangent_source_kind" == mesh ]]; then
  tangent_model="${VIEW_TANGENT_MODEL_DIR:-$tangent_root/3dgs_stride3_aa_lidar_mesh_trainonly_viewtangent_warm7k_lr50}"
  tangent_command+=(--scene-geometry "${VIEW_TANGENT_SCENE:-$tangent_root/lidar_scene_surface_v1/scene.json}"
    --prepare-visibility --occlusion-backend ray --inference-occlusion off)
elif [[ "$tangent_source_kind" == da3 ]]; then
  tangent_command=(scripts/run_da3_occlusion.py --depth-dir "${DA3_DEPTH_DIR:-$tangent_root/da3_lidar_depth_stride3_v2}")
fi
python -u "${tangent_command[@]}" --source "$tangent_source" --model "$tangent_model" \
  --baseline "$tangent_base" --keyframes "$tangent_base/keyframes.json" --gpu "$tangent_gpu" \
  --position-warmup-steps 7000 --position-ramp-steps 2000 --position-lr-scale .02 \
  --view-tangent "$@"
