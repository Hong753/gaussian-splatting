#!/usr/bin/env bash
# Build a full-scene LiDAR surface, prove CUDA ray clipping, inspect one view.
set -euo pipefail
source "${DA3_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ray_gpu="${1:-0}"
ray_model="${2:-/data/3dgs_outputs/orange_4F/1_full/3dgs_stride3_aa_lidar_occlusion}"
ray_image="${3:-06663.jpg}"
if (( $# > 3 )); then
    echo 'Usage: bash scripts/inspect_ray_surface.sh GPU MODEL [IMAGE]' >&2
    exit 2
fi
conda activate 3dgs
ray_config=$(python - "$ray_model" "$ray_image" <<'PY'
from pathlib import Path
import sys
from scripts.view_comparisons import model_info
model=Path(sys.argv[1]).resolve()
spec=model_info(model)
if spec['type']!='3dgs':raise ValueError('Choose a native 3DGS checkpoint')
print(spec['cfg']['source_path'])
print(spec['cfg'].get('resolution',-1))
print(model.parent/'lidar_scene_surface_v1')
PY
)
mapfile -t ray_settings <<< "$ray_config"
ray_output="${RAY_SURFACE_OUTPUT:-${ray_settings[2]}}"
# This extension uses existing GLM headers from the pinned native submodule.
if [[ "${RAY_SKIP_INSTALL:-0}" != 1 ]]; then
    git submodule update --init --recursive submodules/diff-gaussian-rasterization
    python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization-occlusion
    python -m pip install -r requirements-scene-occlusion.txt
    python -m pip install --no-build-isolation -r requirements-occlusion.txt
fi
python scripts/smoke_ray_occlusion.py --gpu "$ray_gpu" --output "$ray_output/cuda_smoke.json"
python scripts/smoke_occlusion.py --lidar-mesh --backend ray --gpu "$ray_gpu"
ray_supplement=()
if [[ -n "${RAY_DA3_SUPPLEMENT_SCENE:-}" ]]; then
    ray_supplement=(--supplement-scene "$RAY_DA3_SUPPLEMENT_SCENE")
fi
python -u scripts/prepare_lidar_scene.py --source "${ray_settings[0]}" \
    --keyframes "$ray_model/keyframes.json" --point-cloud "$ray_model/input.ply" \
    --output "$ray_output" --resolution "${ray_settings[1]}" --relative-margin 0 \
    "${ray_supplement[@]}"
ray_view=(--image "$ray_image")
if [[ "${RAY_PRECOMPUTE_ALL:-0}" == 1 ]]; then
    ray_view=()
fi
python -u scripts/prepare_scene_occlusion.py views --scene "$ray_output/scene.json" \
    --point-cloud "$ray_model/input.ply" --resolution "${ray_settings[1]}" \
    --gpu "$ray_gpu" --device cuda --visibility-backend cuda --split all "${ray_view[@]}"
exec python -u scripts/inspect_ray_occlusion.py --scene "$ray_output/scene.json" \
    --point-cloud "$ray_model/input.ply" --model "$ray_model" --image "$ray_image" \
    --surface-band .03 --margin .05 --gpu "$ray_gpu" --host 0.0.0.0 \
    --port "${RAY_INSPECTOR_PORT:-8085}" --check-gradients
