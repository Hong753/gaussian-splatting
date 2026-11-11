#!/usr/bin/env bash
# Build a multi-frame surface, prove the CUDA wall/hole behavior, inspect one view.
set -euo pipefail
source "${DA3_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ray_gpu="${1:-5}"
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
from scripts.view_comparisons import model_info, image_id
model=Path(sys.argv[1]).resolve()
spec=model_info(model)
if spec['type']!='3dgs':raise ValueError('Choose a native 3DGS checkpoint')
print(spec['cfg']['source_path'])
print(spec['cfg'].get('resolution',-1))
print(model.parent/'da3_lidar_depth_stride3_trajectory40_v1')
print(model.parent/('ray_surface_'+image_id(sys.argv[2])+'_v1'))
PY
)
mapfile -t ray_settings <<< "$ray_config"
ray_depth="${DA3_TRAJECTORY_DEPTH_DIR:-${ray_settings[2]}}"
ray_output="${RAY_SURFACE_OUTPUT:-${ray_settings[3]}}"
# This extension uses existing GLM headers from the pinned native submodule.
git submodule update --init --recursive submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization-occlusion
python -m pip install -r requirements-scene-occlusion.txt
python -m pip install --no-build-isolation -r requirements-occlusion.txt
python scripts/smoke_ray_occlusion.py --gpu "$ray_gpu" --output "$ray_output/cuda_smoke.json"
python scripts/smoke_occlusion.py --mesh --backend ray --gpu "$ray_gpu"
conda activate da3-depth
python -m pip install -r requirements-da3.txt
python scripts/check_da3_environment.py --gpu "$ray_gpu"
python -u scripts/prepare_da3_occlusion.py --source "${ray_settings[0]}" \
    --keyframes "$ray_model/keyframes.json" --point-cloud "$ray_model/input.ply" \
    --output "$ray_depth" --view-grouping trajectory --group-size 40 \
    --shared-views 8 --local-neighbors 2 --process-res 504 \
    --resolution "${ray_settings[1]}" --gpu "$ray_gpu" --inspect-image "$ray_image"
conda activate 3dgs
python -u scripts/prepare_ray_surface_preview.py --depth-dir "$ray_depth" \
    --image "$ray_image" --output "$ray_output" --point-cloud "$ray_model/input.ply"
python -u scripts/prepare_scene_occlusion.py fuse --depth-dir "$ray_output/depth" \
    --point-cloud "$ray_model/input.ply" --output "$ray_output/scene" --allow-partial
exec python -u scripts/inspect_ray_occlusion.py --scene "$ray_output/scene/scene.json" \
    --point-cloud "$ray_model/input.ply" --model "$ray_model" --image "$ray_image" \
    --gpu "$ray_gpu" --host 0.0.0.0 --port "${RAY_INSPECTOR_PORT:-8085}" --check-gradients
