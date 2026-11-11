#!/usr/bin/env bash
# GPU, native model providing LiDAR/keyframes, optional first inspected image.
set -euo pipefail
source "${DA3_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
scene_gpu="${1:-5}"
scene_model="${2:-/data/3dgs_outputs/orange_4F/1_full/3dgs_stride3_aa_lidar_occlusion}"
scene_image="${3:-07519.jpg}"
if (( $# > 3 )); then
    echo 'Usage: bash scripts/prepare_scene_occlusion.sh GPU MODEL [EXACT_IMAGE_NAME]' >&2
    exit 2
fi
conda activate 3dgs
scene_config=$(python - "$scene_model" <<'PY'
from pathlib import Path
import sys
from scripts.view_comparisons import model_info
model = Path(sys.argv[1]).expanduser().resolve()
spec = model_info(model)
if spec['type'] != '3dgs':
    raise ValueError('Choose a native 3DGS model with input.ply and keyframes.json')
for name in ('input.ply', 'keyframes.json'):
    if not (model/name).is_file():
        raise FileNotFoundError(model/name)
print(spec['cfg']['source_path'])
print(spec['cfg'].get('resolution', -1))
print(model.parent/'da3_lidar_depth_stride3_v2')
print(model.parent/'scene_occlusion_stride3_v1')
PY
)
mapfile -t scene_settings <<< "$scene_config"
scene_depth="${DA3_DEPTH_DIR:-${scene_settings[2]}}"
scene_output="${SCENE_OCCLUSION_DIR:-${scene_settings[3]}}"
conda activate da3-depth
python -m pip install -r requirements-da3.txt
python scripts/check_da3_environment.py --gpu "$scene_gpu"
python -u scripts/prepare_da3_occlusion.py \
    --source "${scene_settings[0]}" --keyframes "$scene_model/keyframes.json" \
    --point-cloud "$scene_model/input.ply" --output "$scene_depth" \
    --resolution "${scene_settings[1]}" --process-res 756 --gpu "$scene_gpu"
conda activate 3dgs
python -m pip install -r requirements-scene-occlusion.txt
python -m pip install --no-build-isolation -r requirements-occlusion.txt
python scripts/smoke_occlusion.py --mesh --gpu "$scene_gpu"
python -u scripts/prepare_scene_occlusion.py fuse --depth-dir "$scene_depth" \
    --point-cloud "$scene_model/input.ply" --output "$scene_output"
python -u scripts/prepare_scene_occlusion.py views --scene "$scene_output/scene.json" \
    --point-cloud "$scene_model/input.ply" --resolution "${scene_settings[1]}" --gpu "$scene_gpu"
exec python -u scripts/inspect_scene_visibility.py --scene "$scene_output/scene.json" \
    --point-cloud "$scene_model/input.ply" --image "$scene_image" --host 0.0.0.0 \
    --port "${SCENE_VISIBILITY_PORT:-8083}"
