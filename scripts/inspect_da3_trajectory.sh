#!/usr/bin/env bash
# Preview one trajectory group before committing to full scene preprocessing.
set -euo pipefail
source "${DA3_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
trajectory_gpu="${1:-5}"
trajectory_model="${2:-/data/3dgs_outputs/orange_4F/1_full/3dgs_stride3_aa_lidar_occlusion}"
trajectory_image="${3:-06663.jpg}"
if (( $# > 3 )); then
    echo 'Usage: bash scripts/inspect_da3_trajectory.sh GPU MODEL [EXACT_IMAGE_NAME]' >&2
    exit 2
fi
conda activate 3dgs
trajectory_config=$(python - "$trajectory_model" <<'PY'
from pathlib import Path
import sys
from scripts.view_comparisons import model_info
model = Path(sys.argv[1]).expanduser().resolve()
spec = model_info(model)
if spec['type'] != '3dgs':
    raise ValueError('Choose a native checkpoint')
print(spec['cfg']['source_path'])
print(spec['cfg'].get('resolution', -1))
print(model.parent/'da3_lidar_depth_stride3_trajectory40_v1')
PY
)
mapfile -t trajectory_settings <<< "$trajectory_config"
trajectory_depth="${DA3_TRAJECTORY_DEPTH_DIR:-${trajectory_settings[2]}}"
conda activate da3-depth
python -m pip install -r requirements-da3.txt
python scripts/check_da3_environment.py --gpu "$trajectory_gpu"
python -u scripts/prepare_da3_occlusion.py --source "${trajectory_settings[0]}" \
    --keyframes "$trajectory_model/keyframes.json" --point-cloud "$trajectory_model/input.ply" \
    --output "$trajectory_depth" --view-grouping trajectory --group-size 40 \
    --shared-views 8 --local-neighbors 2 --process-res 504 \
    --resolution "${trajectory_settings[1]}" --gpu "$trajectory_gpu" --inspect-image "$trajectory_image"
conda activate 3dgs
python -u scripts/audit_da3_occlusion.py --depth-dir "$trajectory_depth" --images "$trajectory_image" --collect
python -m pip install 'viser==1.0.30'
python scripts/smoke_da3_occlusion.py --gpu "$trajectory_gpu"
exec python -u scripts/inspect_occlusion.py --model "$trajectory_model" --depth-dir "$trajectory_depth" \
    --image "$trajectory_image" --gpu "$trajectory_gpu" --host 0.0.0.0 \
    --port "${OCCLUSION_INSPECTOR_PORT:-8082}" --check-gradients
