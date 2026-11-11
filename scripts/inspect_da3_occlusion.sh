#!/usr/bin/env bash
# GPU, saved native model, optional exact training image. No training is started.
set -euo pipefail
source "${DA3_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
inspect_gpu="${1:-5}"
inspect_model="${2:-/data/3dgs_outputs/orange_4F/1_full/3dgs_stride3_aa_lidar_occlusion}"
inspect_image="${3:-}"
if (( $# > 3 )); then
    echo 'Usage: bash scripts/inspect_da3_occlusion.sh GPU MODEL [EXACT_IMAGE_NAME]' >&2
    exit 2
fi

conda activate 3dgs
inspect_config=$(python - "$inspect_model" <<'PY'
from pathlib import Path
import json
import sys
from scripts.view_comparisons import model_info
from utils.da3_occlusion import VERSION
model = Path(sys.argv[1]).expanduser().resolve()
spec = model_info(model)
if spec['type'] != '3dgs':
    raise ValueError('Choose a native 3DGS checkpoint')
for filename in ('keyframes.json', 'input.ply'):
    if not (model/filename).is_file():
        raise FileNotFoundError(model/filename)
print(spec['cfg']['source_path'])
print(spec['cfg'].get('resolution', -1))
saved = spec['cfg'].get('train_depth_occlusion_dir')
report = Path(saved)/'metadata.json' if saved else None
current = report and report.is_file() and json.loads(report.read_text()).get('version') == VERSION
print(saved if current else model.parent/f'da3_lidar_depth_stride3_v{VERSION}')
PY
)
mapfile -t inspect_settings <<< "$inspect_config"
inspect_depth="${DA3_DEPTH_DIR:-${inspect_settings[2]}}"
inspect_prepare=(--inspect-views 3)
inspect_viewer=()
if [[ -n "$inspect_image" ]]; then
    inspect_prepare=(--inspect-image "$inspect_image")
    inspect_viewer=(--image "$inspect_image")
fi

conda activate da3-depth
python -m pip install -r requirements-da3.txt
python scripts/check_da3_environment.py --gpu "$inspect_gpu"
python -u scripts/prepare_da3_occlusion.py \
    --source "${inspect_settings[0]}" --keyframes "$inspect_model/keyframes.json" \
    --point-cloud "$inspect_model/input.ply" --output "$inspect_depth" \
    --resolution "${inspect_settings[1]}" --process-res 756 --gpu "$inspect_gpu" \
    "${inspect_prepare[@]}"

conda activate 3dgs
python -m pip install 'viser==1.0.30'
python scripts/smoke_da3_occlusion.py --gpu "$inspect_gpu"
exec python -u scripts/inspect_occlusion.py --model "$inspect_model" \
    --depth-dir "$inspect_depth" --gpu "$inspect_gpu" --host 0.0.0.0 \
    --port "${OCCLUSION_INSPECTOR_PORT:-8082}" --check-gradients "${inspect_viewer[@]}"
