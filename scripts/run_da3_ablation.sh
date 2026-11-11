#!/usr/bin/env bash
# Sequential execution: a failed install, CUDA check or preparation stops here.
set -euo pipefail
source "${DA3_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
da3_gpu="${1:-3}"
if (( $# > 0 )); then shift; fi
da3_root="${DA3_OUTPUT_ROOT:-/data/3dgs_outputs/orange_4F/1_full}"
da3_source="${DA3_SCENE_ROOT:-/data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full}"

conda activate 3dgs
da3_base="${DA3_BASELINE:-}"
if [[ -z "$da3_base" ]]; then
    da3_base=$(python -c 'from scripts.render import resolve_models; import sys; print(resolve_models(None, sys.argv[1], ["sharp_keyframes_stride3_aa"])[0])' "$da3_root")
fi
da3_depth="${DA3_DEPTH_DIR:-$da3_root/da3_lidar_depth_stride3_v2}"
da3_model="${DA3_MODEL_DIR:-$da3_root/3dgs_stride3_aa_da3_v2_occlusion_warm7k_lr50}"

conda activate da3-depth
python -m pip install -r requirements-da3.txt
python scripts/check_da3_environment.py --gpu "$da3_gpu"
python -u scripts/prepare_da3_occlusion.py \
    --source "$da3_source" --keyframes "$da3_base/keyframes.json" \
    --point-cloud "$da3_base/input.ply" --output "$da3_depth" \
    --process-res 756 --resolution -1 --gpu "$da3_gpu"

conda activate 3dgs
python -u scripts/run_da3_occlusion.py \
    --source "$da3_source" --model "$da3_model" --baseline "$da3_base" \
    --depth-dir "$da3_depth" --gpu "$da3_gpu" "$@"
