#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
scene=${ORANGE4F_SFM_SCENE:-/data/3dgs_outputs/orange4f/3dgs_finetuning/spirula_sfm_rgb_w1600}
scene=$(realpath -e -- "$scene")
original=$(realpath -e -- "$(cat "$scene/spirula-input.txt")/..")
out="$scene/superpoint_superglue_lidar"
mkdir -p -- "$out"
# Share the directory lock with the SIFT runner for the whole transaction.
exec 9<"$scene"
flock -n 9 || { echo "An SfM process already owns $scene" >&2; exit 1; }
if [[ -f "$scene/sparse/superglue_lidar.json" ]]; then
    echo "Already published: $scene/sparse/0"
    exit 0
fi
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    while read -r index; do
        index=${index// /}
        pids=$(nvidia-smi -i "$index" --query-compute-apps=pid --format=csv,noheader,nounits)
        if [[ -z "$pids" ]]; then CUDA_VISIBLE_DEVICES=$index; break; fi
    done < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
fi
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
[[ -n "$CUDA_VISIBLE_DEVICES" && "$CUDA_VISIBLE_DEVICES" != *,* ]] || { echo 'Select one GPU' >&2; exit 1; }
source scripts/spirula_runtime.sh
frontend=(python -u -m scripts.orange4f_learned_sfm)
args=(--scene "$scene" --original "$original" --work "$out")
if [[ -n "${ORANGE4F_LIDAR_PLY:-}" ]]; then args+=(--lidar "$ORANGE4F_LIDAR_PLY"); fi
if ! python -c 'import pycolmap as p; assert p.__version__=="3.11.1"' 2>/dev/null; then
    : "${CONDA_PREFIX:?Activate the 3dgs environment first}"
    deps="$CONDA_PREFIX/opt/orange4f-pycolmap-3.11.1"
    export PYTHONPATH="$deps${PYTHONPATH:+:$PYTHONPATH}"
    if ! python -c 'import pycolmap as p; assert p.__version__=="3.11.1"' 2>/dev/null; then
        python -m pip install --no-deps --target "$deps" pycolmap==3.11.1
    fi
fi
"${frontend[@]}" check "${args[@]}"
repo=${SUPERGLUE_REPO:-${XDG_CACHE_HOME:-$HOME/.cache}/3dgs/SuperGluePretrainedNetwork-ddcf11f}
if [[ ! -d "$repo" && -z "${SUPERGLUE_REPO:-}" ]]; then repo="$out/SuperGluePretrainedNetwork"; fi
bash scripts/setup_superglue.sh "$repo"
"${frontend[@]}" matches "${args[@]}" --repository "$repo" 2>&1 | tee -a "$out/learned.log"
if [[ ! -f "$out/mapped.ok" ]]; then
    mapped=$(mktemp -d "$out/map.XXXXXXXX")
    "${spirula[@]}" sfm map "$out/matches.bin" "$out/features" -o "$mapped" \
        --images "$original/images" --init-min-tri-angle 8 \
        --metric-gps none --exif-attitude none --sensor-gauge none --no-sensor-map \
        --no-final-principal-point --no-final-extra-params --ba-real cpu --ba-real-coarse cpu 2>&1 | tee -a "$out/mapper.log"
    "${frontend[@]}" mapped "${args[@]}" --model "$mapped"
fi
"${frontend[@]}" publish "${args[@]}" 2>&1 | tee -a "$out/lidar-ba.log"
