#!/usr/bin/env bash
# Fresh RGB-only SfM. SOURCE and OUTPUT are scene roots; each has its own sparse/.
set -euo pipefail

src=$(realpath -e -- "${1:-/data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full}")
out=$(realpath -m -- "${2:-/data/3dgs_outputs/orange4f/3dgs_finetuning/spirula_sfm_rgb_w1600}")
if (( $# > 2 )) || [[ ! -d "$src/images" || "$out" == / || "$out" == "$src" || "$out" == "$src/"* || "$src" == "$out/"* ]]; then
    echo 'Usage: orange4f_spirula_sfm.sh [source scene with images/] [separate output scene]' >&2
    exit 1
fi
mkdir -p -- "$out"
exec 9<"$out"
flock -n 9 || { echo "An SfM process is using $out" >&2; exit 1; }
if [[ -f "$out/sparse/superglue_lidar.json" ]]; then
    echo "This workspace now contains the published SuperGlue/LiDAR model. Use a separate output for SIFT." >&2
    exit 1
fi

# A fresh workspace or one initialized by this launcher for the same photos.
if [[ -f "$out/spirula-input.txt" ]]; then
    [[ $(cat "$out/spirula-input.txt") == "$src/images" ]] || { echo "Different input recorded in $out" >&2; exit 1; }
elif [[ -n $(find "$out" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
    echo "Choose a fresh SfM output directory; $out already contains other files" >&2
    exit 1
else
    printf '%s\n' "$src/images" > "$out/spirula-input.txt"
fi
if [[ ! -e "$out/images" && ! -L "$out/images" ]]; then
    ln -s -- "$src/images" "$out/images"
fi
[[ "$out/images" -ef "$src/images" && ! -L "$out/sparse" ]] || { echo "Unexpected scene layout in $out" >&2; exit 1; }

source "$(dirname "${BASH_SOURCE[0]}")/spirula_runtime.sh"

cmd=("${spirula[@]}" sfm auto "$out/images" -o "$out" --data-type video --quality high
     --features sift --camera-model pinhole --max-image-size 1600 --no-masks
     --metric-gps none --exif-attitude none --sensor-gauge none --no-sensor-verify --no-sensor-map
     --device "${SPIRULA_DEVICE:-auto}")
{
    printf '%s\n' "$version_info"
    if devices=$("${spirula[@]}" sam devices 2>&1); then
        printf '%s\n' "$devices"
    elif [[ "$devices" == *"no Vulkan devices"* || "$devices" == *"VK_ERROR_INCOMPATIBLE_DRIVER"* ]]; then
        printf '%s\n' "$devices" 'Vulkan loaded, but device enumeration failed. Check NVIDIA Vulkan ICD visibility on this host. In an NVIDIA container, expose the graphics driver capability for Vulkan.'
    fi
    printf '%q ' "${cmd[@]}"; printf '\n'
} | tee -a "$out/sfm.log"
status=0
"${cmd[@]}" 2>&1 | tee -a "$out/sfm.log" || status=$?
if (( status == 0 || status == 3 || status == 4 )); then
    for file in cameras.bin images.bin points3D.bin; do
        [[ -s "$out/sparse/0/$file" ]] || { echo "Missing SfM model file: $out/sparse/0/$file" >&2; exit 1; }
    done
    printf 'Main model: %s/sparse/0\nSpirula exit status: %d\n' "$out" "$status" | tee -a "$out/sfm.log"
fi
exit "$status"
