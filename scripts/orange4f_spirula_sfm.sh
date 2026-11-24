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

bin=${SPIRULA_BIN:-$(command -v spirula || true)}
if [[ -z "$bin" ]]; then
    version=2026.11.12
    cache="${XDG_CACHE_HOME:-$HOME/.cache}/orange4f/spirula-$version"
    bin="$cache/spirula"
    if [[ ! -x "$bin" ]]; then
        mkdir -p -- "$cache"
        tmp=$(mktemp -d "$cache/install.XXXXXXXX")
        trap 'rm -rf -- "$tmp"' EXIT
        curl -fL --retry 3 "https://github.com/harry7557558/spirula-studio/releases/download/v$version/spirula-$version-ubuntu-vulkan-x86_64.zip" -o "$tmp/release.zip"
        (cd "$tmp" && sha256sum -c <<<"07afb39882f336555633a17023a18c52502dcf0a18b23fe62fa5dd1c24b19a93  release.zip")
        unzip -q "$tmp/release.zip" spirula -d "$tmp"
        chmod +x "$tmp/spirula"
        mv -f -- "$tmp/spirula" "$bin"
    fi
fi

# A working CUDA environment may lack the system Vulkan loader. Probe the same
# executable that will run SfM, and keep any fallback local to its processes.
spirula=("$bin")
if ! version_info=$("$bin" sfm --version 2>&1); then
    if [[ "$version_info" != *"libvulkan.so.1: cannot open shared object file"* ]]; then
        printf '%s\n' "$version_info" | tee -a "$out/sfm.log" >&2
        exit 1
    fi
    conda_lib="${CONDA_PREFIX:-}/lib/libvulkan.so.1"
    if [[ -n "${CONDA_PREFIX:-}" && -r "$conda_lib" ]] &&
       version_info=$(env "LD_PRELOAD=$conda_lib${LD_PRELOAD:+:$LD_PRELOAD}" "$bin" sfm --version 2>&1); then
        spirula=(env "LD_PRELOAD=$conda_lib${LD_PRELOAD:+:$LD_PRELOAD}" "$bin")
    else
        loader_dir="${XDG_CACHE_HOME:-$HOME/.cache}/orange4f/vulkan-loader-1.3.204.1"
        if [[ ! -s "$loader_dir/libvulkan.so.1" ]]; then
            echo "Fetching the Ubuntu Vulkan loader into $loader_dir" | tee -a "$out/sfm.log"
            mkdir -p -- "$loader_dir"
            (
                stage=$(mktemp -d "$loader_dir/install.XXXXXXXX")
                trap 'rm -rf -- "$stage"' EXIT
                cd "$stage"
                curl -fL --retry 3 https://archive.ubuntu.com/ubuntu/pool/main/v/vulkan-loader/libvulkan1_1.3.204.1-2_amd64.deb -o loader.deb
                sha256sum -c <<<"192adcff489996b3398e7e7c0012b98e9586b46fe9a9eb13fb02c0feba88548b  loader.deb"
                dpkg-deb -x loader.deb extracted
                cp -L -- extracted/usr/lib/x86_64-linux-gnu/libvulkan.so.1 libvulkan.so.1
                mv -fT -- libvulkan.so.1 "$loader_dir/libvulkan.so.1"
            )
        fi
        spirula=(env "LD_LIBRARY_PATH=$loader_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$bin")
        version_info=$("${spirula[@]}" sfm --version)
    fi
fi

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
