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

# Publish a complete, verified runtime atomically. Invoking its own ELF loader
# keeps libc and ld-linux paired. No library path is exported to shell tools.
install_runtime() (
    mkdir -p -- "$runtime_cache"
    exec 8<"$runtime_cache"
    flock 8
    [[ ! -x "$runtime_loader" ]] || exit 0
    stage=$(mktemp -d "$runtime_cache/install.XXXXXXXX")
    trap 'rm -rf -- "$stage"' EXIT
    cd "$stage"
    echo 'Preparing the private Ubuntu 24.04 runtime for Spirula' >&2
    while read -r digest package; do
        archive=${package##*/}
        if ! curl -fsSL --retry 2 "https://archive.ubuntu.com/ubuntu/$package" -o "$archive"; then
            curl -fsSL --retry 2 "https://snapshot.ubuntu.com/ubuntu/20261123T000000Z/$package" -o "$archive"
        fi
        printf '%s  %s\n' "$digest" "$archive" | sha256sum -c -
        dpkg-deb -x "$archive" root
    done < "$runtime_manifest"
    mv -T -- root "$runtime_root"
)

spirula=("$bin")
probe_spirula() {
    version_info=$(LC_ALL=C "${spirula[@]}" sfm --version 2>&1)
}

select_runtime() {
    if probe_spirula; then return; fi

    # Reuse a previously working Vulkan-only fix on hosts with a recent libc.
    if [[ "$version_info" == *"libvulkan.so.1: cannot open shared object file"* ]]; then
        local loader
        for loader in "${CONDA_PREFIX:+$CONDA_PREFIX/lib/libvulkan.so.1}" \
                      "$cache_base/vulkan-loader-1.3.204.1/libvulkan.so.1"; do
            [[ -n "$loader" && -r "$loader" ]] || continue
            spirula=(env "LD_PRELOAD=$loader${LD_PRELOAD:+:$LD_PRELOAD}" "$bin")
            if probe_spirula; then return; fi
        done
    fi
    case "$version_info" in
        *GLIBC_*|*GLIBCXX_*|*CXXABI_*|*"cannot open shared object file"*)
            printf '%s\n' "$version_info" >> "$out/sfm.log"
            echo 'Selecting the private Ubuntu 24.04 runtime for Spirula' | tee -a "$out/sfm.log" ;;
        *)
            printf '%s\n' "$version_info" | tee -a "$out/sfm.log" >&2
            return 1 ;;
    esac
    local runtime_manifest runtime_cache runtime_root runtime_lib runtime_loader digest
    runtime_manifest=$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")/spirula_ubuntu_amd64.sha256")
    digest=$(sha256sum < "$runtime_manifest")
    runtime_cache="$cache_base/spirula-runtime"
    runtime_root="$runtime_cache/${digest%% *}"
    runtime_lib="$runtime_root/usr/lib/x86_64-linux-gnu"
    runtime_loader="$runtime_lib/ld-linux-x86-64.so.2"
    install_runtime
    # ld-linux requires a path to the ELF, including when SPIRULA_BIN is a command name.
    local executable
    executable=$(realpath -e -- "$(command -v -- "$bin")")
    spirula=("$runtime_loader" --library-path "$runtime_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$executable")
    if ! probe_spirula; then
        printf '%s\n' "$version_info" | tee -a "$out/sfm.log" >&2
        return 1
    fi
}

cache_base=$(realpath -m -- "${XDG_CACHE_HOME:-$HOME/.cache}/orange4f")
select_runtime

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
