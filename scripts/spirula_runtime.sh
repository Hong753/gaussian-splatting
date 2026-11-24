# Shared by the Orange4F SfM launchers. Sets the spirula command array.
bin=${SPIRULA_BIN:-}
if [[ -z "$bin" && ${spirula_use_pinned_release:-0} != 1 ]]; then bin=$(command -v spirula || true); fi
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
