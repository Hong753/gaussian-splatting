#!/usr/bin/env bash
# Six independent rendering jobs, one per physical GPU (0..5).
set -euo pipefail

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run && "$1" != --help ) ]]; then
  printf 'Usage: bash scripts/render_proxygs_reports.sh [--dry-run]\n' >&2
  exit 2
fi
if [[ ${1:-} == --help ]]; then
  cat <<'HELP'
Render both orange_4F Proxy-GS runs and their matched 3DGS+AA comparisons.
GPU 0/1: all-frame PSNR + images (600 keyframes / stride 3).
GPU 2/3: shared random videos (600 keyframes / stride 3).
GPU 4/5: GT + captured-trajectory videos (600 keyframes / stride 3).
Optional environment overrides: SCENE, MODEL_ROOT, OUTPUT_ROOT, ITERATION,
NATIVE_PYTHON, PROXY_PYTHON. --dry-run validates all jobs without rendering.
HELP
  exit 0
fi

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE="${SCENE:-/data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full}"
MODEL_ROOT="${MODEL_ROOT:-/data/3dgs_outputs/orange_4F/1_full}"
ITERATION="${ITERATION:-30000}"
NATIVE_PYTHON="${NATIVE_PYTHON:-/opt/anaconda3/envs/3dgs/bin/python}"
PROXY_PYTHON="${PROXY_PYTHON:-$HOME/.venvs/proxygs/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MODEL_ROOT/comparisons/proxygs_${ITERATION}_$(date -u +%Y%m%dT%H%M%S)}"
BASELINES=(kf600_original sharp_keyframes_stride3_aa)
PROXIES=(proxygs_kf600_original proxygs_stride3_original)
NAMES=(kf600_all stride3_all kf600_random stride3_random kf600_actual stride3_actual)

build_command() {
  local gpu="$1" pair=$(( $1 % 2 ))
  if (( gpu < 2 )); then
    COMMAND=("$PROXY_PYTHON" -u "$REPO_DIR/scripts/render.py"
      --model-root "$MODEL_ROOT" --runs "${PROXIES[$pair]}"
      --split all --iteration "$ITERATION" --gpu "$gpu" --in-process)
  else
    local mode=actual
    if (( gpu < 4 )); then mode=random; fi
    COMMAND=("$NATIVE_PYTHON" -u "$REPO_DIR/scripts/render.py" --video "$mode"
      --source "$SCENE" --model-root "$MODEL_ROOT"
      --runs "${BASELINES[$pair]}" "${PROXIES[$pair]}"
      --labels '3DGS + AA' 'Proxy-GS' --aa on off --iteration "$ITERATION"
      --gpu "$gpu" --width 1280 --fps 30 --output "$OUTPUT_ROOT/${NAMES[$gpu]}")
    if [[ $mode == random ]]; then
      COMMAND+=(--duration 40 --seed 42 --deviation 3 --max-distance-fraction 0.10 --view-degrees 12)
    else
      COMMAND+=(--actual-poses source)
    fi
  fi
}

# Validate every checkpoint/source/environment before starting any GPU work.
for gpu in {0..5}; do
  build_command "$gpu"
  printf 'Checking GPU %s: %s\n' "$gpu" "${NAMES[$gpu]}"
  CUDA_VISIBLE_DEVICES="$gpu" "${COMMAND[@]}" --dry-run >/dev/null
  if [[ ${1:-} == --dry-run ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; fi
done
if [[ ${1:-} == --dry-run ]]; then exit 0; fi
mkdir -p "$OUTPUT_ROOT"
mkdir "$OUTPUT_ROOT/logs" # Refuse reuse of this launch's logs/output directory.

pids=()
trap 'trap - INT TERM; kill "${pids[@]}" 2>/dev/null || true; wait || true; exit 130' INT TERM
for gpu in {0..5}; do
  build_command "$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "${COMMAND[@]}" >"$OUTPUT_ROOT/logs/${NAMES[$gpu]}.log" 2>&1 &
  pids+=("$!")
  printf 'Started GPU %s: %s (PID %s)\n' "$gpu" "${NAMES[$gpu]}" "$!"
done
printf 'Logs and videos: %s\n' "$OUTPUT_ROOT"
status=0
for gpu in {0..5}; do
  if wait "${pids[$gpu]}"; then
    printf 'Finished GPU %s: %s\n' "$gpu" "${NAMES[$gpu]}"
  else
    printf 'FAILED GPU %s: see %s/logs/%s.log\n' "$gpu" "$OUTPUT_ROOT" "${NAMES[$gpu]}" >&2
    status=1
  fi
done
exit "$status"
