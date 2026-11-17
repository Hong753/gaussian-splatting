#!/usr/bin/env bash
# GPU MODEL [ITERATION] [OUTPUT]; recorded cameras with GT, then a novel path.
set -euo pipefail
if (($# < 2 || $# > 4)); then
  echo 'Usage: bash scripts/render_region_videos.sh GPU MODEL [ITERATION=30000] [OUTPUT]' >&2
  exit 2
fi
source "${RECONSTRUCTION_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
conda activate 3dgs
cd "$(dirname "${BASH_SOURCE[0]}")/.."
video_gpu="$1"
video_iteration="${3:-30000}"
video_model="$(realpath -e "$2")"
video_output="${4:-$video_model/videos/iteration_$video_iteration}"

# Read the saved source, and require a fully published snapshot before loading it
# while training may still be saving later checkpoints. No optimizer state load.
video_source="$(python - "$video_model" "$video_iteration" "$video_output" <<'PY'
from pathlib import Path
import sys
from scripts.eval_all_frames import read_config
from utils.checkpoint_snapshots import read_snapshot
model, iteration, output = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
if iteration < 1:
    raise SystemExit('Iteration must be positive')
if read_snapshot(model, iteration, verify=False) is None:
    raise SystemExit(f'Checkpoint is not published yet: {model}/point_cloud/iteration_{iteration}/ready.json')
cfg = read_config(model / 'cfg_args')
source = Path(cfg['source_path']).expanduser().resolve()
if not (source / 'selection.json').is_file():
    raise SystemExit(f'Expected an exported regional dataset: {source}')
for mode in ('actual', 'random'):
    destination = output / mode
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise SystemExit(f'Output already exists: {destination}; supply a fresh OUTPUT as the fourth argument')
print(source)
PY
)"

video_common=(--source "$video_source" --model "$video_model" --iteration "$video_iteration"
  --labels "$(basename "$video_model")" --aa on --occlusion off
  --width 1280 --fps 30 --gpu "$video_gpu" --in-process)
video_actual=(--video actual "${video_common[@]}" --actual-poses source --stride 1
  --output "$video_output/actual")
# Modest deviations suit a cropped region better than the whole-scene defaults.
# These camera-support constraints do not test mesh collisions.
video_random=(--video random "${video_common[@]}" --duration 40 --seed 42
  --source-window 180 --control-step 12 --deviation 0.5
  --max-distance-fraction 0.02 --view-degrees 5 --output "$video_output/random")

# Validate both plans before starting the first GPU render. No training stopped.
python -u scripts/render.py "${video_actual[@]}" --dry-run
python -u scripts/render.py "${video_random[@]}" --dry-run
python -u scripts/render.py "${video_actual[@]}"
python -u scripts/render.py "${video_random[@]}"
