# orange4f: four shared-input runs, AA off

Dataset: `/workspace/colmap_scenes/orange4f`. Run preparation once (the launcher
does this under a lock), then launch one job in each terminal.

| GPU | Run directory | XYZ | Growth | Steps |
| --- | --- | --- | --- | --- |
| 0 | `3dgs` | Native position LR | Native cloning/splitting | 30,000 |
| 1 | `3dgs_fixed_lidar` | Fixed at surviving initialization points | Disabled | 30,000 |
| 2 | `gsprior` | Released GSPrior updates and prior pull | Enabled | 40,000 |
| 3 | `gsprior_fixed_lidar` | Fixed; both Adam and prior pull disabled | Disabled | 40,000 |

All outputs go under:
`/data/3dgs_outputs/orange4f/3dgs_finetuning/stride3_noaa_v1/`.

## Inputs and baseline controls

- Natural filename ordering, consecutive groups of three registered frames, and
  the highest full-resolution grayscale Laplacian variance in each group.
  The last partial group is retained. A single manifest is reused across all runs.
- Every unselected registered image is held out. PSNR covers **every registered
  frame**, with selected/unselected results reported separately; it is not
  hard-coded to the old dataset's 5,009-frame count.
- All four models use one initialization cloud. Default discovery order:
  `input.ply`, then `points3D.ply`, `points3D.bin`, `points3D.txt` inside
  `sparse/0` or `sparse`. Preparation prints the exact selected path and point
  count. A COLMAP filename alone does not establish LiDAR provenance.
- Set `export ORANGE4F_POINT_CLOUD=/absolute/path/to/lidar.ply` in every terminal
  if the desired LiDAR cloud is separate. It must contain raw XYZ/RGB, not
  Gaussian scales/SH. The source cloud, camera files and images are never edited.
- No region crop, ROI mask, external occlusion culling, pose refinement, custom
  scale constraint, distance weighting, position warmup or LR reduction.
- Native 3DGS uses original LRs: XYZ 0.00016 to 0.0000016 times scene extent;
  feature 0.0025, opacity 0.05, scaling 0.005, rotation 0.001. Native density
  threshold 0.01, growth/pruning phase 500–15,000, interval 100.
- **Pruning and opacity resets remain enabled**, including in fixed-position
  runs. Fixed positions do not imply a constant point count: pruning can remove
  initialization rows, but cannot move surviving rows or add children.
- AA compensation is disabled in training, previews, all-frame evaluation and
  checkpoint comparisons. The original rasterizer's native minimum pixel
  footprint remains unchanged.

## Launch

In each terminal:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate 3dgs &&
cd ~/gaussian-splatting &&
git pull --ff-only
```

Then run the corresponding command (one per terminal):

```bash
bash scripts/run_orange4f.sh 0
bash scripts/run_orange4f.sh 1
bash scripts/run_orange4f.sh 2
bash scripts/run_orange4f.sh 3
```

Explicit dataset/version syntax:

```bash
bash scripts/run_orange4f.sh 0 /workspace/colmap_scenes/orange4f v2
```

All four commands must use the same version. Preparation is shared and locked.
A completed preparation is reused; conflicting cached inputs fail rather than
silently changing the experiment. An incomplete preparation needs a new version.
Outputs and shared caches are stored outside the dataset, including when the
source path is a symlink to read-only storage. To choose another output parent,
set `ORANGE4F_OUTPUT_ROOT` in every terminal, or pass `--output-root` to the
Python preparation script. The default parent is
`/data/3dgs_outputs/orange4f/3dgs_finetuning`.

Existing GPU processes are not killed. Training refuses to overwrite a run
already started.

## GSPrior implementation and adaptations

GPUs 2–3 use the released code at
https://github.com/takeshie/GSPrior, pinned to
`dcb7c89fb6b60f068b440de45d064ecc7fbcba55`, with an explicit integration adapter.
This is separate from this fork's older LiDAR-SCP approximation. It retains the
released plane renderer, photometric/geometric losses, opacity prior, schedules,
and direct pull coefficient. The prior is reconstructed from the model's
rendered training-view depth, as in the released method; raw LiDAR supplies
initialization.

The released schedule is 40k, with TSDF refreshes at 20k/30k/40k and position
pulling after 30k. All models therefore have a common 30k checkpoint, while
40k is additionally available for both GSPrior runs.

Necessary integration changes are recorded in `run_protocol.json`:

- Dataset paths/split replace the hard-coded DTU example.
- TSDF bounds cover the input point cloud and cameras in their original units.
  Requested voxel size is 0.02 scene units, automatically increased to fit at
  most 64M voxels. The **actual** voxel size, bounds and dimensions are in
  `suite.json`. Truncation is 96/48/24 voxels. This large-scene discretization
  differs from the DTU example; outliers can force coarse voxels.
- Correct XYZ-ordered trilinear sampling uses bounded GPU temporary storage.
  Fusion and TSDF sampling run on CUDA.
- The growth budget is `max(6,000,000, ceil(1.5 * initial_point_count))` so an
  initialization larger than the released 6M cap does not disable all growth.
  No initialization points are discarded to satisfy that budget.
- Direct pull modifies the XYZ Parameter in place, retaining Adam's reference.
  For GPU 3, XYZ gradients, XYZ LR, both growth operations and direct pull are
  disabled across both prior phases. Appearance, scales, rotation and opacity
  remain learnable.
- Fixes cover an upstream loader indentation error, RGB/plane gradient
  contiguity, debug-output unpacking, fusion bounds checking and total-population
  deletion detection.
- RGB loading is on demand. Only preview images are retained on CPU.

First launch on GPU 2 or 3 installs the official extension and fusion dependencies
in `~/.venvs/gsprior-orange4f`, inheriting the existing 3dgs environment's working
PyTorch. It does **not** replace native 3DGS's rasterizer. The cached checkout is
`~/.cache/3dgs/GSPrior-dcb7c89-orange4f-v1`.
Concurrent setup is locked. Overrides: `GSPRIOR_REPO`, `GSPRIOR_ENV`.

Before training, native/GSPrior launchers run their actual CUDA smoke checks.
GSPrior checks RGB and plane/depth backward, fixed XYZ through Adam and
population changes, and actual TSDF CUDA fusion/sampling. CPU validation on the
development host does not substitute for these GPU checks.

## Previews and evaluation

All runs save fixed random train/held-out GT, rendered and absolute-error images
at step 1, every 500 steps, and the final step. Gaussian counts appear in tqdm.

Every 10k steps, a completed checkpoint is published before a separate evaluator
on the **same GPU** loads it. Evaluation streams all frames and retains a sampled
train/held-out comparison report. It may lag training; GSPrior retries evaluation
OOM while training is active. Native runs use the existing periodic evaluation
queue. GSPrior uses its own official renderer for evaluation.

- `RUN/debug_previews/index.html`: live previews.
- `RUN/psnr_all_frames/iteration_10000/summary.json`: mean PSNR and split counts.
- `RUN/psnr_all_frames/iteration_10000/per_frame.csv`: every image's PSNR.
- `RUN/view_comparisons/iteration_10000/index.html`: sampled image comparisons.
- `RUN/point_cloud/iteration_10000/point_cloud.ply`: Gaussian export.
- `RUN/chkpnt10000.pth`: optimizer/model checkpoint.
- GSPrior: `RUN/prior_cache/iteration_20000/` contains the TSDF and its metadata.
- `RUN/training.log`: training and startup smoke results.
- Native evaluator: `RUN/periodic_evaluation/evaluator.log`.
- GSPrior evaluator: `RUN/evaluator.log`.

Serve the four-run index without using a GPU:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate 3dgs &&
python -m http.server 8098 --bind 0.0.0.0 \
  --directory /data/3dgs_outputs/orange4f/3dgs_finetuning/stride3_noaa_v1
```

Open `http://SERVER_IP:8098/`. Links populate when each run saves its first
preview. To manually evaluate a GSPrior checkpoint, use its dedicated entry point
rather than the native renderer:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate 3dgs &&
cd ~/gaussian-splatting &&
"$HOME/.venvs/gsprior-orange4f/bin/python" scripts/gsprior_entry.py eval \
  --repo "$HOME/.cache/3dgs/GSPrior-dcb7c89-orange4f-v1" \
  --model /data/3dgs_outputs/orange4f/3dgs_finetuning/stride3_noaa_v1/gsprior \
  --iteration 30000
```
