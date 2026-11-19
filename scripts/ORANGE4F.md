# orange4f: shared-input runs, AA off

## Check image alignment with bounded pose refinement

In the `3dgs` environment, run `python -u scripts/orange4f_pose_ba.py` on CPU.
The pilot takes 48 training cameras from three trajectory sections and reserves
12 neighboring photos for the final check. It first matches RGB features across
views. Each track needs at least three training photos. The solver reserves
one observation per track and uses a consistent LiDAR plane when available.
For other tracks it triangulates a free 3D landmark from fitting observations
with at least two degrees of parallax, positive depth and consistent
reprojections. Camera priors tie the solution to the original metric trajectory.

`BA input` gives distinct LiDAR and RGB landmark counts. `Original RGB epipolar
error` measures the perpendicular distance of observed features to lines
predicted by the input poses and intrinsics. `Independent RGB epipolar error`
compares that distance before and after on the reserved photos. The separate
`Independent photos` check compares full 2D reprojection against LiDAR
surfaces. Acceptance requires sufficient map-supported coverage and improved
reprojection on held-out observations and independent photos.

The pilot reads compatible features and matches from v2 when available and
writes `pose_ba_pilot_v3/ba/{matching_report,refinement_report}.json` under the
Orange4F fine-tuning output root. An accepted fit exports the camera poses to
`ba/dataset/` with the original indexed LiDAR associations. Use `--full` for
the full training split in `pose_ba_full_v3/`.

## Measure which splats render the foreground floor

Run `python -u scripts/orange4f_splat_footprints.py` in the `3dgs` environment.
It renders the same floor patch in `09882.jpg` and `09884.jpg` using adaptive
3DGS and fixed-LiDAR 3DGS at their latest shared saved iteration. Use
`--gpu N` to choose a device or `--plain-only` for adaptive 3DGS alone.
When available, the separate 2,000-step one-view fit shows how much detail
transfers from its training frame to its neighbor.

Open `floor_splat_audit_v1/index.html` on the stride-3 dashboard. Start with the
rendered contrast at the marked bright pixels. If contrast is low, check the
contribution map for broad floor splats (blue) or centers outside the 2.5 cm
floor band (red). The measurements use front-to-back compositing weights and
the mutually LiDAR-supported pixels in the two photos. The results apply to
this floor patch and these checkpoints.

## Compare uniform and distance-weighted multiview floor training

Run `python -u scripts/orange4f_multiview_floor_distance.py` in the `3dgs`
environment. This fits two adaptive 3DGS models with identical original
cameras, initialization, crop masks, and numbers of visits per camera. Both
use only the LiDAR-supported part of the `09882.jpg` floor patch that the
selected views share. The launcher searches the full capture for distant
views, which by default must be at least twice as far from the patch as the
reference camera. It targets two near and two distant training views and
requires at least one of each. It prefers stride-3 photos for training and
can use other registered photos to reach the distant views. Evaluation uses
`09884.jpg` and a separate
distant camera. Each run uses 2,000 steps per selected training view. Use
`--gpu 6` to choose a GPU, `--prepare-only` to inspect the selection on CPU,
or `--far-min-ratio` to set the minimum distance ratio.

The uniform run gives each camera's mean RGB loss equal weight. The distance
run multiplies that same loss, including L1 and SSIM, by inverse-squared distance
to the center of the common floor patch, normalized to a mean of one over the
training cameras. Open `multiview_floor_distance_v2/index.html` on the suite
dashboard for the distances, weights, live previews, and final side-by-side
crops. The page puts the near and distant held-out photos first.

## Single-photo control

Run `python -u scripts/orange4f_one_view_weight.py` from the `3dgs` environment.
Use `--gpu 6` for a specific GPU. This runs two fresh adaptive fits sequentially on
`09882.jpg`, each for 2,000 steps from the same LiDAR cloud and seed with the
full stride-3 scene extent. The full photograph retains RGB supervision.
In the weighted arm the floor pixels used in the footprint audit receive extra
L1 weight. For a pixel covering area A on the LiDAR plane the weight is
`1 + min(1, A10 / A)`, where A10 is the 10th percentile on the supported patch.
All other pixels have weight 1. The weights are normalized to mean 1 over the image.
Both arms use the same SSIM loss. The weight map uses LiDAR geometry and camera
calibration. The neighbor's colors are used only to score its render.

Open `one_view_floor_weight_v1/index.html` on the stride-3 dashboard for live
previews, matched training crops and the final transfer check on `09884.jpg`.
The report includes the actual weight range, so nearly constant floor weights
can be interpreted as extra emphasis on the patch. This tests within-photo
weighting. Comparing close and distant observations of the same patch requires
a multi-view run.

## Fixed-LiDAR scale

Run `python -u scripts/orange4f_detail.py --estimate-fixed-scale` to print the
initial sigma and proposed ceiling distribution from the fixed baseline's
`input.ply`. Each anchor's sigma is the root mean square distance to its three
nearest other input points, with the native initialization floor. For clouds
larger than 100,000 anchors, the CPU summary queries a reproducible sample
against the full cloud.

Run `python -u scripts/orange4f_detail.py --train-fixed-scale` to train with a
per-axis ceiling of twice each anchor's own initial sigma. The existing fixed
LiDAR training mode keeps the entire input population and disables pruning and
opacity resets. `--fixed-scale-ratio` changes the proposed multiplier. The
default run saves 30k and 50k checkpoints in `fixed_lidar_knn2x_50k_v1`.
Running the script without these options runs the adaptive split comparison.
Both comparisons report the actual split of the saved floor photograph and
omit its crop if it cannot be identified.

To check whether the fixed-anchor recipe can fit an individual training image
with repeated supervision, run

```bash
python -u scripts/orange4f_detail.py --probe-image 09882.jpg
```

The probe uses the full input cloud, the same scale ceiling, and the original
camera and training resolution. It fits only the selected image for 2,000
steps and writes its own `debug_previews/index.html` and final comparison.
On the port 8098 suite dashboard, open **One-view training previews** after the
first snapshot. For an already running or completed probe, run
`python scripts/refresh_orange4f_dashboard.py` once and reload the dashboard.
The default probe page is `/one_view/fixed_lidar_one_view_09882_knn2x/index.html`
on that same server.

To compare the real speckled floor in `09882.jpg` with nearby original training
photos, run `python -u scripts/orange4f_floor_neighbors.py` on the Orange4F host.
The CPU test uses the suite's original poses, intrinsics and cloud. Open
`/floor_neighbors_09882_v2/index.html` on port 8098. Clicking a crop opens its
native pixels. The JSON includes all examined neighbors and their rejection reasons.

The plane fit uses the nearest return per occupied 4 px reference cell, with
free orientation in the saved coordinate system. The report overlays the
accepted and rejected frontmost samples and lists their camera depths. A plane
needs agreement with 60% of those samples and support in 12 of 16 crop regions.
If fewer than three views yield LiDAR-warp measurements, the report also checks
RGB repeatability. It fits a homography to half the mutual descriptor matches in
each quadrant and scores separate matches in reference pixels. The original-pose
epipolar error on these matches measures cross-line disagreement. A successful
RGB warp alone leaves the LiDAR depth and absolute alignment undetermined.

The test examines stride-3 training cameras within 48 registered frames on either
side, closest camera centers first. It checks crop overlap and similar sampling
in both image directions. Each warp must have at least 50% jointly supported
area and retain at least 55% of the reference crop's high-frequency response.
Up to six qualifying neighbors enter the matching test, independent of their
match scores. A cell needs a nearby floor return and any nearer surface return
outside the 2.5 cm plane tolerance vetoes that neighborhood. Both checks cover
3×3 cells of 8 native pixels per side in each camera. Sparse LiDAR can miss
occluders. A view needs 12 mutual matches, at least 25% of eligible features,
and matches in three crop quadrants. The summary requires at least three measured
views. The subpixel agreement and shift thresholds are diagnostic choices.

## PGSR with a fixed LiDAR warmup

From the activated `3dgs` environment, launch the guided run and its matched control
in separate terminals. Use distinct free GPUs if running both at once.

```bash
python -u scripts/run_orange4f_pgsr.py --variant guided --gpu 4
python -u scripts/run_orange4f_pgsr.py --variant covariance --gpu 5
```

Each launch uses the pinned zju3dv/PGSR trainer and builds its plane rasterizer
in an isolated cache. The active 3dgs environment supplies PyTorch, simple_knn,
SciPy, OpenCV, Pillow, tqdm and plyfile. Each run uses the fixed baseline's
input PLY, original camera poses, centered-camera convention, sharp stride-3
selection, full resolution and disabled exposure compensation. A fresh suite
uses the usual Orange4F preparation. Run with `--plan-only` to inspect the
input hashes and schedule before a GPU launch.

- Steps 1–7,000 freeze all centers and the population. Both variants fit
  24-neighbor LiDAR planes, align each trusted initial shortest axis to the
  normal, and set its sigma to 0.2 times the initial spacing. Each axis keeps
  the current fixed run's upper bound of twice its seed's initial 3-NN sigma.
  Color, opacity, rotation and scale learn during warmup.
- The XYZ LR ramps over the next 2,000 steps to PGSR's initial position LR
  divided by 50, multiplied by the training camera extent. It then decays to
  1/100 of that peak by step 50,000. PGSR's single- and multi-view losses
  start at 7,001. Its density-control clock starts after warmup, with growth
  events every 100 steps from 7,600 through 21,900. Pruning, multi-view
  trimming and opacity resets also use this shifted clock.
- The covariance control uses PGSR's learned ellipsoids to sample clone and
  split offsets, with ordinary Adam center updates. The guided run projects
  both kinds of child offsets onto each supported seed plane and retains 10%
  of the normal component of each actual Adam displacement. Both use ordinary
  PGSR updates where PCA confidence is below 0.6 or the parent center falls
  outside its original 24-neighbor support radius.
- Default outputs are `pgsr_lidar_guided_50k_v1` and
  `pgsr_lidar_covariance_50k_v1` under the Orange4F fine-tuning directory.
  Each writes fixed train and held-out GT/render/error previews at step 1 and
  every 500 steps. After training, the PGSR renderer evaluates all frames at
  30k and 50k, with separate split means in `psnr_all_frames` and image
  galleries in `view_comparisons`. Compare the same held-out views and floor
  texture across the two variants and the existing fixed-center run.
  Rerunning the same command resumes from the latest published checkpoint or
  finishes evaluation if training has completed.

## Camera trajectory and source-image video

From the activated `3dgs` environment, export the full capture to MP4 with

```bash
python -u scripts/render_pose_video.py
```

The exporter reads the full-scene poses and input point cloud recorded together
in `stride3_noaa_v1/suite.json`, or the full-scene baseline's saved inputs if the
suite manifest is absent. It draws the camera path in a fixed 3D overview
and pairs each moving frustum with the original image for that pose. It projects
every finite input point and retains the nearest point at each overview pixel.
Teal marks the geometric field of view up to the displayed depth, including
points behind intervening surfaces. The amber top edge shows camera roll.
The default video includes every registered image in natural filename order at
24 fps and is saved to
`/data/3dgs_outputs/orange4f/3dgs_finetuning/pose_trajectory.mp4`.
The export runs on CPU and requires FFmpeg with libx264.

For a clip starting at the floor diagnostic frame, run

```bash
python -u scripts/render_pose_video.py --start-frame 12569.jpg --max-frames 120 \
  --output /data/3dgs_outputs/orange4f/3dgs_finetuning/floor_trajectory.mp4
```

Use `--frustum-depth` to choose the drawing depth in reconstruction units.
To inspect a different reconstruction, pass `--source DATASET --point-cloud RAW.ply`
with both inputs in the same world frame.

## Scaffold-GS: two additional runs

These extend an existing prepared suite without modifying the first four runs,
their inputs or their manifests. The launchers prepare the shared suite if needed.
Use two terminals:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate 3dgs &&
cd ~/gaussian-splatting &&
git pull --ff-only &&
bash scripts/run_orange4f_scaffold.sh 4 learnable
```

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate 3dgs &&
cd ~/gaussian-splatting &&
git pull --ff-only &&
bash scripts/run_orange4f_scaffold.sh 5 fixed
```

| Setting | `scaffold_learnable_anchors_v1` | `scaffold_fixed_anchors_v1` |
|---|---|---|
| Initial anchors | Same voxelized shared LiDAR cloud | Same |
| Anchor XYZ | Learned, LR 0.00016 → 0.0000016, multiplied by scene extent | Frozen, no XYZ gradient or Adam updates |
| Anchor growth | Scaffold hierarchy, steps 1500–15000 | Disabled |
| Anchor pruning | Scaffold opacity/observation criterion | Same; surviving anchor coordinates stay fixed |
| Neural offsets, features, scale and MLPs | Learned | Learned |
| AA / extra culling / exposure embeddings | Off | Off |
| Duration / previews / saved checkpoints | 30k / every 500 / every 10k | Same |

The stock Scaffold anchor LR in this fork is zero: its geometry normally learns
through offsets and anchor growth. The **learnable-anchor variant is an explicit
ablation**, using native 3DGS's XYZ LR endpoints; other Scaffold optimizer and
loss defaults remain in place. Scaffold uses its own opacity accumulation and
pruning, not 3DGS's periodic opacity resets.

Both variants round the input points to a **0.02 scene-unit voxel grid** and
deduplicate occupied cells. That is 2 cm only if the input coordinates are in
metres. This differs from the older Scaffold launcher’s 0.001-unit default and
reduces redundant anchors for a dense LiDAR cloud. This is anchor spacing,
**not a Gaussian scale cap**. It does not retain every raw point. Fixed anchors
are fixed grid-cell representatives, not the original unrounded LiDAR coordinates.

Rendered centers are `anchor + offset × learned_scale[:3]`, so freezing anchors
does **not** freeze all neural Gaussian centers. Offsets remain trainable in both
runs. Counts show anchors and per-view active Gaussians separately; evaluation
history's Gaussian capacity is `anchors × 10`, not a view-independent active count.

Set `SCAFFOLD_VOXEL_SIZE` before the first launch to choose another grid spacing.
Use the same value for both variants. To change it after preparing a run, also
set `SCAFFOLD_RUN_VERSION=v2`; existing plans and checkpoints are never overwritten.
The third/fourth positional arguments select source and suite version, just as
for the four original runs. GPU numbers are arguments, not tied to the variant.

No dependency installation or compilation is performed. These runs use this
repository's Scaffold implementation, the installed native rasterizer, and the
existing Git-installed `simple_knn` in `3dgs`. A small, clearly labelled synthetic
CUDA preflight runs first; the real voxelized anchor count is printed afterwards.

Outputs live under
`/data/3dgs_outputs/orange4f/3dgs_finetuning/stride3_noaa_v1/`.
Each new run has `debug_previews/index.html` and
`periodic_evaluation/index.html`. Every 10k, a separate process on the same GPU
evaluates **all registered source frames**, with selected/unselected PSNR and
GT/render/error galleries. It can lag training and retry after an evaluator OOM.
The suite index links both new preview pages.

Keep the saved anchor PLY **and all three MLP files** together. Ordinary Gaussian
PLY viewers cannot display this neural representation directly; use the existing
`scripts/view_scaffold.py` viewer or the image reports. Resume optimizer state via
`models/train_scaffold.py --start_checkpoint RUN/chkpnt10000.pth` with matching
original training options; do not resume over already-published later checkpoints.

## Original four runs

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
PyTorch and Git-installed **simple_knn**. Setup verifies that both environments
resolve the same simple_knn extension; it never builds or installs GSPrior's
bundled simple-knn. Only the plane rasterizer is built from GSPrior.
It does **not** replace native 3DGS's rasterizer. The cached checkout is
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

Serve the suite index without using a GPU:

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
