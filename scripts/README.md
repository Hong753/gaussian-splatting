# Training

For trajectory-spanning DA3 groups inspired by PlanarGS geometry priors and
diagnostics for empty depth references, see [DA3_TRAJECTORY.md](DA3_TRAJECTORY.md).

For fixed whole-scene surfaces, precomputed visible/occluded LiDAR IDs per view,
and native 3DGS + AA culling at training and novel poses, see
[SCENE_OCCLUSION.md](SCENE_OCCLUSION.md).

For the independent LiDAR-calibrated Depth Anything 3 occlusion experiment,
see [DA3_OCCLUSION.md](DA3_OCCLUSION.md). It includes conda setup, preprocessing,
the warmup/reduced-LR training command, calibration previews, and automatic
all-frame evaluation and comparison videos.

Edit the defaults at the top of `scripts/train.py` (especially `DATA_ROOT` and
`SCENES`), then run `python scripts/train.py`. Or provide settings as arguments:

```bash
python scripts/train.py --data-root /path/to/scenes --scenes scan65 scan69 \
  --output-root output --name baseline --gpu 0 --iterations 30000
```

The launcher trains scenes sequentially, with outputs in
`OUTPUT_ROOT/SCENE/NAME`. It uses the active Python environment and the native
`train.py` implementation. The scene must have a COLMAP `sparse/0` or `sparse` model and
images, or a supported NeRF synthetic dataset. Use `--dry-run` to check the
commands, and `--help` for options.

Scenes may be nested paths relative to `--data-root`, such as
`--scenes orange_4F/1_full`; this preserves that path under the output root.
If `--data-root` itself is a scene, use `--scenes .`; its folder name is used
as the output scene name.

The launcher enables Graphdeco's anti-aliasing and exposure compensation by
default. Use `--no-antialiasing` or `--no-exposure-compensation` to disable
them. Exposure uses the native `--train_test_exp` option. Since that option
changes the test split, `--eval` requires `--no-exposure-compensation`.
For ordinary training without a held-out split, exposure uses the full image.

Native `render.py` requires `--antialiasing` separately when rendering a model
trained with anti-aliasing. Per-image exposures are saved to `exposure.json`;
they are not baked into the Gaussian PLY.

The default optimizer remains Graphdeco's regular Adam. Optionally specify
`--optimizer-type sparse_adam` after installing the compatible accelerated
rasterizer. Depth regularization is off unless `--depths` is supplied; it
requires the appropriate depth maps and COLMAP depth parameters.

Pass additional native options after `--`:

```bash
python scripts/train.py --scenes scan65 -- --densify_grad_threshold 0.0001
```

The live SIBR viewer socket uses port 6009 by default. For concurrent runs,
set a different `--viewer-port` for each run and connect the SIBR remote
viewer to the same port. Use `--disable-viewer` when live viewing is not
needed. Free-viewpoint GUI renders do not use per-image exposure transforms;
training image exposure optimization is unaffected.

The launcher refuses to use a nonempty output directory unless
`--allow-existing` is specified; that option does not resume training by
itself. Native `--start_checkpoint` is required to resume a checkpoint.

## Actual images versus rendered train/eval views

COLMAP training and `--evaluate-only` now produce image comparisons by default,
before the existing all-frame PSNR evaluation. `--no-render-comparisons` explicitly
opts out. Future configs save the anti-aliasing setting as well.

```bash
python scripts/render.py train --model /path/to/run --gpu 2
python scripts/render.py --split both --models /path/to/run_a /path/to/run_b --gpu 2
python scripts/render.py --eval --model /path/to/run --iteration 7000 --gpu 2
```

`train` contains the images actually selected for training. `eval` contains the
original lexical 1-in-8 held-out split when enabled; unused training candidates
are not relabeled as eval images. `all` includes every original registered COLMAP
image, including unstaged/unselected frames for Proxy-GS. `both` is the default. Comparing runs uses
their shared image names and latest common saved iteration, or `--iteration N`.
Each run uses its own saved source calibration, which preserves the intended
original-versus-refined-pose comparison. Different ground-truth pixels or image
sizes are rejected rather than shown against a misleading shared target.

For the current two keyframe runs, locate their nested output directories by name:

```bash
python scripts/render.py \
  --model-root /data/3dgs_outputs/orange_4F/1_full/keyframe_ablation \
  --runs kf600_original kf600_refined --train --gpu 2 --antialiasing \
  --output /data/3dgs_outputs/orange_4F/1_full/keyframe_ablation/train_comparison
```

Open `OUTPUT/index.html`. Each split contains an image browser with
`actual | run A | run B`, full-resolution PNGs, absolute-error maps (displayed at
4×), labeled comparison JPEGs, `per_frame.csv`, and `summary.json`. The browser
can sort by lowest PSNR. Metrics use mean per-channel PSNR, matching
`scripts/eval_all_frames.py`, and base appearance without per-image exposure.
Rendering streams one camera and one model at a time. `--max-views N` optionally
samples uniformly from the shared split; zero (default) renders every view.

The default output is `RUN/view_comparisons/iteration_N` for a single model and
a directory under the common parent for multiple models. `--dry-run` checks
selection/checkpoints without loading CUDA or creating outputs. Older configs
without an AA field default to AA for 3DGS/Scaffold; override explicitly with
`--no-antialiasing` when appropriate. Proxy-GS uses its own rasterizer without AA.

For Proxy-GS, render and evaluate **every original frame** directly in its environment:

```bash
~/.venvs/proxygs/bin/python scripts/render.py --model /path/to/proxygs_run \
  --split all --iteration 30000 --gpu 6 --in-process
```

This produces GT/render/error images and a browser under
`RUN/view_comparisons/iteration_30000/all/`, plus
`RUN/psnr_all_frames/iteration_30000/{summary.json,per_frame.csv}`. The summary
separates all, selected training, and unselected frames. It is written only when
the complete original camera set was rendered; sampled or shared subsets never
overwrite it. Metrics are computed before PNG/video encoding, at the model's saved
image resolution, without per-image exposure. Proxy depth is rasterized at each
requested camera; no new training or depth-cache preparation is needed.

## Proxy-GS trial with a LiDAR surface proxy

`scripts/run_proxygs.py` integrates the [official Proxy-GS implementation](https://github.com/Visionary-Laboratory/Proxy-GS)
at commit `eac937e8f34ab46690d73cf4e4d0217ae32d645e`. It preserves the selected
training images, held-out split, original camera poses, and input LiDAR cloud.
The official anchor/MLP representation is initialized from the LiDAR geometry;
this is a method comparison, not a visibility-only ablation of vanilla 3DGS.
Its optimizer defaults remain upstream defaults, with a 30k iteration limit and
saved/validated checkpoints at 7k and the final iteration.

Set up once from the CUDA-enabled `3dgs` environment (Python >=3.11, a matching
CUDA toolkit/compiler, and Git are required):

```bash
conda activate 3dgs
python scripts/run_proxygs.py setup
python scripts/run_proxygs.py run \
  --source /data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full \
  --keyframes /data/3dgs_outputs/orange_4F/1_full/keyframes_coverage600/keyframes.json \
  --model /data/3dgs_outputs/orange_4F/1_full/keyframe_ablation/proxygs_kf600_original \
  --iterations 30000 --gpu 4
```

The isolated environment at `~/.venvs/proxygs` reuses the active environment's
PyTorch but installs Proxy-GS's extensions separately. The pinned checkout is
at `~/Proxy-GS`; both locations can be overridden. The launcher replaces its own
process when switching environments and starting the official trainer; there is
no Python supervisor and training exceptions appear directly. On success, the
trainer replaces itself with the train/eval reporter, which uses `--in-process`
rendering. Each replacement releases the preceding stage's CPU/CUDA allocations.
`--dry-run` prints commands. Use `prepare` with the same arguments to build just
the proxy/depth cache.

Setup includes the missing `<cfloat>` header in upstream `simple-knn`, fixing
`identifier "FLT_MAX" is undefined` on newer CUDA toolchains. If an earlier setup
failed at that error, pull the update and rerun `setup`; it reuses the checkout
and environment and applies the fix before rebuilding the extension.

For an already-installed version, apply the Python memory/runtime fixes without
reinstalling packages or rebuilding extensions:

```bash
python scripts/run_proxygs.py patch
```

Anchor growth pools candidate features in chunks instead of repeating every
anchor feature for all offsets. It computes only selected offset positions and
uses integer-row membership instead of pairwise voxel comparisons. Candidate
selection and feature maxima are preserved. The default feature-gather chunk is
65,536 candidates (8 MiB for 32 float32 features); output features, model state,
and optimizer state still need memory proportional to the anchor count.

`--voxel-size` controls **only the proxy mesh**. Anchor spacing is controlled by
upstream `base_layer` and `levels`, exposed as `--base-layer` and `--levels`.
The defaults remain 10 and adaptive, respectively. Reducing `--base-layer` from
10 to 9 doubles the base anchor spacing for the same scene bounds; it can reduce
anchor count at the cost of spatial detail. Do not change this to work around
the unnecessary feature-replication allocation before applying the memory fix.

After a failure before the first saved checkpoint, repeat the `run` command with
`--restart`. This restarts training at step 0, reuses the proxy mesh/depth cache,
and moves previous logs/configs into `RUN/failed_attempts/`. Saved checkpoints
are protected: a run containing them requires a new output directory.

The default proxy uses 5 cm LiDAR voxel sampling, estimated normals, and ball
pivoting, discarding long triangle edges. This deliberately leaves unsupported
regions open; gaps in the mesh can still allow occlusion leakage. A registered
triangle mesh can instead be supplied with `--mesh PATH`. `--voxel-size` is in
the input map's units (assumed metres), and affects proxy construction, not the
official model's anchor resolution.

Preparation stages symlinks and NeRF-format camera transforms under
`RUN/proxygs_data`, leaving the source dataset untouched. It caches frontmost
camera-z proxy depths using nvdiffrast's CUDA rasterizer, including on A100 GPUs.
It follows the same centered projection as the baseline. Inspect
`proxygs_data/depth/*_rgb_depth.jpg` and `depth/metadata.json` for registration and
per-frame coverage; black pixels in the depth preview have no mesh support.
Preparation fails if every training view has zero proxy coverage.

The adapter avoids upstream COLMAP-loader incompatibilities and applies small
checked integration fixes: honor the requested GPU, load test-view depth caches,
accept empty test splits, fail clearly on empty Gaussian renders, and use our
streaming final comparison report. The memory changes preserve anchor selection,
feature pooling, and proxy culling. After training, `scripts/render.py --split both` runs automatically
in the Proxy-GS environment and keeps proxy culling enabled during rendering.
The same entry point can also compare the Proxy-GS output with native runs.

## Keyframe selection

Add `--select-keyframes` to score registered COLMAP training images in natural
filename order and keep the sharpest per three consecutive images (or set
`SELECT_KEYFRAMES = True` at the top of the launcher). Change the group size
with `--keyframe-group-size`. The score is full-resolution grayscale Laplacian
variance; ties go to the first frame. The final incomplete group is included.

With `--eval`, the native lexicographic 1-in-8 test split is held out first;
only the remaining training images are grouped, and test cameras stay intact.
The chosen names, every candidate's score and settings are written to
`OUTPUT_ROOT/SCENE/NAME/keyframes.json` before training. The selection does not
change the input images or COLMAP model. The original 3DGS point-cloud cache
behavior is unchanged.

Use `--selection-only` together with `--select-keyframes` to select without
training. To train afterwards, run the same command without `--selection-only`
and add `--allow-existing`; the existing matching manifest is reused. The
option currently supports COLMAP scenes, not NeRF synthetic scenes.

### LiDAR coverage keyframes and constrained pose refinement

The separate CPU pipeline below selects reusable frames from the complete
registered image set, then optionally refines just their poses. It does not
require a COLMAP database or existing feature tracks. Dependencies are NumPy,
SciPy, OpenCV (SIFT), and plyfile; the CUDA training stack is not imported.

```bash
python -m pip install scipy opencv-python plyfile

SCENE=/data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full
KF=/data/3dgs_outputs/orange_4F/1_full/keyframes_coverage600
REFINED=/data/3dgs_outputs/orange_4F/1_full/keyframes_coverage600_refined

python scripts/select_keyframes.py --source "$SCENE" --output "$KF" \
  --target-frames 600 --eval

python scripts/refine_keyframe_poses.py --source "$SCENE" \
  --keyframes "$KF/keyframes.json" --output "$REFINED"
```

`--eval` excludes the original lexicographic 1-in-8 test images **before**
selection. Omit it to use all registered frames as candidates. Keep this choice
identical for comparisons; use launcher `--eval` when the selection used it.
`--exclude-list names.txt` additionally reserves arbitrary filenames. All names
are relative to the image directory and include extensions/subdirectories.

Selection writes:

| File | Contents |
| --- | --- |
| `keyframes.json` | Selected filenames, selection settings, scores, coverage counts and excluded filenames; accepted by the training launcher |
| `image_list.txt` | Selected filenames only, one per line |
| `coverage_cache/` | Per-image scores and approximate LiDAR visibility for resumable selection |

The selector compares sharpness against nearby frames, discounts clipped pixels
and distant observations, and greedily adds coverage of LiDAR surface patches
and viewing directions. Repeated views of an already well-observed patch have
diminishing value. It also seeds each block of 30 candidate images, plus stream
ends, to retain trajectory coverage. This is a block safeguard, not a strict
30-frame bound between consecutive selections or a guarantee of visual overlap.
Natural filename order **within each parent directory** defines each camera
stream; interleaved cameras should be placed in separate directories. The
default budget is 600, capped by available eligible frames; selection can stop
earlier if additional frames add no coverage or observation quality. If trajectory
seeds exceed the budget, increase `--max-gap` or the budget. No images are copied.

Refinement is a SciPy sparse bundle adjustment that reads/writes COLMAP models,
with extra LiDAR constraints. It is not ordinary unconstrained COLMAP BA:

1. Extract RootSIFT features on selected images, match temporal/spatial neighbor
   pairs, and reject matches inconsistent with the existing poses and LiDAR.
2. Associate features with ray intersections on locally planar LiDAR patches.
   A point z-buffer filters occlusions, but sparse clouds cannot guarantee full
   visibility. Nonplanar neighborhoods and grazing rays are excluded.
3. Construct tracks with at least three images and optimize camera poses plus
   their 3D landmarks. Intrinsics and LiDAR points stay fixed. Point-to-plane
   residuals tie landmarks to the map; finite patch support and pose priors
   prevent large drift. Defaults assume metres: 3 cm translation-prior std,
   0.2 degree rotation-prior std, 1.5 cm surface residual std. Hard pose/landmark
   bounds further restrict updates. Scale these flags if the map uses other units.
4. Withhold one observation per track from both landmark initialization and BA.
   Compare against a baseline that fits landmarks while fixing the original
   poses. Reject the entire export if held-out median error does not improve,
   the p90 degrades beyond a small tolerance, or a sufficiently supported image
   degrades substantially.
   Cameras with fewer than `--min-observations` training observations stay fixed.

Both steps require undistorted `PINHOLE`/`SIMPLE_PINHOLE` images at their COLMAP
dimensions and a LiDAR PLY already in the **same world frame** as the camera
poses. They use `sparse/0/points3D.ply` (or flat `sparse/points3D.ply`) by default;
pass `--point-cloud /path/to/lidar.ply` to both commands when the real map is
elsewhere. Selection and refinement use voxel representatives for efficiency;
the exported training initialization uses the original full PLY. It must have
the RGB and normal fields expected by this repository's training loader.

An accepted refinement creates `REFINED/dataset/` with standard COLMAP binary
cameras, images and newly constructed feature tracks, a symlink to the original
LiDAR PLY, a symlink to all source images, and its own `keyframes.json`. All
original camera entries are retained, including unselected/test poses, so frame
ordering and evaluation splits stay unchanged. Only supported selected poses
are refined. Original images, poses, and LiDAR files are never modified.

`matching_report.json` reports usable pairs/tracks and feature counts;
`refinement_report.json` reports before/after withheld reprojection error,
per-image pose changes and acceptance. Pixel errors are at the feature working
resolution (maximum side 1600 by default). Rejection exits nonzero and produces
no training dataset. Do not loosen thresholds just to force an export: inspect
coverage, feature support, map alignment and image quality first. Pure planes
and repetitive/textureless images may not constrain every pose direction.
This check establishes improved image consistency, not ground-truth accuracy.
It does not estimate rolling shutter, rig calibration, exposure, or LiDAR drift.

Caches can be reused by repeating an identical command. Changed settings or
input file stamps require a different output directory. Keep the source dataset
available because exports use symlinks. An interrupted final export leaves a
`dataset.tmp` directory to inspect/remove before retrying.

Train the same selected images with original and refined poses to isolate the
effect of pose refinement (both commands use the existing launcher):

```bash
python scripts/train.py --data-root "$SCENE" --scenes . \
  --output-root /data/3dgs_outputs/keyframe_ablation --name coverage600_original \
  --keyframe-manifest "$KF/keyframes.json" --eval \
  --no-exposure-compensation --antialiasing --gpu 0

python scripts/train.py --data-root "$REFINED/dataset" --scenes . \
  --output-root /data/3dgs_outputs/keyframe_ablation --name coverage600_refined \
  --keyframe-manifest "$REFINED/dataset/keyframes.json" --eval \
  --no-exposure-compensation --antialiasing --gpu 1
```

Compare both on the **same reserved filenames**, including floor crops. A
separate all-training-frame baseline using the same `--eval` split tests frame
selection itself. For sparse LiDAR depth supervision, regenerate depth samples
using the refined dataset: old per-pixel depth files were projected with old
poses. This pipeline deliberately does not copy those cached depth priors.


### PSNR on all registered COLMAP frames

After training, the launcher evaluates the final saved PLY on every COLMAP-registered
frame. It loads one image at a time and prints progress. Results are written under
`<run>/psnr_all_frames/iteration_<N>/`: `per_frame.csv` contains one PSNR per
frame and whether it was selected for training, and `summary.json` contains
frame counts and mean per-image PSNR for all, selected, and unselected frames.
The source images use the run's saved resolution setting. Failure to read a frame
fails evaluation rather than silently changing the evaluated set.

All frames are rendered **without** per-image exposure transforms, including
training frames, so every image uses the same base-appearance protocol. This is
not the same metric as the five-view exposure-adjusted training PSNR. Antialiasing
is set from the launcher option; use `--no-antialiasing` for models trained without
it. Mean over all frames includes training images and thus is not held-out PSNR.
For a fair held-out comparison across runs, compare the same filenames excluded
from every model's training selection.

For a saved model (including a run already in progress), use the normal scene,
output, name, and GPU arguments with `--evaluate-only`; add `--eval-iteration N`
to select a saved PLY instead of the latest. Evaluation can use substantial GPU
memory and time. It overwrites prior PSNR files for the same saved iteration.
Use a free GPU and evaluate an iteration after the model has finished saving.
Pass `--no-eval-all-frames` to skip automatic evaluation for a new training run.


### Scaffold-GS

Select the anchor-based model with `--model-type scaffold` on `scripts/train.py`.
The existing model remains `--model-type 3dgs` (the default). Scaffold-GS uses a
separate model, renderer, and training loop. Keyframe selection, COLMAP data,
CPU image storage, antialiasing, per-training-image exposure optimization, and
all-frame PSNR are supported. The all-frame evaluator detects the model type from
`cfg_args` and evaluates the base appearance without per-image exposure.

Defaults follow the official Scaffold-GS model for 32-dimensional anchor features,
10 offsets and a voxel size of 0.001. Override them with `--scaffold-feat-dim`,
`--scaffold-n-offsets`, and `--scaffold-voxel-size`; voxel size is in COLMAP's
coordinate units. The per-camera appearance embedding is disabled so unseen views
have a well-defined appearance; optional exposure compensation is independent.
The implementation adapts the [official Scaffold-GS source](https://github.com/city-super/Scaffold-GS)
at commit `59c833b56bbbf510f3f64d40f81721995caced66`, under the 3DGS license.
It uses PyTorch's `scatter_reduce_` instead of `torch_scatter` and a frustum
prefilter compatible with this repository's rasterizer.

A saved Scaffold model consists of anchor `point_cloud.ply` and the three MLP
files in the same `point_cloud/iteration_N/` directory. This PLY is **not** an
ordinary 3DGS PLY and cannot be opened directly in SuperSplat. Keep the entire
iteration folder when moving a model. Live SIBR viewing, training resumption,
depth regularization, held-out `--eval` training, and sparse Adam are not yet
implemented for Scaffold-GS. The regular all-frame PSNR evaluation works.

Example smoke run before longer training:

```bash
python scripts/train.py --data-root /path/to/data --scenes my_scene \
  --model-type scaffold --name scaffold_smoke --gpu 0 --data-device cpu \
  --select-keyframes --keyframe-group-size 15 --iterations 100 --no-eval-all-frames
```


### Interactive Scaffold-GS viewer

Install the browser viewer dependency once in the existing `3dgs` environment:

```bash
python -m pip install 'viser==1.0.30'
```

Run `python scripts/view_scaffold.py --gpu 0 --port 8080` and open
`http://127.0.0.1:8080` in a browser on the **same machine**. The viewer binds
to localhost and the file picker browses the viewer machine's filesystem.
The file picker starts in your home directory; enter a different folder in the
GUI and click **Open folder / refresh**. On a remote headless machine, forward the port over SSH
from your computer (`ssh -L 8080:127.0.0.1:8080 USER@HOST`) and open the same
URL in your local browser. A GPU on the machine running Python does the rendering.

In the left panel, enter a folder path and click **Open folder / refresh**, or
select a folder from the list and click **Open selected** to navigate. Select
`point_cloud.ply` and click **Open selected** to load it. Alternatively paste
its full path and click **Load PLY path**. The PLY must be from a Scaffold run
in this repository and have `opacity_mlp.pt`, `cov_mlp.pt`, and `color_mlp.pt`
beside it; its model run directory must retain `cfg_args` and `cameras.json`.
The starting camera comes from that file. Drag to orbit and scroll to zoom; use
**Go to first training camera** to reset. Toggle AA to match training and adjust
render width to trade quality for responsiveness. The viewer uses the base
appearance without per-training-image exposure compensation. Rendering a saved
iteration is separate from training; select a different PLY to inspect another.



### Compare 3DGS and Scaffold-GS along one camera trajectory

`scripts/render_model_comparison.py` renders the same COLMAP-registered camera
sequence for both saved models, in natural filename order. It saves `original.mp4`,
`3dgs.mp4`, `scaffold.mp4`, `side_by_side.mp4`, and `trajectory.json`, including
all camera poses, frame order, rendering settings and a SHA-256 fingerprint.
The side-by-side video has three panels: original | 3DGS | Scaffold-GS.
Both runs must have been trained from the specified scene path. Rendering uses the
base scene appearance; per-image exposure transforms are not applied.

The script requires `ffmpeg` with the `libx264` encoder. Matching input images
are read from the scene's `images` folder one at a time; supply `--images` if
they are stored elsewhere. COLMAP camera intrinsics and poses supply the
trajectory. The default video
width is 1280 pixels and all source cameras must have the same aspect ratio.
Set the AA option for each model explicitly: saved 3DGS configs do not record
whether AA was enabled. Use an empty output directory for each comparison.
For a short check use `--max-frames 60`; omit it for the full trajectory. Output
fps defines playback speed; it is not derived from the original timestamps.

```bash
python scripts/render_model_comparison.py \
  --source /path/to/scene \
  --model-3dgs /path/to/3dgs_run \
  --model-scaffold /path/to/scaffold_run \
  --output-dir /path/to/comparison \
  --iteration 30000 --gpu 0 --3dgs-aa on --scaffold-aa off \
  --width 1280 --fps 30
```

`--start`, `--stride`, and `--max-frames` choose a subset from the natural-order
COLMAP trajectory. A stride larger than 1 omits frames (and therefore changes
playback speed unless fps is adjusted). The two models are loaded and rendered
sequentially on the same GPU. Default `--iteration 30000` selects the same
iteration for both; `-1` selects each model's latest, which may differ.



### Ablation videos and saved result tables

`scripts/render.py --video random` and `--video actual` accept any mixture of
native 3DGS/AbsGS, Scaffold, Textured Gaussians and Proxy-GS. Supply at least two
`--models` paths, or use `--model-root ROOT --runs NAME ...` to resolve nested
run folders. `--labels` and `--aa` accept one value per model. Proxy-GS and
Textured Gaussians require AA off; `auto` uses saved AA or assumes on for legacy
native configurations that lack the field. Set AA explicitly for old runs.

```bash
python scripts/collect_ablation_results.py --root /data/3dgs_outputs/orange_4F/1_full --list
python scripts/collect_ablation_results.py --root /data/3dgs_outputs/orange_4F/1_full --iteration 30000

python scripts/render.py --video random --source /path/to/scene \
  --model-root /path/to/results --runs baseline ablation \
  --labels "3DGS + AA" "Ablation" --aa on on --iteration 30000 \
  --duration 40 --seed 42 --deviation 3 --max-distance-fraction 0.10 \
  --view-degrees 12 --width 1280 --fps 30 --gpu 6 --output /path/to/random_video

python scripts/render.py --video actual --source /path/to/scene \
  --model-root /path/to/results --runs baseline ablation \
  --labels "3DGS + AA" "Ablation" --aa on on --iteration 30000 \
  --actual-poses model --width 1280 --fps 30 --gpu 6 --output /path/to/actual_video
```

Random videos use one shared trajectory and no GT panel. Identical source,
seed and path settings reproduce the same trajectory across ablations. These
camera-derived paths are not checked for collisions with walls. Actual videos
place GT first, then each model in the supplied order. They cover all registered
frames unless `--start`, `--stride` or `--max-frames` restrict the selection.
`--actual-poses source` (default) uses identical source calibration for every
model; `model` uses each run's calibration for the same GT image names. Use the
latter to inspect pose refinement, and disclose that poses differ. A refined
source must share the original LiDAR PLY and original images through symlinks
or hard links; unrelated coordinate systems/GT images are rejected.

The default iteration is the latest *common* checkpoint. Explicit per-model
`--iterations 30000 60000` allows a different-budget comparison; each panel
displays its iteration. Missing checkpoints/MLPs fail before rendering.
`--dry-run` validates the plan without CUDA or output writes. Outputs are
`model_0.mp4`, `model_1.mp4`, etc., `original.mp4` for actual views,
`side_by_side.mp4`, and `trajectory.json` with camera hashes, poses, settings,
model paths and completion state. Use a new output directory for each run.

Models render sequentially using process replacement, including environment
switches for Proxy-GS; no Python supervisor holds CUDA state. Proxy-GS computes
fresh mesh depth at every requested pose and output resolution, retaining proxy
culling for novel views. Existing training depth caches are left intact.

The CPU-only result collector reads saved all-frame PSNR summaries and reports
missing values as dashes. `valid` means unselected frames in that protocol.
Train/eval image-report scores appear in a separate table with frame counts;
they are never presented as all-registered-frame PSNR. `--runs NAME ...` filters
the tables, and `--iteration -1` selects each run's latest available metric
while retaining its actual step count.

To run both current Proxy-GS comparisons concurrently on GPUs **0–5**:

```bash
bash scripts/render_proxygs_reports.sh
```

GPUs 0/1 render all original frames and compute PSNR for `proxygs_kf600_original`
and `proxygs_stride3_original`. GPUs 2/3 render their random videos; GPUs 4/5
render their captured-trajectory videos with GT. Each Proxy-GS run is compared
with its matched 3DGS+AA control (`kf600_original` or
`sharp_keyframes_stride3_aa`). Every job gets one GPU. Rendering models within
a video still uses process replacement, without a Python supervisor.

All six jobs are validated before launching. Per-job logs and four videos are
written beneath `MODEL_ROOT/comparisons/proxygs_ITERATION_TIMESTAMP/`; each
video subdirectory contains `side_by_side.mp4`. All-frame galleries/PSNR stay
under their model directories. The launcher waits for all jobs and exits with
an error if any fail. `--dry-run` validates and prints commands without rendering.
Override `SCENE`, `MODEL_ROOT`, `OUTPUT_ROOT`, `ITERATION` (default 30000),
`NATIVE_PYTHON`, or `PROXY_PYTHON` through environment variables when needed.

### Standalone LiDAR occlusion: 3DGS + AA

The occlusion framework builds its own fixed surface patches directly from the
same `input.ply` used to initialize native 3DGS. It needs no Proxy-GS repository,
environment, model, mesh, depth files, or training stage. `--occlusion_mode lidar`
enables it in native `train.py`; native preparation is automatic.

Preparation retains a measured point per voxel, fits local PCA normals, and
rejects poorly planar or one-sided neighborhoods. Trusted patches have bounded
radii and become opaque local surfels. No global mesh completion or learned
opacity is used. Defaults are voxel size 0.05, maximum patch radius 0.1, and
occlusion margin 0.05, all in **scene units**. These are starting values, not
sensor-noise estimates. Missing/uncertain surface coverage remains unoccluded.
Check the reported trusted-patch and culling fractions on the real scene.

At each camera pose, a depth-only CUDA pass finds the first fixed surface.
A Gaussian is removed before the unchanged AA rasterizer only when its nearest
3-sigma extent is behind that surface throughout its full rasterizer tile
footprint. AA covariance support, boundary padding, and unknown pixels are
accounted for. Partially occluded splats stay. Culled rows have zero rendering
gradients and radii; their Adam parameter/moment updates and per-view
densification statistics are skipped. The standard initialization, SH appearance,
losses, position LR, densification schedule, and opacity resets stay unchanged.
Default Adam is required.

**Novel views use the same fixed 3D surfels from the new camera pose.** Exact-pose
cached depths are reused when available; otherwise depth is computed on demand.
No association with particular training frames is needed. Culling stays enabled
at inference. Maximizing foreground opacity is not a substitute for this depth
constraint: opacity saturation cannot guarantee spatial coverage. Rendering
without culling is an ablation, not an equivalent deployment mode. A generic PLY
viewer will not apply this mechanism; deployment needs the native renderer plus
`occlusion_geometry.npz`, which is saved alongside the checkpoint.

Only small 16x16 tile-depth maxima are cached, keyed by geometry content, pose,
FoV, resolution, clipping planes, and boundary padding. At most four tile maps
stay in GPU memory. Surface triangle expansion and Gaussian culling are chunked.
The only added CUDA dependency is the general-purpose nvdiffrast rasterizer;
the existing GS extension needs no rebuild. All model computation remains native
3DGS. Install it once in the native environment:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate 3dgs
cd /home/sequor_vfm/gaussian-splatting
python -m pip install --no-deps --no-build-isolation -r requirements-occlusion.txt
python scripts/smoke_occlusion.py --gpu 2
```

The smoke test checks surface depth, image orientation, an unseen camera pose,
AA rendering, and zero gradients for hidden splats on a real GPU. CPU tests also
check opposite-side camera visibility, point-cloud patch fitting, open holes,
cache identity, and frozen optimizer momentum. CPU tests do not validate CUDA.

One command runs the CUDA smoke test, prepares geometry, trains for 30k steps, evaluates **all original
registered frames first**, then saves GT/render/error galleries for both train
and held-out views. It also evaluates the new weights with culling disabled.
There are no Python supervisor subprocesses or environment switches:

```bash
python scripts/run_occlusion.py \
  --source /data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full \
  --model /data/3dgs_outputs/orange_4F/1_full/3dgs_stride3_aa_lidar_occlusion \
  --gpu 2
```

Without `--keyframes` or `--baseline`, the workflow selects the sharpest frame per
consecutive stride-3 group. A supplied baseline's manifest is reused by default.
To reuse the exact previous baseline selection and generate all
four train/render culling combinations plus **random-trajectory** and
**GT/captured-trajectory** comparison videos, supply the baseline:

```bash
ROOT=/data/3dgs_outputs/orange_4F/1_full
BASE=$(python -c 'from scripts.render import resolve_models; import sys; print(resolve_models(None, sys.argv[1], ["sharp_keyframes_stride3_aa"])[0])' "$ROOT")
python scripts/run_occlusion.py \
  --source /data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full \
  --model "$ROOT/3dgs_stride3_aa_lidar_occlusion" \
  --baseline "$BASE" --gpu 2
```

The default is AA enabled, exposure compensation disabled, and 30,000 steps.
`--voxel-size`, `--max-radius`, and `--margin` control the geometric prior.
`--report-max-views 64` limits only the saved image galleries; PSNR always uses
all registered frames (5,009 for this scene). Each original image is assigned to
the selected/unselected PSNR split using the saved keyframe manifest.
`--dry-run` checks inputs and prints the stages without writing files or using
CUDA. `--evaluate-only` retries post-training work and skips recorded completed
stages with identical commands. Incomplete videos retain their output; choose a
fresh `--video-output DIRECTORY` or inspect it before retrying that stage.

For a separate combined large-scene experiment on GPU 1, add the fixed-position
warmup and reduced xyz LR. This keeps 30k total steps and the same keyframes:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate 3dgs
cd /home/sequor_vfm/gaussian-splatting
git pull --ff-only origin experiment/my-3dgs-changes
ROOT=/data/3dgs_outputs/orange_4F/1_full
BASE=$(python -c 'from scripts.render import resolve_models; import sys; print(resolve_models(None, sys.argv[1], ["sharp_keyframes_stride3_aa"])[0])' "$ROOT")
python -u scripts/run_occlusion.py \
  --source /data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full \
  --model "$ROOT/3dgs_stride3_aa_lidar_occlusion_warm7k_lr50" \
  --baseline "$BASE" --iterations 30000 --resolution -1 --gpu 1 \
  --position-warmup-steps 7000 --position-ramp-steps 2000 \
  --position-lr-scale 0.02
```

Positions and their Adam state stay fixed through step 7,000; appearance,
opacity, scale, and rotation still train. The xyz LR then ramps linearly from
zero to `3.2e-6` at step 9,000 and decays exponentially to `3.2e-8` at step
30,000. Both endpoints are still multiplied by scene extent. The launcher
sets the decay duration to `iterations - warmup - ramp` (21,000 here), unless
`--position-lr-max-steps` explicitly overrides it.

Densification and opacity resets automatically shift by the warmup duration.
This preserves 144 densification events, at steps 7,600 through 21,900, and
moves opacity resets to 10k/13k/16k/19k. `--densification-phase-offset` can delay
them further, but cannot place them inside the fixed-position warmup. Neither
xyz warmup nor lower LR enables a depth regularization loss. Culling remains
active throughout training and rendering. This is a combined ablation; compare
against the occlusion-only run to measure the schedule's additional effect.

The resolved schedule is printed at launch, recorded in `occlusion_workflow.json`
and `cfg_args`, and identified in comparison-video labels. Default occlusion-only
settings remain unchanged. Concurrent runs use separate temporary report files
when evaluating their shared baseline.

Outputs under the model directory:

- `occlusion_geometry.npz`: fixed centers, normals, radii, source SHA-256, and build settings.
- `occlusion_depth_cache/`: reusable exact-camera depth tiles, including novel views.
- `occlusion_workflow.json`: current stage and completed stages.
- `psnr_all_frames/iteration_30000/summary.json`: saved culling-on all-frame scores.
- `psnr_all_frames/iteration_30000/occlusion_off/`: the same trained weights without culling.
- `view_comparisons/`: actual/render/error galleries for both rendering protocols.
- `comparisons/random/side_by_side.mp4`: baseline versus trained model on one shared novel path.
- `comparisons/actual/side_by_side.mp4`: GT plus both models on captured cameras.

For individual renders, `scripts/render.py` and `scripts/eval_all_frames.py`
automatically load saved settings; `--occlusion off` disables culling explicitly.
Use `--occlusion on --occlusion-geometry MODEL/occlusion_geometry.npz` to apply
the same geometry to a baseline checkpoint. Override metrics are saved separately
under `occlusion_on_HASH/`, preserving existing baseline scores. Optional
`scripts/prepare_occlusion.py --point-cloud INPUT.ply --output GEOMETRY.npz` and
`scripts/precompute_occlusion.py --source SCENE --geometry GEOMETRY.npz --cache CACHE`
expose preparation/caching independently; the normal workflow does not require
these extra commands.

For per-image 3D inspection of the actual native culling mask, use
`scripts/inspect_occlusion.py --model MODEL --gpu 5 --check-gradients`.
It displays complete active/culled Gaussian populations, reference depth and
camera geometry, ON/OFF renders, and an optional native image-gradient check.
Use `bash scripts/inspect_da3_occlusion.sh GPU MODEL` to prepare three training
views and compare the saved patches against LiDAR-calibrated DA3 depth on the
same checkpoint. The viewer shows reference coverage, missing-depth retention,
and a reference-only display. This diagnostic does not start training.
See [OCCLUSION_INSPECTOR.md](OCCLUSION_INSPECTOR.md) for setup and interpretation.

To stop the two earlier all-frame rendering jobs on their server, the scoped
helper below checks physical GPU indices, process ownership, script names, and
exact run names before sending SIGTERM through PID-safe handles. It leaves jobs
on other GPUs and unrelated processes alone. Without `--stop`, it only lists.

```bash
python scripts/stop_render_jobs.py --gpus 0 1 --stop
```

### AbsGS densification

`--absgs` enables the AbsGS homodirectional gradient for splitting large
Gaussians. The signed gradients continue to guide cloning of small Gaussians;
all other 3DGS settings, including the anti-aliasing filter, remain configurable.
The absolute gradient is accumulated **per pixel inside the backward CUDA
kernel**, before different contributions can cancel; taking the absolute
value after ordinary backpropagation is not equivalent.

The optional backend is a separate package adapted from Graphdeco rasterizer
commit `9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0` with the gradient
accumulation approach of [AbsGS](https://github.com/TY424/AbsGS). It retains
the Graphdeco rasterizer's non-commercial license, included in the backend
folder. Build once in the existing environment:

```bash
python -m pip install --no-build-isolation --no-deps ./submodules/diff-gaussian-rasterization-abs
CUDA_VISIBLE_DEVICES=0 python scripts/check_absgs.py
```

The build needs a CUDA toolkit, compiler, and GLM. It normally reuses
`submodules/diff-gaussian-rasterization/third_party/glm`; if the GLM headers
are elsewhere, set `ABSGS_GLM_PATH` to the folder containing `glm/glm.hpp`
before installation. The smoke test checks AA rendering and signed/absolute
gradient consistency against the installed standard rasterizer on a small
scene; run it before starting a long experiment.

The default `--absgs-split-threshold 0.0004` is the AbsGS repository's
default; the standard clone threshold stays at 0.0002. This run retains the
baseline's `percent_dense`, pruning, and opacity reset settings to isolate
the splitting change; it is not a full reproduction of every AbsGS
hyperparameter. The AbsGS backend uses the default optimizer, not sparse Adam.
Saved models remain standard 3DGS PLYs, and the usual all-frame PSNR evaluator
uses the normal 3DGS renderer.

### Monocular depth aligned to a registered point cloud

`scripts/preprocess_depth.py` predicts single-image relative inverse depth
with Depth Anything V2 Large for the exact images listed in a keyframe manifest.
It saves 16-bit PNGs and fits a per-image affine transform to inverse
camera-space z values of the point cloud. The default cloud is the scene's
COLMAP `points3D.ply`; pass `--point-cloud` to use another PLY that is
already registered in the exact same coordinate system. The script cannot
determine whether the default PLY actually came from LiDAR.

Each image is calibrated using projected points and a coarse nearest-point
z-buffer. This approximates visibility; it does not guarantee correct
occlusion, registration, synchronization, or static-scene assumptions. A
random point holdout and relative residual checks flag some unreliable frames,
but are not independent multi-view geometric validation. Training disables
the depth loss for those frames and masks nonpositive aligned values.
`alignment_report.json` lists the accepted frames and diagnostics. By default
the preparation exits with an error if fewer than 80% pass.

The first invocation downloads a pinned DA-v2 checkout and approximately 1.3 GB
of pretrained weights into `~/.cache/3dgs_depth`. DA-v2 Large weights have
their own non-commercial license. A working GPU PyTorch, torchvision, OpenCV,
and internet access are required. The output folder must be empty or contain an
interrupted preparation with the same inputs and settings. The command
does not modify the original images, point cloud, COLMAP reconstruction, or the
original `depth_params.json`. Passing the output directory to the training
launcher with `--depths` selects its own alignment file.

### Textured Gaussians

`--model-type textured` uses the public [Textured Gaussians
reimplementation](https://github.com/bchao1/textured_gaussians), not the
unreleased code used for the paper's experiments. It uses the upstream custom
2DGS textured CUDA rasterizer and MCMC strategy, with this repository's camera
and keyframe pipeline. Two training stages run sequentially: 2DGS pretraining
for `--textured-pretrain-iterations` steps, then texture refinement for
`--iterations` steps. Defaults are 30,000 steps each, up to 10,000 initial
and MCMC Gaussians, and 50 x 50 RGBA textures per Gaussian. RGB and alpha
textures are enabled. Initial points are sampled when the input exceeds the cap.

The texture stage disables relocation and growth but retains the upstream MCMC
position noise and trains the other Gaussian attributes. These settings do not
freeze Gaussian centers. It is a different optimization setup from Graphdeco
3DGS and is not a direct reproduction of the paper's internal implementation.
Graphdeco anti-aliasing, exposure compensation and depth regularization are not
supported; explicitly pass `--no-antialiasing --no-exposure-compensation`.
The live SIBR viewer and the three-panel original-image comparison renderer do
not support this model. The two-panel novel-view renderer accepts
`--model-textured` instead of `--model-scaffold`; it renders 3DGS on the
left and Textured Gaussians on the right using exactly the same synthetic
poses. Set `--3dgs-aa` for the 3DGS model; Graphdeco AA is unavailable for
the textured renderer. A texture-aware all-frame PSNR evaluator is run after
training and saves per-frame and aggregate results in the usual output folder.
The checkpoint `point_cloud/iteration_<refinement steps>/textured.pt` contains
both Gaussian attributes and textures; it is not a SuperSplat-compatible PLY.

Install the backend in the existing 3dgs environment (requires a compatible
CUDA toolkit, a C++ compiler, and the GLM headers):

```bash
git clone https://github.com/bchao1/textured_gaussians.git ~/textured_gaussians
git -C ~/textured_gaussians checkout b7ed080d4bb5436204ca7e94598d26ff54ead5ff
python -m pip install ninja jaxtyping rich
BUILD_NO_CUDA=1 python -m pip install --no-build-isolation --no-deps -e ~/textured_gaussians
```

The upstream package JIT-compiles CUDA kernels on first use. Its CUDA backend
uses a generic extension name shared with gsplat, so set a dedicated
`TORCH_EXTENSIONS_DIR` for this backend. The upstream source expects GLM
headers; install them in the compiler include path or provide its expected
`textured_gaussians/cuda/csrc/third_party/glm` checkout.

### Novel-view side-by-side video

`scripts/render_novel_comparison.py` synthesizes a smooth path without using
original image pixels. By default it picks a seeded random continuous segment of
the naturally sorted COLMAP cameras, interpolates a capture-guided path, and
adds smooth random translation and camera pan. Both models use the exact same
poses. `side_by_side.mp4` shows 3DGS on the left and the model selected with
`--model-scaffold`, `--model-textured`, or `--model-absgs` on the right;
individual videos and `trajectory.json` are also saved. AbsGS uses the standard
3DGS renderer: pass its training run with `--model-absgs` and set
`--absgs-aa on` to compare it with an anti-aliased 3DGS baseline.

Defaults are 20 seconds at 30 fps, up to 600 source cameras and a control point
every 24 images. Change `--seed` for a different reproducible path and increase
`--deviation` for larger excursions; `--max-distance-fraction` controls the
approximate camera-coverage radius as a fraction of the camera-position bounding
box diagonal. The actual deviation may be smaller because of these bounds.
`--view-degrees` controls additional random camera pan. Use `--path-mode smooth`
for the earlier central capture-guided path with a small lateral excursion
controlled by `--lateral-fraction`.

The camera-position bounding box and distance to registered cameras are only
coverage heuristics, not a map of free space. The path is **not collision
checked** and may cross walls or objects. Filename ordering must reflect the
capture sequence. The manifest records the random seed, constraints, actual
offsets, source segment, generated poses, AA flags, and model iterations. As
with the original-trajectory comparison, models use base appearance without
per-image exposure transforms. Set AA for each model explicitly and use an
empty output folder.

### Experimental LiDAR surface-aware AbsGS

`preprocess_surface.py` estimates PCA normals, planarity confidence and local
support radius in the exact vertex order of the input COLMAP `points3D.ply`.
It stores a hash of the ordered coordinates; training rejects a mismatched PLY.
Optionally set `--max-support` in scene units to reject large sparse patches.
Fit once and reuse the same patch directory for multiple runs. PCA can mistake
mixed or under-sampled neighborhoods for planes, so inspect confidence and
renderings. The kd-tree uses substantial CPU memory for large point clouds.

Pass the following native arguments after `--` in `scripts/train.py`:

- `--surface_patches PATH --surface_mode tangent`: trusted initial Gaussians
  are flattened and aligned, and AbsGS split offsets are projected into the
  LiDAR tangent plane. Existing clone behavior is colocated. Others keep the
  usual 3D behavior.
- `--surface_patches PATH --surface_mode constrained`: additionally use
  confidence-weighted losses on sampled visible Gaussians to penalize normal
  displacement, orientation mismatch, and excessive thickness.

Use `--absgs` in the launcher for either variant. The default threshold is
`--surface_confidence 0.6`; `--surface_thickness 0.2` controls initial normal
scale. The loss weights are `--surface_position_weight`,
`--surface_normal_weight`, and `--surface_thickness_weight` (each 0.01 by
default). Normal displacement is divided by local patch radius before
squaring; the weights are experimental. `--surface_loss_samples` defaults to
100000 visible Gaussians per step.

References are inherited during splitting and cloning, pruned with their
Gaussians, and saved in training checkpoints. Constraints are disabled beyond
an inherited patch's support; there is no periodic reassociation. The final
PLY remains a standard 3DGS model. `measure_lidar_distance.py` measures
sampled Gaussian-center distance to the nearest input point, in scene units;
this is not a point-to-surface distance and is sensitive to LiDAR density.

### Sparse LiDAR depth and position warmup

`scripts/preprocess_lidar_depth.py` projects the original registered input PLY
into selected COLMAP cameras and stores sparse camera-z samples in each training
image. A coarse cell z-buffer keeps frontmost projected samples, but is only an
approximation to visibility and can miss a foreground surface. It uses the
centered principal point and pixel convention of this repository's Graphdeco
renderer, which does not use COLMAP principal-point offsets. The sparse depth
resolution must match the launcher's `--resolution`.

Pass native `--lidar_depth_dir PATH --lidar_depth_weight 0.02` to supervise
sampled rendered inverse depth. The robust loss compares alpha-weighted rendered
inverse depth to 1 / LiDAR camera-z, normalized per valid sample. The renderer's
output is not normalized by accumulated opacity, so incomplete opacity also
affects this loss. `--lidar_depth_start` delays the loss until the requested
iteration; no depth is used if the weight is zero.

`--position_warmup_steps N` freezes xyz and its Adam moments through N;
`--position_ramp_steps M` linearly ramps the xyz learning rate after warmup.
The usual exponential schedule starts after this ramp. Other Gaussian
attributes still train during warmup. Set `--densification_phase_offset N` to
shift the ordinary densification and opacity-reset schedule by N iterations;
without these options the original training behavior is unchanged.
