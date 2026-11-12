# Training

For CUDA per-pixel surface clipping and the all-points/frustum/surface/culling
viewer, see [RAY_OCCLUSION.md](RAY_OCCLUSION.md).

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

For the current **full-cloud connected LiDAR mesh + per-pixel CUDA culling**, use
[RAY_OCCLUSION.md](RAY_OCCLUSION.md) and `bash scripts/inspect_ray_surface.sh`.
That path does not gate measured walls by DA3 confidence or sparse planar-patch
acceptance. The original patch/whole-Gaussian ablation is documented below.

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

### Soft surface opacity and position attraction

`scripts/run_soft_surface.py` is a separate native 3DGS + AA ablation. Both
training RGB and inference use ordinary alpha compositing, with hard culling
off. The full LiDAR scene mesh supplies fixed first-hit depths and local planes
for two training losses; it is not required to render the resulting PLY.

For each training view, current Gaussian centers are associated with trusted
local surface interiors. A second native CUDA pass uses white colors for
surface-associated Gaussians, black colors for the others, and black background.
All Gaussians remain in the transmittance product. Its image is
`A_surface(p) = sum_i T_i(p) alpha_i(p) q_i`, where `q_i` is the per-view surface
membership tag. The loss is the mean squared positive deficit below 0.98,
weighted by 0.05, over trusted pixels. Geometry and tags are detached in this
pass: it only trains opacities. Unassociated foreground floaters can lose opacity;
far background Gaussians are not tagged as foreground coverage. Membership is
per Gaussian, not an exact per-fragment surface-depth test.

The normal attraction loss applies only to centers within 0.10 scene units of
a supported first-hit surface, in both camera depth and local plane distance.
It uses Smooth L1 of signed plane distance divided by 0.10, with beta 0.2 and
weight 0.01. Tangential motion stays free. Opacity membership uses a narrower
0.05 band. Distances are scene units (meters if the LiDAR coordinates are metric).
Missing depths, holes, depth discontinuities, masked pixels and unsupported local
planes are skipped. Both associations are recomputed each view, including for
newly cloned/split Gaussians. Neither loss pulls distant rooms onto the front wall.

Defaults: 30k steps; xyz and its Adam state frozen through 7k; 2k linear LR ramp;
xyz LR endpoints 3.2e-6 to 3.2e-8 before the usual scene-extent multiplier (50x
below baseline). Ordinary densification runs from 7,600 through 21,900. Periodic
opacity resets are disabled to preserve the learned barrier. Each loss ramps
over 1k steps from its start (opacity at step 1, attraction after xyz warmup).
This soft version reduces rear gradients through learned opacity; it does not
guarantee zero gradients, freeze rear optimizer momentum, or fill pixels with
no Gaussian support. The chosen weights and bands are experimental defaults.

Run from a fresh output directory; the baseline provides the keyframe manifest
and comparison checkpoint, not the initialization weights:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate 3dgs
python -u scripts/run_soft_surface.py \
  --source "$SOURCE" --baseline "$BASE" --model "$MODEL" \
  --scene-geometry "$SCENE_JSON" --gpu 0
```

For the existing stride-3 floor study, `bash scripts/run_soft_surface.sh 0`
resolves the original AA baseline and uses a fresh `soft_surface_warm7k_lr50_v1`
output. It activates `3dgs` itself. Pass `--dry-run` to inspect stages, or
`--evaluate-only` to retry post-training outputs. Weight flags include
`--soft-surface-opacity-weight` and `--soft-surface-position-weight`; either can
be set to zero for a single-term ablation with a separate `--model` directory.

Before training, `scripts/smoke_soft_surface.py` runs actual CUDA mesh-depth,
opacity-gradient, normal-attraction and rear-gradient-attenuation checks. The
workflow then trains, evaluates PSNR over every registered frame, renders train
and held-out GT comparisons, and produces actual/novel trajectory comparison
videos when a baseline is supplied. Every evaluation uses culling off.

Raw per-view depths are saved in the existing mesh depth cache and reused;
at most four derived surface maps reside in the GPU LRU cache. No all-view
Gaussian membership array is saved or loaded. The auxiliary pass follows RGB
backward to limit peak graph memory. `soft_surface_training.jsonl` logs surface
coverage, composited opacity, both losses, normal residuals, position LR and
Gaussian counts before/after population changes every 100 steps. CUDA smoke
results are saved as `soft_surface_cuda_smoke.json`.

### Inspect large-scene densification before extending training

`inspect_densification.py` loads the saved native Gaussian model and its LiDAR
initialization. It reconstructs scene extent from the **selected training camera
centers**, and shows the actual `percent_dense * scene_extent` threshold. This
threshold compares a Gaussian's largest **standard deviation**, not its diameter
or projected radius. Existing saved optimization settings are used; legacy
configs without `percent_dense` explicitly report the upstream 0.01 assumption.

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate 3dgs
cd /home/sequor_vfm/gaussian-splatting
ROOT=/data/3dgs_outputs/orange_4F/1_full
MODEL="$ROOT/3dgs_stride3_aa_soft_surface_warm7k_lr50_v1"
python -u scripts/inspect_densification.py \
  --model "$MODEL" --iteration 30000 --image 06663.jpg \
  --scene "$ROOT/lidar_scene_surface_v1/scene.json" \
  --gpu 0 --port 8087 --probe-gradients
```

Open `http://HOST:8087`. The mesh is optional, initially hidden and visual only;
all diagnostic renders use native 3DGS + AA with hard culling **off**. LiDAR and
Gaussian layers have independent controls. Point centers show population and
size classes; selected 3-sigma ellipsoids show physical covariance extents.

| Color | Meaning |
|---|---|
| Orange | CUDA bounding radius above the diagnostic threshold, but world size falls in the clone branch |
| Magenta | Large CUDA radius and world size falls in the split branch |
| Teal | Smaller projected footprint |
| Gray | No native raster footprint for this image |

The initial screen-radius threshold is 20 pixels; changing it changes only the
diagnostic. Set a normalized ROI with left/top/right/bottom controls and press
**Analyze selected region**. Counts include every Gaussian whose native bounding
footprint overlaps the crop. The contribution panel is a separate native CUDA
pass: class tags replace color, while every Gaussian remains in ordinary alpha
compositing. Its orange intensity and reported mass fraction therefore measure
actual image contribution; many large points hidden behind a wall can have a
large count but negligible weight. These are not culling colors.

The 250k point display cap and 100 ellipsoid cap do not affect any statistics or
renders. `--max-points 0` displays every point; that can be expensive in a browser.
`--probe-gradients` additionally computes the current image's photometric screen
gradient. A PLY does **not** retain accumulated gradients or historical clone/split
events, so size membership alone never establishes that a clone actually occurred.
The viewer reports whether the densification window had already ended.

**Export this diagnostic and ZIP** saves GT, render, error, contribution and
footprint PNGs, `summary.json`, the 1,000 largest ROI footprints as CSV, and
`MODEL/densification_inspection/iteration_30000/densification_debug.zip`.
Use `--export-only` for the same outputs without starting Viser. The archive
contains the most recently exported view/ROI. Checkpoint and configuration hashes
identify exactly what was inspected. To compare the baseline, rerun with its
model path at the same iteration, image and ROI.

Future native training runs also write **actual** growth events to
`densification.jsonl`: clone/split parent counts, pruning/population changes,
world-size cutoff, and how many eligible parents exceeded a 20-pixel radius in
the accumulated window. Logging does not alter selection, splitting, cloning,
opacity resets, learning rates or pruning. Existing PLYs cannot reconstruct this
history. Large footprints are not inherently wrong on smooth regions; inspect
their contribution and residuals on texture-rich surfaces before changing policy.

### Prepared 100k soft-surface run with parallel evaluation

Inspect the above diagnostics first. When ready, the longer-run launcher is:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate 3dgs
cd /home/sequor_vfm/gaussian-splatting
bash scripts/run_soft_surface_long.sh 0
```

This starts fresh LiDAR initialization in
`3dgs_stride3_aa_soft_surface_warm7k_lr50_100k_v1`. It keeps the soft objective,
7k fixed-position warmup, 2k ramp, 50x reduced position LR endpoints, 21k decay
duration and original 7,600–21,900 densification window. Position LR reaches its
floor at 30k and remains there; steps 30k–100k are additional refinement.
It does not claim to fix a splitting bottleneck. Both training and inference use
ordinary compositing, and inference needs no occluding surface.

Two independent Python processes run on the same selected GPU. Training publishes
PLY and optimizer checkpoints **after the same optimizer step** at 10k, 20k, ...,
100k. A `ready.json` marker is written only after both files are closed and hashed;
the worker never treats a partial PLY as a job. No inline validation blocks training,
although saving still pauses it for serialization and disk I/O. All ten optimizer
checkpoints are retained and can be large. Native `train.py --start_checkpoint`
accepts the saved `chkpntN.pth` files; the wrapper intentionally refuses to overwrite
an existing experiment.

The single evaluator processes all published checkpoints in order, streaming all
**5,009 original frames** at the saved resolution. It verifies exact frame and
training-split membership and reports all/selected/unselected mean per-channel
PSNR. It also calls `scripts/render.py --split both --eval-set unselected` for
64 sampled training and 64 sampled non-training GT/render/error comparisons.
These sampled galleries are not used to calculate full-trajectory PSNR.

Outputs under the new model:

- `point_cloud/iteration_N/point_cloud.ply`, `ready.json`, and `chkpntN.pth`.
- `psnr_all_frames/iteration_N/{summary.json,per_frame.csv}`.
- `view_comparisons/iteration_N/index.html` and images.
- `periodic_evaluation/{history.csv,history.json,index.html}` for the learning curve.
- `periodic_evaluation/{training.json,worker.json,evaluator.log}` and `training.log`.
- `soft_surface_training.jsonl` additionally logs sampled RGB/soft gradient norms
  and ratios every 100 steps. Sampling is deterministic and does not perturb RNG.

Evaluation shares compute and VRAM with training. Its default PyTorch allocation
cap is 20% of the GPU, not a reservation. If an evaluation OOMs while training is
active, it releases temporary allocations and retries the same queued checkpoint;
it never launches overlapping evaluators or silently skips snapshots. It can fall
behind training. After training ends the shell waits for the queue to finish. A
remaining error is reported as incomplete, not a successful 100k evaluation.

Restart only the evaluator, skipping results with verified checkpoint/output hashes:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate 3dgs
python -u scripts/run_soft_surface_long.py watch --model "$MODEL"
# If training has finished and the evaluator's cap was too small:
python -u scripts/run_soft_surface_long.py watch --model "$MODEL" --eval-memory-fraction 0.8
```

Here `MODEL` must point to the new 100k experiment. Its plan fixes the GPU and frame
list. `SOFT_SURFACE_MODEL`, `SOFT_SURFACE_SOURCE`, `SOFT_SURFACE_BASELINE`,
`SOFT_SURFACE_SCENE` and `SOFT_SURFACE_OUTPUT_ROOT` override wrapper paths.
`SOFT_SURFACE_EVAL_MEMORY_FRACTION` and `SOFT_SURFACE_REPORT_MAX_VIEWS` set worker
limits. The launcher never assumes a nonexistent 100k baseline checkpoint.

### Fixed LiDAR centers with a local maximum scale (no growth or pruning)

This independent native **3DGS + AA** ablation starts from the stride-3 baseline's
original `input.ply`, not its learned Gaussians. All initialized centers remain
fixed for the entire run, with no XYZ Adam state. Colors/SH, opacity, scales and
rotations learn. Cloning, splitting, pruning, opacity resets, exposure compensation,
hard culling and soft-surface losses are disabled. No mesh or DA3 cache is needed.

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate 3dgs &&
cd /home/sequor_vfm/gaussian-splatting &&
git pull --ff-only &&
FIXED_LIDAR_MAX_SCALE_RATIO=2.0 bash scripts/run_fixed_lidar.sh 0
```

The default upper bound is `sigma_max(i) = 2 * sigma_init(i)` for **each principal
axis**, using the existing nearest-neighbor point-cloud initialization scale.
It does not grow with the global scene extent. This is local point spacing, not
LiDAR sensor range, and is an experimental setting rather than a claim about
XGRIDS. Scales may shrink. Rotations remain free; this ablation adds no normal
alignment or thickness objective. A Gaussian's 3-sigma radius is three times its
sigma, so the value is not an ellipsoid diameter or pixel-radius cutoff.

Scale bounds are enforced before the first render and after every Adam step.
Projected axes have their Adam moments cleared; unprojected axes retain theirs.
Caps are stored in optimizer checkpoints, and exported PLYs already contain the
bounded physical scales, so ordinary renderers need no special cap implementation.
The immutable XYZ checksum, point count and scale inequalities are checked at
step 1, every 1k and each saved checkpoint in `fixed_lidar_training.jsonl`.

Defaults: 100k training iterations, coherent PLY + optimizer checkpoints every
10k, plus one separate evaluator on the same GPU scoring **all 5,009 frames**.
Each checkpoint also gets GT/render/error galleries for 64 train and 64 unselected
frames. Evaluation can queue behind training and never silently skips checkpoints.
The shell waits for the queue after training. It runs the native CUDA AA smoke
test `scripts/smoke_fixed_lidar.py` before loading the full scene.

Default model directory:

```text
/data/3dgs_outputs/orange_4F/1_full/3dgs_stride3_aa_fixed_lidar_scale2_noprune_100k_v1
  psnr_all_frames/iteration_10000/{summary.json,per_frame.csv}
  view_comparisons/iteration_10000/index.html
  periodic_evaluation/{history.csv,history.json,index.html,evaluator.log}
  fixed_lidar_training.jsonl
```

For an additional absolute cap, set `FIXED_LIDAR_MAX_SCALE_WORLD` (default 0,
disabled). The bound becomes `min(ratio * sigma_init(i), world_cap)` in **scene
units**; interpret those as meters only when the cloud is metric. Use a new model
directory when changing settings, e.g.:

```bash
FIXED_LIDAR_MODEL=/data/3dgs_outputs/orange_4F/1_full/3dgs_fixed_lidar_scale2_cap005_v1 \
FIXED_LIDAR_MAX_SCALE_RATIO=2.0 FIXED_LIDAR_MAX_SCALE_WORLD=0.05 \
bash scripts/run_fixed_lidar.sh 0
```

The absolute cap above is an optional example, not the default. Other overrides:
`FIXED_LIDAR_BASELINE`, `FIXED_LIDAR_POINTS`, `FIXED_LIDAR_SOURCE`,
`FIXED_LIDAR_OUTPUT_ROOT`, `FIXED_LIDAR_ITERATIONS`,
`FIXED_LIDAR_CHECKPOINT_EVERY`, `FIXED_LIDAR_EVAL_MEMORY_FRACTION`, and
`FIXED_LIDAR_REPORT_MAX_VIEWS`. Native resolution handling is unchanged (`-1`,
which limits source images wider than 1600 pixels to 1600).

To restart only evaluation after a failure (activate `3dgs` first):

```bash
python -u scripts/run_fixed_lidar.py watch --model "$MODEL"
# Once training has finished, a larger evaluator cap can be used if needed:
python -u scripts/run_fixed_lidar.py watch --model "$MODEL" --eval-memory-fraction 0.8
```

The original cloud is dense but may have duplicate samples or sparsely sampled
surfaces. Inspect same-camera crops and scale statistics before choosing a tighter
ratio; bounds that are too small can leave gaps. Pruning is deliberately deferred.

### Second run: differentiable scale penalty, fixed LiDAR centers

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate 3dgs &&
cd /home/sequor_vfm/gaussian-splatting &&
git pull --ff-only &&
bash scripts/run_scale_regularized.sh 1
```

The default target for each principal sigma axis is
`tau_i = min(2 * sigma_init_i, 0.05)` in scene units. After clipping initialization
once to this target, training adds
`0.01 * mean(relu(log(sigma_ik / tau_i))**2)` over all Gaussians and all three axes.
There is **no post-step clamping**, scale reset or pruning in this run. The penalty
is zero below the target and differentiably penalizes excess above it. Targets are
independent of opacity, view and scene extent. **5 cm is a soft penalty target,
not a guaranteed maximum.** Positions remain fixed; appearance, opacity,
rotation and scales learn. No geometry/occlusion dependency is introduced.

Defaults: GPU 1, 100k steps, fresh model directory
`3dgs_stride3_aa_fixed_lidar_scalereg_100k_v1`. PLY and optimizer snapshots every
10k use the same independent all-5,009-frame evaluator and GT/render/error
galleries described above. `SCALE_REG_MODEL`, `SCALE_REG_RATIO`,
`SCALE_REG_TARGET_WORLD`, `SCALE_REG_WEIGHT` override the second-run settings;
the `FIXED_LIDAR_*` source/iteration/evaluator overrides also apply.

`scale_regularization_training.jsonl` records the actual weighted penalty and
its gradient norm relative to RGB on a deterministic sample of visible Gaussians
every 100 steps. `fixed_lidar_training.jsonl` records the largest sigma and fraction
of Gaussians exceeding their targets every 1k/checkpoint, along with immutable XYZ
and count checks. A zero penalty below target is expected; a persistently tiny
regularizer/RGB gradient ratio with oversized scales indicates the weight needs
adjustment. Optimizer checkpoints preserve original spacing targets and any
learned exceedances exactly.

The original GPU 0 preset uses only the relative `2 * sigma_init_i` cap unless
an absolute cap was explicitly provided. Comparing its default against this
GPU 1 preset changes both the constraint type and the absolute target. For a
matched-reference clamp-versus-penalty ablation, use `SCALE_REG_TARGET_WORLD=0`
against that original preset, or an independent hard-cap run with
`FIXED_LIDAR_MAX_SCALE_WORLD=0.05`.

### Third run: adaptive activation on discrete LiDAR coordinates, size pruning

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate 3dgs &&
cd /home/sequor_vfm/gaussian-splatting &&
git pull --ff-only &&
bash scripts/run_discrete_lidar.sh 2
```

This native **3DGS + AA** experiment uses only coordinates already present in
the complete original `input.ply`. XYZ gradients are disabled, XYZ LR is zero,
and no XYZ Adam moments accumulate. Every active Gaussian has one immutable
candidate ID; at most one Gaussian occupies each unique measured coordinate.
No interpolated surface points, tangent offsets, mesh, DA3 or hard/soft occlusion
losses are used. Ordinary PLY inference needs none of the preprocessing data.

The full unique cloud is the candidate pool. The default initialization activates
50% of it, preserving at least one original point per occupied 5 cm voxel. This
may activate more than 50% if needed for voxel coverage. **It changes the initial
population relative to the two full-cloud runs.** Starting with all candidates
active would leave no unused coordinates for growth. Exact positional duplicates
are represented once and their first input-row mapping is retained. To start
with the full unique cloud instead, set `DISCRETE_LIDAR_INITIAL_FRACTION=1` and
choose a different `DISCRETE_LIDAR_CACHE`; growth can then occur only after
pruning makes nearby coordinates available.

At every 100 steps after step 500, through the final iteration:

1. Use accumulated screen-space RGB gradients to select observed parents.
   Observed parents exceeding the scale threshold are prioritized for replacement.
2. Propose up to two unused measured neighbors per parent from its cached 16-NN
   list, within 10 cm. Resolve conflicts so each coordinate is activated once;
   at most 50,000 new Gaussians are activated per event.
3. Delete every Gaussian whose largest principal sigma exceeds **0.05 scene
   units** (5 cm for metric input), regardless of global scene extent. There is
   no opacity-based or image-radius pruning and no opacity reset.

Survivors retain their trained scales and Adam moments. Children inherit parent
appearance/rotation, start with opacity at most 0.1, and use isotropic sigma from
the candidate's full-cloud 3-NN RMS spacing, bounded to 0.025 scene units at
creation. The same bounded initialization is used for initially active points.
There is **no continuous scale clamp or scale penalty**; oversize is allowed
between pruning events. Pruned coordinates are unavailable for immediate
re-seeding in that same event but may be activated by a neighbor later. Population
edits occur after the current optimizer step, preserving that step's RGB update.
A 5 cm sigma threshold corresponds to a 15 cm 3-sigma radius, not a 5 cm diameter.

The neighborhood search is spatial only: it does not establish surface identity
or visibility across a thin wall. That is intentionally left for a later
surface-aware experiment. Size pruning may still delete useful surface coverage;
its quality needs comparison on the saved images, not just a smaller point count.

The launcher first runs `scripts/smoke_discrete_lidar.py` against the installed
native CUDA AA renderer, testing backward before and after activation/pruning,
fixed XYZ and checkpoint restoration. It then builds a reusable CPU/scipy kNN
cache. Cache files are written incrementally with a final completion marker and
content checksums; interrupted preprocessing can be rebuilt, and complete
matching caches are reused. Training loads the candidate XYZ/spacing/neighbor
table to GPU and performs candidate selection and population edits there.
For 10 million candidates and 16 neighbors the table alone is about 640 MB;
it is not repeated in every optimizer checkpoint. **Keep the cache for training
resume**, but rendered PLYs remain independent of it. This does not hold all
camera images or per-view renderings in VRAM.

Defaults: GPU 2, 100k steps, model
`3dgs_stride3_aa_discrete_lidar_prune5cm_100k_v1`, cache
`lidar_discrete_pool_v1` under the same output root. Every 10k, an independent
process on GPU 2 evaluates all 5,009 frames and creates the same train/unselected
GT-versus-render galleries. Evaluation queues sequentially if it cannot keep up;
the launcher waits for all saved checkpoints to be evaluated.

`discrete_lidar_initialization.json` records exact source/unique/initial counts.
`discrete_lidar_growth.jsonl` records additions, scale-pruned count, remaining
population and unused candidates at every event. Every 1k/checkpoint,
`discrete_lidar_training.jsonl` checks that **every center**, including children,
exactly equals its original candidate coordinate, IDs are unique, and XYZ stays
outside Adam. Failed coordinate checks stop training. The periodic report
includes growth/prune totals and flags intervals without any activations or with
major population loss.

Overrides: `DISCRETE_LIDAR_MODEL`, `DISCRETE_LIDAR_CACHE`,
`DISCRETE_LIDAR_BASELINE`, `DISCRETE_LIDAR_POINTS`, `DISCRETE_LIDAR_SOURCE`,
`DISCRETE_LIDAR_OUTPUT_ROOT`, `DISCRETE_LIDAR_PRUNE_SIGMA`,
`DISCRETE_LIDAR_INITIAL_FRACTION`, `DISCRETE_LIDAR_INIT_VOXEL`,
`DISCRETE_LIDAR_GROWTH_RADIUS`, `DISCRETE_LIDAR_MAX_NEW`,
`DISCRETE_LIDAR_ITERATIONS`, `DISCRETE_LIDAR_CHECKPOINT_EVERY`,
`DISCRETE_LIDAR_EVAL_MEMORY_FRACTION` and `DISCRETE_LIDAR_REPORT_MAX_VIEWS`.
Additional controls are listed by `python scripts/run_discrete_lidar.py prepare --help`.
An existing cache rejects changed preprocessing settings; select a new cache
directory when changing its input, fraction, voxel size, neighbor count or seed.

To restart only the third run's evaluation queue, activate `3dgs`, set `MODEL`
to its output directory, then use:

```bash
python -u scripts/run_discrete_lidar.py watch --model "$MODEL"
```

### GPU 3: local KNN scale control for dense, fixed LiDAR

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate 3dgs &&
cd /home/sequor_vfm/gaussian-splatting &&
git pull --ff-only &&
bash scripts/run_local_scale.sh 3
```

This preset keeps **every original LiDAR center fixed throughout**. It trains
native **3DGS + AA**, with no pruning, densification, opacity resets, mesh,
Proxy-GS, culling or surface loss. Color, opacity, rotation and scales learn.
It is a recommended experiment, not a claim of measured quality improvement.

Making a scene bigger should not enlarge a Gaussian on an unchanged wall. The
native clone/split decision uses `percent_dense * scene_extent`; here, there is
no density-control operation and **no scene extent in the scale rule**. For each
point, preprocessing computes the original three-neighbor RMS initialization
scale `s0_i` and distance `d10_i` to its tenth nearest *other* input point:

| Quantity | GPU 3 default |
| --- | --- |
| Hard maximum principal sigma `b_i` | `min(s0_i, d10_i / 3, 0.015)` |
| Preferred principal sigma `t_i` | `min(b_i, d10_i / 6)` |
| Differentiable loss | `0.01 * mean(relu(log(sigma_ik / t_i))**2)` over all points and axes |
| Initialization | Clip original KNN sigma to `b_i`; do not force it to `t_i` |
| After each Adam step | Project stored log-scales to `log(b_i)`; clear moments on clipped axes only |
| Positions/population | Exact original XYZ, constant row count; XYZ has no gradient or Adam state |

Distances are in scene units. For metric LiDAR, **0.015 means 1.5 cm sigma**, or
a **9 cm diameter at the 3-sigma contour**. It is not a 1.5 cm splat diameter.
Many points receive smaller limits from their local spacing. The guard prevents
sparse outliers from receiving meter-scale primitives. When rescaling the scene
coordinates, rescale this absolute guard too (or explicitly disable it with 0).
No geometric lower scale bound is imposed; axes can shrink below their targets.
All three principal axes are bounded, while anisotropy and orientation remain
learnable. The native PLY contains exactly the resulting learned log-scales;
exported inference needs no extra clamp, cache or occluder.

Why two thresholds: for similar nearby Gaussian sizes, radius `3*sigma` and
pairwise support intersections suggest `sigma ~= d10/6` for roughly ten
neighbors. Making that an unconditional maximum can leave holes at sparse or
irregularly sampled surfaces. The softer target permits some extra overlap when
RGB needs coverage, while `b_i` still prevents large primitives. **Neither the
soft target nor the hard ceiling guarantees at most ten overlapping ellipsoids
or ten contributing Gaussians per pixel.** Sizes vary, projected AA adds a pixel
filter, and Gaussians have nonzero tails. This is a density-based size control,
with measured overlap diagnostics, not an exact combinatorial overlap constraint.

The global guard is optional, but the default is deliberately tighter than the
earlier 5 cm sigma reference. This run also changes the relative bound to
1x initial spacing and adds the ten-neighbor rule; it is a combined experiment,
not a single-variable clamp-versus-regularization ablation. Fixed measured points
cannot represent arbitrary detail finer than their projected sampling spacing;
smaller scales alone do not fix pose errors or missing surface samples. Compare
same-camera textured crops and coverage as well as PSNR.

Defaults: GPU 3, **100k steps**, model
`3dgs_stride3_aa_fixed_lidar_local_knn10_100k_v1`, reusable cache
`lidar_local_scale_knn10_v1` under `/data/3dgs_outputs/orange_4F/1_full`.
Every **10k** steps, PLY/optimizer snapshots are atomically published. A separate
process on the **same GPU** evaluates **all 5,009 frames**, with selected and
unselected PSNR, and writes 64-view train and unselected GT/render/error galleries.
Evaluations queue sequentially; the launcher waits for all of them to finish.
The evaluator's 20% allocator limit does not reserve VRAM from training; OOMs are
retried by the existing checkpoint queue.

Outputs include:

- `psnr_all_frames/iteration_N/summary.json` and `per_frame.csv`.
- `view_comparisons/iteration_N/index.html`: actual images, renders and errors.
- `periodic_evaluation/index.html`: PSNR history, invariant checks and diagnostics.
- `local_scale_initialization.json`: exact sigma ceiling/target and d10 quantiles.
- `fixed_lidar_training.jsonl`, every 1k/checkpoint: all-point XYZ/count/scale checks,
  sampled 3-sigma bounding-sphere intersection counts, centers inside the actual
  rotated ellipsoid, fraction of sphere counts exceeding K and sigma quantiles.
- `scale_regularization_training.jsonl`, every 100 steps: weighted penalty,
  regularizer/RGB scale-gradient ratio, and sampled visible rasterizer **pixel
  radius** quantiles for the current training camera (not diameters).

The overlap audit samples 2,048 points, enumerates all possible neighbors under
the immutable hard ceilings, and tests the current scales/rotations on GPU.
Sphere intersection counts are an **upper bound** on actual anisotropic
ellipsoid intersections, not a count of rendered contributors. A two-million-pair
budget limits memory: if a full sample does not fit, the **whole sample is omitted
and explicitly reported**, never silently truncated. Omission can bias the
retained sample toward lower-density areas; increase `--audit-pair-budget` and
use a new cache if many samples are omitted. Neither point sampling nor omitted
audit rows changes the training population.

The launcher first runs `scripts/smoke_local_scale.py` on the installed native
CUDA AA renderer. It checks real forward/backward, penalty gradients, bounded
Adam updates, immutable XYZ, cached counts against all-pairs counts, checkpoint
restoration and PLY serialization. Only then does full preprocessing/training
start. CPU regression tests cannot substitute for this host CUDA smoke.

Full-cloud KNN preprocessing uses **CPU scipy/cKDTree once**, with chunked queries
and disk-backed output arrays. The tree temporarily holds the entire XYZ cloud
in RAM; camera images and per-view depths are not cached there. Complete caches
are reused after source/config/checksum validation. Training loads two float32
bound vectors (about **80 MB for 10 million points**) and the bounded sampled
audit pairs into GPU memory. Bound vectors are retained in optimizer checkpoints;
the audit cache is not duplicated in each checkpoint. Keep the cache for training
resume/audits. Regularization, projection and training-time diagnostics use
PyTorch CUDA operations; no additional custom CUDA extension is required.

Overrides: `LOCAL_SCALE_MODEL`, `LOCAL_SCALE_CACHE`, `LOCAL_SCALE_BASELINE`,
`LOCAL_SCALE_POINTS`, `LOCAL_SCALE_SOURCE`, `LOCAL_SCALE_OUTPUT_ROOT`,
`LOCAL_SCALE_MAX_SCALE_RATIO`, `LOCAL_SCALE_MAX_SCALE_WORLD`, `LOCAL_SCALE_WEIGHT`,
`LOCAL_SCALE_NEIGHBORS`, `LOCAL_SCALE_SUPPORT_SIGMA`, `LOCAL_SCALE_OVERLAP_FACTOR`,
`LOCAL_SCALE_AUDIT_SAMPLES`, `LOCAL_SCALE_AUDIT_PAIR_BUDGET`,
`LOCAL_SCALE_CACHE_WORKERS`, `LOCAL_SCALE_ITERATIONS`, `LOCAL_SCALE_CHECKPOINT_EVERY`,
`LOCAL_SCALE_REPORT_MAX_VIEWS` and `LOCAL_SCALE_EVAL_MEMORY_FRACTION`.
Changed point inputs or preprocessing settings require a new cache directory;
new experiments require a fresh model directory.

Image resolution stays at `-1` to match the existing runs (native auto-downscale
above 1600-pixel width). Set `LOCAL_SCALE_RESOLUTION=1` for original image
resolution, using a fresh output directory and matching evaluation resolution
when comparing to another experiment. A smaller world-space sigma does not
itself raise the image resolution.

Restart only the evaluation queue, after activating `3dgs`:

```bash
python -u scripts/run_local_scale.py watch \
  --model /data/3dgs_outputs/orange_4F/1_full/3dgs_stride3_aa_fixed_lidar_local_knn10_100k_v1
```

### Replace GPU 0's first clamp run with distance-weighted updates

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate 3dgs &&
cd /home/sequor_vfm/gaussian-splatting &&
git pull --ff-only &&
bash scripts/run_distance_weighted.sh 0
```

Hypothesis: far-camera observations may provide poorer texture sampling and
should have less influence than nearby observations of the same measured point.
This experiment uses Euclidean distance `d_i = ||XYZ_i - camera_center||` for the
current camera, in the same world coordinates as the original LiDAR:

`w_i = max(0.05, (2 / max(d_i, 2))**2)`

| Camera distance (metric LiDAR) | Gradient multiplier | Adam displacement multiplier |
| --- | --- | --- |
| Up to 2 m | 1 | 1 |
| 4 m | 0.25 | 0.25 |
| 8 m | 0.0625 | 0.0625 |
| At least about 8.94 m | 0.05 | 0.05 |

Both factors are intentional. Adam divides its first moment by the square root
of the second moment; multiplying every gradient by a constant positive factor
alone can largely cancel in this normalization. The implementation feeds `w*g`
into Adam's moments, then also multiplies the resulting parameter displacement
by `w`. It is a modified optimizer policy, not simply a weighted photometric
loss. The 5% floor leaves learning possible for surfaces only observed from far
away. The distance reference is in **scene units**; change it with the geometry
if the cloud is rescaled. This does not estimate camera calibration quality,
visibility or depth uncertainty; its quality benefit remains to be measured.

The factor applies to SH/color, opacity, scale and rotation updates. XYZ stays
exactly fixed and has no gradients or Adam state. It also scales residual Adam
motion on zero-current-gradient rows; it does not add a frustum mask or freeze
previous momentum. Forward rendering, alpha compositing, RGB loss values and
evaluation are unchanged. Inference needs only the usual exported PLY. There
is no new occlusion mechanism, mesh, DA3 dependency, pruning or densification.

The replacement **inherits the old run's actual clamp settings**, including any
optional absolute sigma limit, resolution, original input cloud, keyframes,
iterations and checkpoint schedule. It verifies that the training command changes
only the output destinations and new distance options. It does **not** substitute
the GPU 3 local-KNN preset or resume the old learned weights. By default the
source run is `3dgs_stride3_aa_fixed_lidar_scale2_noprune_100k_v1`, and the fresh
output is `3dgs_stride3_aa_fixed_lidar_scale2_distance_100k_v1`, under
`/data/3dgs_outputs/orange_4F/1_full`.

The shell command first validates/prepares the replacement, then sends SIGTERM
only to this user's recognized Python trainer/evaluator/render processes with
the exact **old model path**. It checks the old plan's GPU, uses PID handles and
revalidates process start times, then waits until matching processes exit.
Other models and users are excluded; no GPU-wide kill or SIGKILL is used. Old
checkpoints and evaluation outputs remain on disk. If a matching process remains
after 20 seconds, the command fails before starting training.

After the old run stops, a native CUDA AA smoke test checks forward/backward,
unit-weight Adam parity, actual update attenuation, fixed centers, scale limits
and checkpoint restoration. A failed preflight can be retried with the same
command if preparation is identical and no trained `cfg_args`/checkpoint exists.
Once training has begun, use a fresh output directory for a new run.

Training and all-frame evaluation run in separate processes on **GPU 0**. For
the default first run, that is **100k steps, checkpoints every 10k**, PSNR over
**all 5,009 images**, plus the existing selected/unselected GT/render/error
galleries. Evaluation is queued sequentially and the launcher waits for it.

`distance_updates_training.jsonl` logs sampled visible-point distances/weights
and per-parameter raw/weighted gradient norms every 100 steps. It also logs the
measured Adam displacement against that same weighted-moment Adam proposal
before its final distance multiplier; scale displacements are measured **before
the subsequent hard scale projection**. `fixed_lidar_training.jsonl` verifies
the immutable cloud, scale bounds and saved distance policy. The periodic
checkpoint diagnostics link includes these records. Optimizer checkpoints reject
a changed distance policy on restore, including attempts to silently switch
between weighted and unweighted training.

All weighting and optimizer operations use PyTorch CUDA tensors in bounded
chunks; no new CUDA extension or scene preprocessing is required. Gradients and
Adam buffers keep their standard shapes, and no full extra parameter snapshot is
allocated. The native CUDA smoke runs before loading the large training scene.

Overrides: `DISTANCE_UPDATE_PREVIOUS_MODEL`, `DISTANCE_UPDATE_MODEL`,
`DISTANCE_UPDATE_OUTPUT_ROOT`, `DISTANCE_UPDATE_REFERENCE`,
`DISTANCE_UPDATE_POWER`, `DISTANCE_UPDATE_MIN_WEIGHT`, `DISTANCE_UPDATE_CHUNK_SIZE`.
The old experiment must have its `periodic_evaluation/plan.json`; it supplies the
original settings and source fingerprints. To inspect the exact replacement
without stopping or launching anything, use:

```bash
python scripts/run_distance_weighted.py prepare \
  --previous-model "$OLD_MODEL" --model "$NEW_MODEL" --gpu 0 --dry-run
```

### Tune the successful fourth run: relax the cap, keep the smaller target

Start with **2x the hard ceiling and 1x the soft target**, using GPU 0 while
leaving GPU 3's original 1x run as the reference:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate 3dgs &&
cd /home/sequor_vfm/gaussian-splatting &&
git pull --ff-only &&
bash scripts/run_local_scale_tune.sh 0
```

This command prints the reference's available scale statistics, prepares a fresh
matched **30k-step** trial, and stops only the old first-run trainer/evaluator
(`3dgs_stride3_aa_fixed_lidar_scale2_noprune_100k_v1`) using the exact-run helper.
It then starts the local-scale experiment and a separate all-frame evaluator on
GPU 0. **It does not enable distance weighting.** The original GPU 3 model,
weights, cache and running processes are left untouched.

For the original local bounds `b_i` and preferred scale `t_i`, the new limits are:

`b_new = ceiling_multiplier * b_i`

`t_new = min(b_new, target_multiplier * t_i)`

| Trial | Hard ceiling | Preferred scale | Maximum sigma for metric input |
| --- | --- | --- | --- |
| Original fourth run | `b_i` | `t_i` | 1.5 cm |
| First tuning trial | `2 * b_i` | `t_i` | 3 cm |
| Optional next trial | `2.5 * b_i` | `t_i` | 3.75 cm |

The multiplier applies to **all local ceiling terms**, not just the absolute
guard. Increasing only `max_scale_world` could do nothing if the KNN term is
already tighter. Relaxing the ceiling does not multiply every learned Gaussian's
size: the soft penalty still favors the original smaller scales, while RGB can
grow individual axes when the coverage/detail tradeoff supports it. Initialization
also retains the original 1x ceiling, up to floating-point roundoff. XYZ remains
fixed, with no pruning, densification or opacity resets.

The reference's source, original LiDAR cloud, keyframes, resolution, regularizer
weight and other local-scale settings are inherited. Only the cap/target
multipliers, output/cache location and shorter training horizon change. Compare
the reference and trials at the **same 10k, 20k or 30k snapshot**, not the reference
at 100k against a trial at 30k. Fixed-position schedules mean there is no position
LR decay or density schedule being shortened by this tuning horizon.

Every 10k (inherited checkpoint interval), the existing worker computes PSNR
over **all 5,009 frames** and writes train/unselected GT/render/error galleries.
Use the unselected PSNR together with matched floor/texture crops to choose the
cap; a higher average PSNR alone does not establish sharper fine texture.

Original KNN distances and bounds are reused from the fourth run's cache. The
new cache is separate. Its **overlap candidate lists are rebuilt for the larger
ceilings**, since reusing the old candidate lists would undercount intersections.
This needs a temporary CPU spatial tree, but skips the full-cloud KNN queries.
Training-time regularization, projection and audit remain on CUDA. The native
CUDA smoke tests the selected multipliers before preprocessing/training.

To understand whether the ceiling is actually restrictive:

```bash
python scripts/tune_local_scale.py --model "$MODEL"
```

The new `scale_regularization_training.jsonl` fields measure, on a sampled set
of visible Gaussians with **nonzero RGB gradients**:

- `active_at_cap_fraction`: at least one principal axis touches its ceiling.
- `active_growth_at_cap_fraction`: an axis touches its ceiling and its RGB
  gradient requests growth (negative log-scale gradient).
- `active_net_growth_at_cap_fraction`: growth is still requested after the
  scale regularizer is added, a stronger indication that the hard cap intervenes.
- `active_above_target_fraction`: an axis exceeds its soft target.

These are directional RGB diagnostics, not a proof that larger scales improve
quality. Repeated growth pressure at the cap supports testing a larger ceiling.
If little active geometry reaches the cap, relaxing it further is unlikely to
address the bottleneck; examine `active_regularizer_to_rgb_scale_grad_ratio` and test the
preferred-scale multiplier separately. Existing runs started before these fields
were added only have global cap statistics. **Global at-cap fraction can include
untouched initialization points and must not be mistaken for growth pressure.**

After the 2x trial has finished, the 2.5x trial uses distinct output/cache paths:

```bash
LOCAL_SCALE_TUNE_CEILING=2.5 bash scripts/run_local_scale_tune.sh 0
```

Defaults: reference `3dgs_stride3_aa_fixed_lidar_local_knn10_100k_v1`; trial
`3dgs_stride3_aa_fixed_lidar_local_cap2_target1_30000_v1`. Both are under
`/data/3dgs_outputs/orange_4F/1_full`; each trial's derived cache is stored inside
its own `local_scale_cache` directory. Do not launch two tuning trials on GPU 0
at once. Overrides: `LOCAL_SCALE_TUNE_REFERENCE`, `LOCAL_SCALE_TUNE_STOP_MODEL`,
`LOCAL_SCALE_TUNE_MODEL`, `LOCAL_SCALE_TUNE_CEILING`, `LOCAL_SCALE_TUNE_TARGET`,
`LOCAL_SCALE_TUNE_ITERATIONS`, `LOCAL_SCALE_TUNE_OUTPUT_ROOT`.
`LOCAL_SCALE_TUNE_TARGET=1.5`, for example, is a separate relaxation of the soft
preference; keep it at 1 for the first cap-only test. Changing a multiplier requires
a fresh trial; optimizer checkpoints reject mismatched multipliers on restore.

### Gradual scale allowance and independent parameter learning rates

To test a gradual allowance together with more conservative shape/opacity rates:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate 3dgs &&
cd /home/sequor_vfm/gaussian-splatting &&
git pull --ff-only &&
LOCAL_SCALE_TUNE_SCHEDULE=joint bash scripts/run_local_scale_tune.sh 0
```

Use this in place of the static 2x command. The fresh output is
`3dgs_stride3_aa_fixed_lidar_local_cap2_target1_joint_30000_v1`. The original
GPU 3 experiment remains the reference. This wrapper still stops only the
explicit `LOCAL_SCALE_TUNE_STOP_MODEL` (the old GPU 0 clamping run by default).
If the static 2x trial has already been started on GPU 0 and should be replaced,
set `LOCAL_SCALE_TUNE_STOP_MODEL` to its full model path before this command.
Do not run both trainers on GPU 0 concurrently.

For each original local ceiling `b_i`, the allowed sigma becomes `m(t) * b_i`:

- Steps 0–7k: `m=1`.
- Steps 7k–20k: a half-cosine ramp from `m=1` to `m=2`.
- Steps 20k–30k: `m=2`.

`LOCAL_SCALE_TUNE_CEILING=2.5` changes the final allowance to 2.5x. The cap is
computed from immutable bounds every step, so there is no cumulative rounding
drift. Changing it does **not** scale the learned Gaussians or optimizer moments.
The smaller soft target remains fixed. RGB optimization must justify any growth.
The final-cap cache also bounds the overlap audit for every earlier step.

The joint preset proposes the following LRs. These are starting hypotheses to
evaluate, not measured optimal values. The scale LR acts on **log sigma**; no
global scene-extent multiplier is applied to these groups.

| Parameter | Previous constant LR | Steps 0–7k | At 30k |
| --- | --- | --- | --- |
| Scale | 0.005 | 0.001 | 0.0001 |
| Opacity logits | 0.025 | 0.01 | 0.001 |
| Rotation parameters | 0.001 | 0.0005 | 0.00005 |
| SH DC | 0.0025 | 0.0025 | 0.00025 |
| Higher-order SH | 0.000125 | 0.000125 | 0.0000125 |
| XYZ | 0 | 0 | 0 |

Each non-position group holds its initial LR through 7k, then smoothly interpolates
in log-LR space to its own final value at 30k. Exposure compensation is disabled.
All original LiDAR centers and the population remain fixed; there are no opacity
resets, pruning, densification, mesh culling, or distance-weighted updates.

For controlled comparisons, `LOCAL_SCALE_TUNE_SCHEDULE` accepts `off` (the previous
static cap and constant LRs), `cap` (cap ramp only, previous constant LRs), `lr`
(new LR schedules, static final cap), or `joint`. Each mode gets a distinct output
name. Compare at matching 10k/20k/30k snapshots and shared camera poses. Every
10k, the independent evaluator still processes all 5,009 frames and produces
train/unselected PSNR and actual/render/error comparisons. Changing both caps
and LRs together measures the combined recipe; the cap-only run isolates the
effect of the new learning-rate recipe.

Overrides for the schedule are `LOCAL_SCALE_TUNE_START_CEILING`,
`LOCAL_SCALE_TUNE_RAMP_START`, `LOCAL_SCALE_TUNE_RAMP_END`,
`LOCAL_SCALE_TUNE_LR_START`, and `LOCAL_SCALE_TUNE_LR_END`. Initial LRs use
`LOCAL_SCALE_TUNE_LR_SCALE`, `_OPACITY`, `_ROTATION`, `_DC`, and `_REST`;
append `_FINAL` for their separate final values. For example, after another
trial has finished:

```bash
LOCAL_SCALE_TUNE_SCHEDULE=joint \
LOCAL_SCALE_TUNE_LR_SCALE=0.0005 \
LOCAL_SCALE_TUNE_LR_SCALE_FINAL=0.00005 \
LOCAL_SCALE_TUNE_MODEL=/data/3dgs_outputs/orange_4F/1_full/local_ramp2_scale_lr_half_v1 \
bash scripts/run_local_scale_tune.sh 0
```

Use a fresh model name when changing the rates or timing. The saved plan records
the exact policy, and optimizer checkpoints include its current iteration and
bounds. Resume rejects a changed policy and continues the original cap/LR phase.
The native CUDA smoke exercises all phases on a tiny scene before full training.

`scale_regularization_training.jsonl` logs `current_ceiling_multiplier`,
`schedule_iteration`, and the actual `lr_xyz`, `lr_scaling`, `lr_opacity`,
`lr_rotation`, `lr_f_dc`, and `lr_f_rest` alongside active growth pressure and
projected sizes. `fixed_lidar_training.jsonl` reports the current absolute cap,
not just the eventual 2x cap. The inspection command above summarizes these
fields. Use held-out quality, floor/texture crops, and these diagnostics to choose
the next LR trial; reaching a cap alone is not evidence that an LR is too high.

### Alternating bounded camera-pose refinement

Start an additional fixed-LiDAR run with the joint scale/LR recipe above and
per-training-image camera corrections:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate 3dgs &&
cd /home/sequor_vfm/gaussian-splatting &&
git pull --ff-only &&
bash scripts/run_pose_refinement.sh 0
```

The GPU argument selects the physical device. This launcher requires that device
to be free before starting; it does not stop an existing experiment. Training
and the independent checkpoint evaluator then share that device. The fresh model
directory is
`/data/3dgs_outputs/orange_4F/1_full/3dgs_stride3_aa_fixed_lidar_local_cap2_target1_joint_poses_30000_v1`.
It reuses the original fourth run's input and KNN bounds, not its trained Gaussian
checkpoint. `LOCAL_SCALE_TUNE_MODEL` can name another fresh directory.

This is alternating block optimization, similar in spirit to EM. There is no
latent-variable expectation step:

| Gaussian steps | Camera update | Gaussian update |
| --- | --- | --- |
| 1–7,000 | Original poses, no update | Fixed XYZ; learn appearance, opacity, scale and rotation |
| 7,001–20,000 | One pose-only proposal for the sampled training view, with Gaussians detached | One ordinary update using that view's detached corrected pose |
| 20,001–30,000 | Freeze learned pose corrections | Continue Gaussian refinement |

Pose proposals are extra work; 30k still means 30k Gaussian optimizer steps.
LiDAR centers never move. There is no densification, pruning, opacity reset,
mesh culling, exposure optimization, or distance weighting in this run. Its scale
allowance ramps from 1x to 2x over 7k–20k, and the parameter LR schedule is the
joint preset above. Compare with the joint run without pose updates to isolate
the effect of pose refinement.

Each camera has an axis-angle rotation offset and a translation offset expressed
in the original camera axes. **Total displacement from the original camera** is
bounded by a 0.03 scene-unit translation norm (3 cm for metric LiDAR) and a 0.5°
rotation norm. These are vector-norm bounds, not independent per-axis allowances.
The fixed world-space LiDAR anchors the reconstruction. Intrinsics are unchanged.

The pose-only loss is a masked smooth absolute RGB residual plus a zero-offset
prior of `0.001 * (||translation/0.03||² + ||rotation/radians(0.5)||²)`.
Translation/rotation LRs start at 0.001 / 0.0002 and decay to 0.1x during the pose
window. Pose images are resized to at most 960 pixels wide; Gaussian training
retains the configured full resolution. Adam state is separate per view. A
proposal must have a finite, nonworsening photometric-plus-prior objective with
the Gaussians fixed; otherwise both the offsets and their Adam state are restored.
Projection onto either offset bound clears that vector's momentum.

The native rasterizer does not differentiate its camera-settings object. During
the pose-only pass, the implementation instead differentiably transforms the
frozen Gaussian means and orientations into the camera frame and evaluates SH
colors using the corresponding world-space directions. The launcher first runs
`smoke_pose_refinement.py`: native-render parity, finite differences for all six
pose coordinates, frozen Gaussian parameters, bounded accepted updates, and
cache/resume checks on CUDA. A failed check stops the experiment before training.

The pose-only pass applies AA opacity compensation explicitly, with chunked
recomputation for its derivative. This avoids an incorrect derivative in the
pinned rasterizer, which evaluates the compensation derivative using covariance
entries that already contain the 0.3 pixel blur variance. Its forward factor is
unchanged; ordinary Gaussian training continues to use the installed native
kernel. No extension rebuild is required. Full-image forward parity is checked
on the original 25-splat fixture. Six-coordinate gradient checks use a separate
anisotropic splat, nonidentity camera and nonconstant SH, in both AA modes,
against a float64 matrix-covariance reference and two finite-difference steps.
The fixed probe pixels must remain away from alpha cutoff, opacity cap, tile and
transmittance branches for every perturbation. Arbitrary full-image differences
can jump across those hard branches; the gradient tolerances are not relaxed.

If this preflight fails before Gaussian training starts, pulling a fix and
rerunning the same launcher reuses the prepared directory and resets its failed
status before starting the evaluator. A failed directory that already contains
training artifacts is not silently restarted from initialization.

Original COLMAP poses, intrinsics, images and LiDAR input are never rewritten.
The source calibration is fingerprinted. With an explicit input PLY, loading also
skips point-cloud conversion inside the source directory. Each saved snapshot has
`point_cloud/iteration_N/poses.json`, containing original and corrected world-to-
camera matrices, offsets, bounds and proposal counts. Optimizer checkpoints
include the pose optimizer states. Resume rejects a different policy or source
calibration. The ready marker fingerprints the exact iteration's pose cache.

Every 10k, the separate evaluator processes **all 5,009 frames**, with two explicit
protocols:

- `psnr_all_frames/iteration_N/`: original poses for every frame; primary PSNR for
  comparison against earlier runs.
- `psnr_all_frames/iteration_N/poses_refined/`: corrected poses for training
  keyframes, original poses for every unselected image. Held-out poses are never
  optimized or interpolated; their PSNR must agree between these protocols.

Both protocols include sampled actual/render/error galleries under the matching
`view_comparisons/iteration_N/` directories. Full per-frame PSNR is separate from
the sampled galleries. `periodic_evaluation/index.html` links both and keeps their
scores separate. Better corrected training PSNR alone is not held-out pose
accuracy. Rendering defaults to original poses; explicitly use the cached poses:

```bash
python -u scripts/render.py \
  --model "$MODEL" --iteration 30000 \
  --split both --eval-set unselected --antialiasing \
  --camera-poses refined --gpu 0 --in-process
```

`pose_refinement_training.jsonl` logs each attempted view, accepted/rejected status,
objective before/after, translation/rotation gradient norms, current LRs and total
offset norms. Inspect accepted updates and these norms as well as matched image
crops. Defaults can be changed for a fresh trial with `POSE_REFINE_START`,
`POSE_REFINE_END`, `POSE_REFINE_EVERY`, `POSE_REFINE_MAX_TRANSLATION`,
`POSE_REFINE_MAX_ROTATION_DEGREES`, `POSE_REFINE_TRANSLATION_LR`,
`POSE_REFINE_ROTATION_LR`, `POSE_REFINE_PRIOR_WEIGHT`, and `POSE_REFINE_MAX_WIDTH`.
The existing `LOCAL_SCALE_TUNE_*` schedule controls remain available.

### GPU 2: larger scales and bounded LiDAR offsets with alternating poses

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate 3dgs &&
cd /home/sequor_vfm/gaussian-splatting &&
git pull --ff-only &&
bash scripts/run_pose_lidar_offsets.sh 2
```

This fresh experiment uses the same input points and stride-3 keyframes as the
fourth run, with the camera-pose alternation above. It requires GPU 2 to be free;
the training process and independent all-frame evaluator then share GPU 2.
The default output is
`/data/3dgs_outputs/orange_4F/1_full/3dgs_stride3_aa_lidar_offsets_cap5_poses_30000_v1`.

- **Scale allowance:** ramp the hard ceiling from 1x to **5x** over 7k–20k,
  doubling the previous bounded-offset experiment's 2.5x final allowance.
  The original per-point KNN ceiling still applies, multiplied by 5; the
  absolute maximum principal-axis sigma is `0.015 * 5 = 0.075` scene units.
  This is 7.5 cm if the LiDAR coordinates are meters, not the splat diameter.
  Initialization and the 1x soft scale target are unchanged. Scales grow only
  through optimization; increasing a ceiling does not enlarge them directly.
- **Position allowance:** for each original input row `p_i`, keep an immutable
  anchor and a fixed radius `r_i = min(0.005, 0.25 * sigma_init_i)`, using the
  native initial KNN RMS sigma before scale clamping. Enforce the Euclidean
  bound `||xyz_i - p_i|| <= r_i` after every Gaussian update and before export.
  The limit is total displacement from the original row, never per-step drift
  or independent per-axis clipping. There is no reassignment to another point.
- **Position schedule:** XYZ stays frozen through 7k, with no XYZ Adam moments.
  A cosine ramp raises its LR from zero to `1e-5` by 9k; exponential decay then
  lowers it to `1e-6` at 30k. These are absolute scene-unit learning rates;
  **there is no scene-extent multiplier**. A weak anchor prior adds
  `0.001 * mean_i(||(xyz_i - p_i)/r_i||^2)` to Gaussian updates. It helps resist
  joint drift of camera poses and the point cloud.
- **Alternation:** during 7k–20k, a pose-only step detaches every Gaussian
  parameter, including its bounded XYZ; the following Gaussian step detaches
  the corrected camera. Poses freeze after 20k. Scale, appearance, opacity and
  rotation keep the existing joint LR schedule. Population stays constant:
  no cloning, splitting, pruning or opacity reset.

Projection clears XYZ Adam moments only for rows that exceeded their radius;
other rows retain their state. Checkpoints save original anchors, radii, their
fingerprints, current XYZ, the offset policy/iteration, Gaussian Adam state and
the separate camera-pose state. Resume rejects changed policies or anchors.
The ordinary exported PLY contains the learned bounded positions and current
clamped scales; inference needs no anchor constraints or occluding surfaces.
Original COLMAP files and input LiDAR remain untouched.

The overlap audit expands its precomputed candidate search by twice the maximum
offset, so it can account for two centers moving toward each other. Counts use
current positions; they are not frozen at initialization. The original KNN
distances remain reusable, while the enlarged candidate lists are cached in
this experiment's own directory.

The launcher runs the existing pose/scale CUDA checks and
`scripts/smoke_lidar_offsets.py` before full training. The latter exercises native
AA with both optimization blocks, frozen-position warmup, finite XYZ gradients,
per-point projection, selective momentum clearing, checkpoint resume, complete
sampled overlap counts and PLY export. CPU regression tests validate Adam and
checkpoint invariants but do not replace these native CUDA checks.

Every 10k steps, the parallel evaluator computes PSNR over **all 5,009 frames**
and writes sampled GT/render/error galleries, with the same original-pose and
cached-refined-pose protocols described above. `fixed_lidar_training.jsonl` now
records `positions_within_bounds`, `anchors_unchanged`, `offset_max`,
`offset_mean`, `offset_near_bound_fraction` and `offset_xyz_lr`; actual
`positions_unchanged` becomes false after positions learn. The offset prior loss
is logged alongside the scale penalty. A nonzero offset is not an audit failure.

Optional overrides for a fresh output directory: `LIDAR_OFFSET_MAX_WORLD`,
`LIDAR_OFFSET_SIGMA_RATIO`, `LIDAR_OFFSET_WARMUP`, `LIDAR_OFFSET_RAMP`,
`LIDAR_OFFSET_LR_INIT`, `LIDAR_OFFSET_LR_FINAL`, `LIDAR_OFFSET_PRIOR_WEIGHT`.
Scale and pose overrides remain the `LOCAL_SCALE_TUNE_*` / `POSE_REFINE_*`
variables above. Changing settings requires a new `LOCAL_SCALE_TUNE_MODEL`.

### GPU 3: cap5 + bounded LiDAR offsets + poses + CUDA mesh culling

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate 3dgs &&
cd /home/sequor_vfm/gaussian-splatting &&
git pull --ff-only &&
bash scripts/run_pose_lidar_culling.sh 3
```

This is a fresh, **30k-step** counterpart to GPU 2's cap5/offset/pose run. It
keeps the same scale/LR schedules, original anchors and displacement limits.
There is **no pruning, densification, or opacity reset**: visibility does not
delete Gaussians. It requires a free GPU 3 and does not stop other jobs. Training
and the independent all-frame evaluation process then share GPU 3.

The measured full-scene mesh is read from
`/data/3dgs_outputs/orange_4F/1_full/lidar_scene_surface_v1/scene.json`.
Override with `LIDAR_CULLING_SCENE`. The mesh must match the original LiDAR,
source cameras and training keyframes. Predicted supplementary meshes are not
used by this experiment. Mesh and input checksums are recorded before launch;
optimizer resumes also validate the culling policy and geometry fingerprints.

For each **current corrected camera**, nvdiffrast rasterizes first-hit mesh
depth on CUDA. The Gaussian CUDA compositor rejects a fragment when its
Gaussian center's camera-z exceeds the pixel cutoff:

`cutoff = conservative_mesh_depth + min(0.15, 0.05 + 0.005 * conservative_mesh_depth)`

The constants are scene units: with metric LiDAR this is a 5 cm + 0.5% depth
band, capped at 15 cm. It accommodates small registration/surface errors while
blocking distant rooms. A two-pixel maximum-depth neighborhood relaxes culling
near boundaries; unknown pixels stay infinite. Doorways and mesh holes are not
filled. This is a per-pixel center-depth test, matching the compositor's depth
ordering, not an ellipsoid/ray intersection or an exemption for large splats.
Partially visible splats retain their unblocked contributions.

During a pose-only step, all Gaussian parameters are detached. Mesh depth is
recomputed for the proposed corrected camera and detached from autograd; pose
gradients are piecewise gradients of the splat renderer. Pose fitting uses mesh
interiors (valid neighboring depths spanning no more than the tolerance), with
the pixel mask held fixed for the proposal and acceptance check. Views with
fewer than 256 usable pixels skip the pose update. Gaussian optimization still
uses the full RGB image, including pixels with unknown mesh depth.

Gaussian rows with **no accepted rendering contribution** do not receive Adam
updates or moment decay for that view. This includes fully blocked and
out-of-view rows. Partial contributors update from their surviving pixels.
Adam's parameter-group scalar step counter still advances. The local scale and
anchor priors remain enabled, but do not move these frozen rows. Increased scale
ceilings do not directly enlarge hidden Gaussians. This policy prevents stale
momentum from moving a hidden point while preserving the original population.

Depth caching keeps at most **two frames** on the device, plus the fixed mesh.
Pose proposals and training steps during pose refinement are memory-only.
Warmup and final frozen-pose depths use one replaceable disk entry per image and
resolution, validated against the full camera signature. This does not load all
view depths into RAM/VRAM or accumulate thousands of obsolete pose versions.

The launcher checks native ray clipping/backward, corrected-pose depth and
six-coordinate pose gradients, zero blocked-background RGB gradients, frozen
Adam rows, and a complete pose/offset/scale update and checkpoint-resume cycle.
All use the installed CUDA extensions before the full scene loads. Install the
existing dependencies if absent; no new CUDA kernel is introduced:

```bash
python -m pip install --no-build-isolation -r requirements-occlusion.txt &&
python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization-occlusion
```

Output:
`/data/3dgs_outputs/orange_4F/1_full/3dgs_stride3_aa_lidar_offsets_cap5_poses_culling_30000_v1`.
Every 10k steps the independent evaluator computes **all 5,009-frame PSNR with
culling OFF**, separately for original and cached training poses. It also
renders matched train/held-out samples with culling ON and assembles
**GT | culling OFF | culling ON**, including error maps and per-view PSNR, under
`view_comparisons/iteration_STEP/culling_comparison/`. The periodic dashboard
links these previews. They diagnose dependence on the external mesh; training
with culling does not guarantee opaque foreground coverage at mesh-free inference.

`pose_culling_training.jsonl` logs every 100 steps: mesh cutoff coverage,
pose-interior coverage, image-footprint candidates, fully/partially blocked and
contributing Gaussians, and the depth signature. `blocked/candidates` uses the
number of Gaussians whose projected bounding footprints intersect the image,
not the total scene population and not the fraction of pixels blocked.
