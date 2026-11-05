# Training

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






