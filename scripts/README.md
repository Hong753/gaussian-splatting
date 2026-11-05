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

