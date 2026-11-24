# Indoor triangle reconstruction reference experiment

This worktree adds a CPU reference implementation and a locked synthetic study. For an observed ScanNet++ comparison, run the stock references first and then the paired experiment with the same prepared input set. The CPU implementation targets small diagnostic reconstructions and uses an all-faces visibility pass.

## Reproduce the synthetic evidence

Run these commands from the fork root with Python 3.12, NumPy, SciPy and Pillow. LPIPS separately requires PyTorch, torchvision and `lpips==0.1.4`, plus the standard pretrained AlexNet weights. That environment is used only for evaluation.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
OUT="$(realpath -m ../triangle_evidence)"
mkdir -p "$OUT"
python -m scripts.indoor_triangles make-synthetic --out "$OUT/data"
python -m scripts.indoor_triangles check --out "$OUT/sanity.json"

python -u -m scripts.indoor_triangles fit --data "$OUT/data" --out "$OUT/sparse_seed17" --seed 17
python -u -m scripts.indoor_triangles fit --data "$OUT/data" --out "$OUT/sparse_seed29" --seed 29 --minimal
python -u -m scripts.indoor_triangles fit --data "$OUT/data" --out "$OUT/dense_seed17" --seed 17 --dense

for run in sparse_seed17 sparse_seed29 dense_seed17; do
  python -u -m scripts.indoor_triangles evaluate --data "$OUT/data" --run "$OUT/$run" --out "$OUT/${run}_eval"
  python -m scripts.indoor_triangles lpips --eval "$OUT/${run}_eval"
done
python -m scripts.indoor_triangles fit-diagnostics --checkpoint "$OUT/sparse_seed17/shared_start.npz" \
  --fit "$OUT/data/fit.npz" --out "$OUT/fit_image_coverage.json"
python -m scripts.indoor_triangles render --checkpoint "$OUT/sparse_seed17/connected_hard.npz" \
  --views "$OUT/data/final.npz" --out "$OUT/hard_render"
python -m scripts.indoor_triangles export --checkpoint "$OUT/sparse_seed17/connected_hard.npz" \
  --threshold 0.90 --out "$OUT/hard_export"
```

The fitter reads only fitting and edit-validation RGB and camera arrays. The room generator keeps the procedural triangles and material IDs in a separate evaluator file. The synthetic rig uses fixed identity world-to-camera rotations with x right, y down, z forward, and normalized rays with metric range. Final views are 0.27 m and 0.28 m from the closest fitting center. Atlas UVs are assigned from a fitting-camera projection. Each trainable texel stores three FP64 channels in this reference implementation.

The sparse initialization uses rectified, multiview patch correspondences at corner tracks. The matched denser initialization also selects reliable stereo seeds at non-corner pixels and uses closer spatial sampling. The two initialization cohorts have different geometry sizes. The full candidate uses image-space Delaunay with gates on screen edge length, world edge length, and triangle aspect, then welds initially private faces through known fitting-track indices. This differs from restricted Delaunay in 3D and reconstructs only one reference chart. Both controlled arms start from an identical checkpoint and reset Adam identically. They use the same fixed fitting-track support policy, texture, hyperparameters, randomized pixels and trial budget of zero. The global arm hardens all nontransmitting faces at that same branch point.

Range comes from per-sample ray–plane intersections. Welded edges get deterministic exact ownership, and committed opacity is always one on the interior. Each sample composites positive events in range order. The CPU implementation evaluates the surrogate geometry derivative by paired finite-displacement probes with shared subpixel jitter. It is sensitive to sample count and does not implement an exact visibility-boundary gradient, GPU rasterization, mipmapping, or scan-scale topology. The appearance gradient is analytic. Warps gate low variance, saturation, view translation, support at five reference and target patch locations and same-face visibility before computing robust ZNCC. Track free-space uses actual transmittance weights, and missing anchor assignments pay the capped penalty. Bending uses a frozen initial per-phase dihedral gate, with zero coupling beyond 28 degrees. Stage transitions reset Adam. The Gaussian color control has fixed chart-lattice RBF locations and widths, equal trainable color bytes and fixed hard-hit geometry. The analytic pane is supported by the compositor. The fitted arms lack a translucency classifier.

## Elastic fitting with track-supported occlusion

`fit --hybrid` uses the elastic kernel on exposed edges while fitting provisional triangles. Shared-edge ownership stays exact. Candidate commitments still need the fitting-track support and local edit-validation checks. Committed triangles use unit-opacity first hits, and rejected or unsupported faces retain soft coverage. A final fitting-track check can reopen faces whose adjusted geometry loses support. Clipped faces use the existing pixel-distance kernel on their clipped polygon.

The elastic weight follows Eq. (5) of [Elastic Triangle Splatting](https://arxiv.org/abs/2608.29106). It is `exp(-sigma * x ** (2 / sigma))`, where `x = max(0, 1 - 3 * min_barycentric)` on the exposed edges. The prototype uses a shared geometric anneal from 0.45 to 0.055 instead of optimizing softness. Override these values in `elastic_sigma_start` and `elastic_sigma_end` in the JSON config. Local edit trials and appearance-only passes keep softness fixed. Other fitting modes keep their smoothstep default and also accept `--kernel elastic`.

```bash
python -u -m scripts.indoor_triangles fit --data "$OUT/data" \
  --out "$OUT/elastic_hybrid" --seed 17 --dense --hybrid
python -u -m scripts.indoor_triangles evaluate --data "$OUT/data" \
  --run "$OUT/elastic_hybrid" --out "$OUT/elastic_hybrid_eval"

python -u -m scripts.indoor_triangles fit --data "$OUT/data" \
  --out "$OUT/kernel_pair" --seed 17 --dense --compare-kernels
python -u -m scripts.indoor_triangles evaluate --data "$OUT/data" \
  --run "$OUT/kernel_pair" --out "$OUT/kernel_pair_eval" --first-hit
```

The paired run fits the same connected mesh, atlas and camera rays with both kernels. In this comparison `--first-hit` gives both fits the same opaque renderer for RGB, leaving transmitting faces soft. Mesh scoring and export keep the checkpoints' metadata. For room data replace `--dense` with the prepared `--tracks` file, and supply the registered reference via `evaluate --reference-mesh`.

On the synthetic room, denser stereo initialization and 125 fitting steps gave the following all-face F-scores at 5 cm and held-out RGB results. The paired runs use the common first-hit renderer. The end-to-end runs render each checkpoint with its fitted visibility.

| Comparison | F-score, smoothstep → elastic | PSNR, smoothstep → elastic | LPIPS, smoothstep → elastic |
|---|---:|---:|---:|
| Same initializer, seed 17 | 39.58% → 38.61% | 9.47 → 9.48 | 0.457 → 0.466 |
| Same initializer, seed 29 | 40.01% → 39.15% | 9.48 → 9.49 | 0.464 → 0.476 |
| End-to-end hybrid, seed 17 | 39.09% → 39.93% | 9.69 → 9.66 | 0.446 → 0.445 |

The synthetic results show similar image quality and mixed mesh results. They do not establish a ScanNet++ gain.

## Room preparation and candidate commands

Prefer the first available suitable room among `8b5caf3398`, `b20a261fdf`, `66c98f4a9b`, and `88cf747085`, in this order, after inspecting fitting RGB coverage and checking that a co-registered scan exists. Keep the reference scan out of fitting and choice of hyperparameters. Record the exact release and pose files. Use the official [ScanNet++ DSLR undistortion](https://github.com/scannetpp/scannetpp) if the supplied cameras are fisheye, and export the calibration as an undistorted PINHOLE COLMAP model. Original transforms can have been estimated from all views. The procedure preserves those calibration parameters and rebuilds the actual feature tracks from fitting images only. The measured integration preflight used synthetic fixed calibration.

Create a JSON manifest with keys `fit`, `edit`, and `final`, each listing original relative image filenames in the desired order. Use translated camera blocks and select a fitting reference view from RGB and camera coverage. The adapter enforces at least 0.15 m final-to-fit and 0.08 m edit-to-fit translation. A full room needs a registered scan in the supplied camera frame, with distance units in metres and any registration fixed before scoring. Record the alignment provenance beside the results.

```bash
python -m pip install pycolmap==3.11.1 open3d
python -m scripts.indoor_triangles prepare-room \
  --model "$PINHOLE_COLMAP" --images "$UNDISTORTED_RGB" --split "$SPLIT_JSON" \
  --reference-view "$FIT_REFERENCE_IMAGE" --pose-provenance "$POSE_PROVENANCE" \
  --width 240 --cell 18 --out "$ROOM_INPUT"

CANDIDATE_CONFIG=scripts/indoor_triangles/room_config.json
python -u -m scripts.indoor_triangles fit --data "$ROOM_INPUT" --out "$ROOM_RUN" \
  --tracks "$ROOM_INPUT/fitting_tracks.npz" --config "$CANDIDATE_CONFIG" --seed 17
python -u -m scripts.indoor_triangles evaluate --data "$ROOM_INPUT" --run "$ROOM_RUN" \
  --out "$ROOM_EVAL" --reference-mesh "$REGISTERED_SCAN" --reference-masks "$EVALUATOR_MASKS"
python -m scripts.indoor_triangles lpips --eval "$ROOM_EVAL"
```

The optional evaluator mask NPZ uses `kinds` of shape (final views, height, width), with 0 for floor, 1 for wall, 2 for doorway rear targets and −1 for unspecified. Without it only whole-scene metrics and RGB scores are defined. Set mask labels from evaluator geometry and view definitions independent of any candidate. Use repeating locked scene manifests. Classical stereo has a measured synthetic arm only.

## Orange4F full capture with stock Triangle Splatting

Use the original [Triangle Splatting repository](https://github.com/trianglesplatting/triangle-splatting)
at `685ff270a8273afc3a2b094ae3148dd0b98e31e9`. To add its CUDA rasterizer to
the existing `3dgs` environment, initialize the pinned submodule and its GLM
dependency, then build against the installed PyTorch and CUDA toolkit:

```bash
conda activate 3dgs
cd ~/triangle-splatting-upstream
git submodule update --init --recursive -- submodules/diff-triangle-rasterization
MAX_JOBS=4 python -m pip install --no-build-isolation --no-deps ./submodules/diff-triangle-rasterization
python -c 'import torch; from simple_knn._C import distCUDA2; from diff_triangle_rasterization import TriangleRasterizer'
```

This uses the existing `simple_knn` in `3dgs`. Run the launcher from this fork
in the same environment. If the rasterizer is already installed in the upstream
`triangle_splatting` environment, use that environment. Video export requires
`ffmpeg` with `libx264`.

```bash
conda activate 3dgs
cd ~/gaussian-splatting-orange4f
python -u -m scripts.orange4f_stock_triangle
```

The default uses 1,670 sharp stride-3 fitting keyframes and the other 3,339
views as test. It reuses a matching baseline manifest if available, or selects
the sharpest image per consecutive group of three. Set `--keyframe-stride` to
change the group size or supply any registered keyframe subset using
`--keyframes /path/to/keyframes.json` with a `selected_images` list. All 5,009
views are rendered and scored. The launcher checks the frame count and RGB
files before starting.

Training uses a copy of the pinned upstream source with a coverage fix. The
trainer records triangle coverage from the first step so pruning can use it
before a full pass over the training cameras. It skips densification windows
with no eligible parents and pruning windows that would remove every triangle.
`training_source.json` records the upstream revision and patch digest.

The default run uses 30,000 iterations and stages images up to 1,600 pixels
wide. Staging links existing RGB images that already fit the width cap and
caches PNGs for images that require resizing or RGB conversion. Use `--width 640`
for a lower-resolution debugging run. RGB camera tensors live on the CPU.
Larger groups of render cameras run in batches. The source is
the existing COLMAP BA export at
`/data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full`. The trainer uses its
fixed poses and sparse point cloud. The stock projection uses centered
principal points. The supplied SfM may include observations from held-out
frames, so this is an Orange4F stress test with supplied initialization. For a
fitting-only initialization, pass `--point-cloud` in the same coordinates.

Outputs are under `/data/3dgs_outputs/orange4f/3dgs_finetuning/`. The run saves
`split_manifest.json`, raw commands and logs, stock checkpoints, and the
direct `direct.off` mesh. `results.json` contains the per-image-average PSNR
for train, test, and all views. `comparisons/` contains one labeled side-by-side
GT and render per image. The `train/`, `test/`, and `all/` folders each contain
an RGB and error gallery and per-frame scores.

To render and score the 7,000-step checkpoint while training continues on
GPU 0, launch this in a separate terminal on GPU 1:

```bash
conda activate 3dgs
cd ~/gaussian-splatting-orange4f
CUDA_VISIBLE_DEVICES=1 python -u -m scripts.orange4f_stock_triangle \
  --width 1600 --evaluate-iteration 7000
```

Evaluation waits for the checkpoint files to finish writing and takes a
private snapshot. It reuses the staged images and split from the active run.
The `evaluation_7000/` directory within that run contains renders, galleries,
and train, test, and all-view PSNR in `results.json`. Pass `--run` for a custom
training directory.

### Live training previews

New RGB and DA3-guided training jobs save GT, rendered images, error maps and
per-frame PSNR at step 1, every 500 steps, and the final step. The same two
training and two held-out views are used across runs with the same split. The
default selection includes the established floor frames `09882.jpg` and
`09884.jpg` when available. Set `--debug-preview-every 250` for a shorter
interval, or repeat `--debug-preview-image NAME` to use specific original
capture filenames. Previews show the live model and use the existing trainer
GPU. Held-out images are only read for the previews.

Once training prints the live preview path, use a second terminal after
setting `RUN` to that experiment's output directory:

```bash
conda activate 3dgs
cd ~/gaussian-splatting-orange4f
python scripts/serve_training_previews.py --model "$RUN/model" --host 127.0.0.1 --port 8082
```

Open `http://localhost:8082/` locally or forward port `8082` from the server
through your SSH session. The dashboard follows the latest step and lets you
inspect earlier steps. Images and `latest.json` are in `model/debug_previews/`.

For a fixed-pose comparison on the latest held-out previews of the DA3 + planes
run, run the following from the fork in the `3dgs` environment. This CPU pass
reads the existing RGB renders and training-view DA3 cache.

```bash
python -u -m scripts.orange4f_photo_detail
```

The pass selects up to four nearby training cameras with similar viewing
directions. It reprojects their photographs at staged resolution with the
cached camera-space depths and checks visibility. Where two sources agree
on depth, color and local detail, it combines fine detail from one source
with low-frequency color from the baseline. Elsewhere it uses the original
render. The target photograph enters only during scoring. The fixed Orange4F
calibration and supplied SfM provenance are inherited from the baseline.

Open the `*_gt_baseline_photo.jpg` comparisons in the printed output directory.
The support PNG identifies eligible pixels. The JSON records full-frame PSNR,
high-frequency RGB RMSE on the 0–255 scale, and support fraction per view.
Use `--run` for another DA3-guided run and `--all-held-out` for a completed
run with its flat `frames/` renders. These are view-dependent RGB results
using additional stored source images and depths. Mesh quality requires
separate evaluation.

For a full-color rendering from the same poses, use
`python -u -m scripts.orange4f_photo_detail --full-color`. The nearest eligible
training camera supplies RGB anywhere its warp is valid. The triangle render
fills the rest. Source selection uses camera centers and orientations only.
The output includes the full-frame comparison, lossless RGB, mask, and PSNR.
Add `--all-held-out` to apply the same rule to the completed test renders.

To inspect a failed transfer, run `python -u -m scripts.orange4f_photo_detail --inspect-warps`.
For each held-out preview, this writes a full-resolution GT / baseline / single
reprojected training photo comparison for each candidate source. Gray marks
unsupported pixels. The separate PNG records the valid pixels. Each source
and the baseline are scored on the same valid pixels, also reported for
reference edges with luminance gradient of at least 16 levels per pixel.
The GT is used only after the warps are computed. The command uses the CPU
and the same snapshot inputs as the preview comparison.

To continue training an earlier run from its 7k snapshot, set `RUN` to the
original run directory and launch the updated trainer:

```bash
conda activate 3dgs
cd ~/gaussian-splatting-orange4f
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.orange4f_stock_triangle \
  --resume-from "$RUN" --resume-iteration 7000
```

The original snapshots stay in place. The new run is written beside `RUN`
with the suffix `_warm_7000_to_30000`. It inherits the source split, camera
calibration, and DA3 and plane weights, if present. The first preview shows
the loaded 7k model. Training starts at 7,001 with fresh Adam moments,
camera sampling, and densification statistics because the older snapshots
only saved render state.

Open `frames/` in the run or inside `evaluation_7000/` to browse rendered PNGs
in one folder by capture filename. Render batches are internal and keep memory
use bounded. To create the same flat view for an existing 7k evaluation,
including batches that have already finished during an ongoing evaluation:

```bash
conda activate 3dgs
cd ~/gaussian-splatting-orange4f
python -m scripts.orange4f_stock_triangle --link-rendered-iteration 7000
```

The link command uses the saved split and creates symlinks to existing PNGs.

### Orange4F DA3 and plane ablation

Compare the existing RGB run with two matched Triangle Splatting arms. All
three use the same 1,670 training views, sparse initialization, camera poses,
resolution, and 30k schedule. Prior preparation reads only the training RGBs.
It uses the AmbiSuR DA3 model in overlapping fixed-pose windows, aligns scale
to the supplied cameras, checks it against observed SfM tracks when available,
and keeps confident depth interiors with agreement in two other views. The
cache manifest reports which windows could validate scale against SfM.

```bash
cd ~/gaussian-splatting-orange4f
BASE=/data/3dgs_outputs/orange4f/3dgs_finetuning/triangle_splatting_stock_full_stride3_w1600_30000
CACHE="$BASE/priors_da3"

conda activate gsstudio  # existing DA3-enabled environment
CUDA_VISIBLE_DEVICES=1 python -u -m scripts.orange4f_triangle_priors --run "$BASE" --out "$CACHE"

conda activate 3dgs
CUDA_VISIBLE_DEVICES=1 python -u -m scripts.orange4f_stock_triangle \
  --match-run "$BASE" --da3-priors "$CACHE"

CUDA_VISIBLE_DEVICES=2 python -u -m scripts.orange4f_stock_triangle \
  --match-run "$BASE" --da3-priors "$CACHE" --plane-weight 0.08
```

The first arm adds a robust log camera-z loss on stock rendered median depth.
The renderer's median depth uses triangle centroids. The second arm adds a
PlanarGS-style point-to-plane loss directly on contributing triangle vertices.
Its planar patches are fitted to multiview-validated DA3 pixels. Training logs
show valid depth pixel and plane face counts every 500 steps. The guided runs
have `_da3_depth` and `_da3_planes` suffixes. The existing train and test
evaluation produces comparable PSNR, renders, and mesh exports. Independent
registered LiDAR can be used to measure geometric error on the same views.

`original_trajectory.mp4` shows GT and renders side by side at their staged
resolution for all registered frames in natural filename order at 30 fps.
At the default resolution each side is up to 1,600 pixels wide.
`random_trajectory.mp4` is a
40-second render along a seeded smooth path derived from a continuous capture
segment, with no matching GT. Its actual poses and seed are recorded in
`random_trajectory.json`. `--prepare-only` stages the data without starting
training. Rerun with the same settings to use the saved final checkpoint.
To also generate a fused TSDF mesh using stock code:

```bash
conda activate triangle_splatting
RUN=/data/3dgs_outputs/orange4f/3dgs_finetuning/triangle_splatting_stock_full_stride3_w1600_30000
cd ~/triangle-splatting-upstream
time python mesh.py -s "$RUN/fit_input" -m "$RUN/model" -r 1 --data_device cpu \
  --iteration 30000 --skip_train --skip_test
```

The fused and postprocessed meshes are written to `model/train/ours_30000/`.

## Orange4F local-chart pilot

Run directly on the Orange4F COLMAP BA export at
`/data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full`. The default
reference is `12569.jpg`, the floor diagnostic frame. The command selects
nearby, same-facing cameras using the fixed COLMAP poses, reserves translated
edit and final views, and rebuilds triangulated tracks from fitting RGB only.
The loader normalizes stored quaternions before camera selection and triangulation
and checks that the resulting rotations and translations stay fixed.
This is a CPU pilot on one local chart. It starts at 384 pixels wide and uses
up to 18 fitting views, three edit views and four held-out views.

```bash
RUN="/data/3dgs_outputs/orange4f/3dgs_finetuning/elastic_hybrid_12569_$(date +%Y%m%d-%H%M%S)"
python -u -m scripts.indoor_triangles orange4f --out "$RUN"
```

The command needs `pycolmap==3.11.1` in the active environment. If the source is
elsewhere, set `--source`. Try another local chart with `--reference-view`, or
adjust `--window` (registered frames on either side) and `--radius` (metres)
if there are too few translated views. The camera split is saved in
`split.json`. The hybrid checkpoint is `run/hybrid.npz`, and mesh exports,
per-view RGB/depth/normal previews, and the metrics table are linked from
`eval/index.html`. View depth and normals are predicted geometry. Supply
`--reference-mesh` with a registered triangle mesh to also score geometry.

## Stock CUDA references

Pinned Git revisions, inspected here, have separate original renderers and shader capacities.

| Reference | Revision | Surface path |
|---|---|---|
| [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) | `54c035f7834b564019656c3e3fcc3646292f727d` | Fit-view depth TSDF adapter below |
| [Triangle Splatting](https://github.com/trianglesplatting/triangle-splatting) | `685ff270a8273afc3a2b094ae3148dd0b98e31e9` | Official fit-only bounded TSDF and its separate cleaned export |
| [MeshSplatting](https://github.com/meshsplatting/mesh-splatting) | `2a810a6c353215685307da3d4cc6ebd73b1c387b` | Direct PLY from final checkpoint |

Use the above revisions, their submodules and separate CUDA environments. Invoke from each reference repository root. Paths for both stock inputs and all models should be absolute. Verify that `stock_centered_camera_compatible` is true in the room manifest before using these stock loaders. With off-center principal points, a common fixed crop/reprojection or an explicit calibrated adapter is needed. The stock train source contains only fitting RGB and points. Render the separate final-view source at evaluation. Keep final-view renders in the original model's ordered camera manifest, with filename order checked against the staged undistorted images.

```bash
FIT="$ROOM_INPUT/stock_fit"
FINAL="$ROOM_INPUT/stock_final"
ITER=30000

# From the stock 3DGS checkout.
time python train.py -s "$FIT" -m "$GS_RUN" -r 1 --iterations "$ITER"
python render.py -m "$GS_RUN" -s "$FINAL" -r 1 --iteration "$ITER" --skip_test
# From the fork root, using the stock 3DGS CUDA environment with Open3D.
python -m scripts.indoor_triangles.stock_3dgs_tsdf --repo "$GS_REPO" --source "$FIT" \
  --model "$GS_RUN" --iteration "$ITER" --out "$GS_RUN/fitting_tsdf.ply"

# From stock Triangle Splatting.
time python train.py -s "$FIT" -m "$TRI_RUN" -r 1 --iterations "$ITER"
python render.py -m "$TRI_RUN" -s "$FINAL" -r 1 --iteration "$ITER" --skip_test
time python mesh.py -m "$TRI_RUN" -s "$FIT" -r 1 --iteration "$ITER" \
  --skip_train --skip_test --voxel_size 0.025 --sdf_trunc 0.075 --depth_trunc 12

# From stock MeshSplatting. Keep optional depth and normal prior assets absent.
time python train.py -s "$FIT" -m "$MESH_RUN" -r 1 --iterations "$ITER"
python render.py -m "$MESH_RUN" -s "$FINAL" -r 1 --iteration "$ITER" --skip_test
time python create_ply.py "$MESH_RUN/point_cloud/iteration_$ITER" \
  --out "$MESH_RUN/direct.ply" --cpu
```

MeshSplatting's `--indoor` preset also enables an optional supervised normal weight; verify the exact preset and input prior flags before using it for another labeled arm. OMeGa's published geometry results use different input guidance and are a prior-assisted reference. [OMeGa paper](https://openaccess.thecvf.com/content/WACV2026/papers/Cao_OMeGa_Joint_Optimization_of_Explicit_Meshes_and_Gaussian_Splats_for_WACV_2026_paper.pdf)

Score the stock direct/TSDF and postprocessed meshes separately:

```bash
python -m scripts.indoor_triangles score-external --data "$ROOM_INPUT" \
  --reference-mesh "$REGISTERED_SCAN" --reference-masks "$EVALUATOR_MASKS" \
  --mesh "$GS_RUN/fitting_tsdf.ply" --renders "$GS_RUN/train/ours_$ITER/renders" \
  --config scripts/indoor_triangles/room_config.json --out "$GS_RUN/paired_evaluation.json"
```

The experimental 3DGS adapter conditions the rasterizer's accumulated inverse depth by alpha before TSDF at 2.5 cm voxels, 7.5 cm truncation, 12 m maximum z and alpha at least 0.55. Its time and memory are tracked separately. These stock and real-room commands were inspected but not executed on CUDA or ScanNet++ in this workspace.
