# Regional blur ablation

## Headless SuperPoint + SuperGlue with sequential overlap (GPU 1)

For the video sequence, use SuperPoint features and the official indoor
SuperGlue model on GPU 1. Each selected training image is paired with the next
20 selected images in natural filename order. For 1,349 images this is 26,770
pairs, rather than 909,226 exhaustive pairs. The overlap counts **selected
frames**, not raw video frame numbers. If this graph is disconnected, automatic
recovery adds bounded cross-component temporal and appearance proposals below.

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate 3dgs &&
cd /home/sequor_vfm/gaussian-splatting &&
git pull --ff-only &&
bash scripts/run_region_superglue.sh 1 \
  /data/3dgs_outputs/orange_4F/1_full/3dgs_finetuning/regions/wall_06663_v1
```

The first invocation caches the official
`magicleap/SuperGluePretrainedNetwork` code/weights at commit
`ddcf11f42e7e0732a0c4607648f9448ea8d73590` under
`~/.cache/3dgs/SuperGluePretrainedNetwork-ddcf11f` (or `SUPERGLUE_REPO`).
It uses the current `3dgs` environment and does not install packages or change
PyTorch. The models use CUDA without any display, Qt window, or OpenGL context.
The native COLMAP executable may be a CPU-only build.

Defaults are a maximum image side of 1,600 pixels (no upscaling), at most 4,096
SuperPoint keypoints, NMS radius 4, keypoint threshold 0.005, indoor SuperGlue
weights, 100 Sinkhorn iterations, and match threshold 0.2. The script converts
resized integer-centered SuperPoint coordinates into original-resolution
COLMAP half-pixel coordinates before geometric verification. RGB training
resolution and the original camera calibration are unchanged.

COLMAP's `feature_importer` first creates its native image/camera/rig records
using empty feature placeholders; these are replaced with SuperPoint keypoints.
The float SuperPoint descriptors stay in the disk cache. They are not converted
into fake SIFT descriptors. `matches_importer --match_type raw` then runs CPU
geometric verification on the **SuperGlue correspondences**, with GPU SIFT
disabled. Neither the SIFT extractor nor the exhaustive/sequential SIFT matcher
is invoked. Mapping, fixed-intrinsics BA, LiDAR-frame alignment and Gaussian
training follow automatically. Disconnected verified groups trigger recovery
in the same workspace; missing registrations still stop before GS training.

New outputs (the failed SIFT workspace is left available):

- `$REGION/colmap_superpoint_superglue_seq20_v1/`: `superglue.json` records model
  hashes/settings; `pairs.txt` records the exact temporal pairs. Per-image
  `superpoint/` and per-pair `superglue_matches/` caches are written atomically
  as they finish. CPU memory holds at most 32 feature sets; GPU memory holds
  the current network/pair. Restarting the same command reuses completed work.
- `superglue_matching_report.json`: raw correspondence counts;
  `verification_report.json`: verified pair counts, connected components and
  isolated images. `bridge_pairs.json` records any additional pairs and their
  selection reason. All bridge matches pass the same COLMAP verification.
- `$REGION/runs/crop_colmap_superglue_seq20_fixed_v1/`: the same fixed-LiDAR
  3DGS + AA control, 90k steps for this export, no culling or scale cap, previews
  every 500 steps and parallel regional ROI evaluation every 10k. Only training
  poses are replaced; held-out poses and LiDAR initialization remain unchanged.

For a different overlap, prefix the command with e.g.
`COLMAP_SEQUENTIAL_OVERLAP=30`; the output names then contain `seq30`.
`SUPERPOINT_MAX_SIZE` and `SUPERPOINT_MAX_KEYPOINTS` override feature extraction
limits. Changing those requires a fresh VERSION (third positional argument).

### Recover a disconnected sequential graph without starting over

Rerun the **same command and version** after pulling. Existing `features.done`,
per-image descriptors, pair caches and COLMAP geometry rows are reused. A row
with zero inliers is also a completed attempt and is not reverified. Interrupted
new pairs resume from their existing match caches.

The launcher first verifies only missing pairs. If components remain, it
proposes up to 512 untried pairs per round, balancing short temporal gaps and
pooled SuperPoint descriptor similarity across components. Similarity only
selects candidates: SuperGlue matches them, then COLMAP verifies them with the
same settings. No old camera poses or held-out images are used for recovery.
Up to three rounds run per invocation; `SUPERGLUE_BRIDGE_BUDGET` and
`SUPERGLUE_BRIDGE_ROUNDS` change these bounded budgets. Original sequential
settings and the workspace name remain unchanged. `pairs.txt` now includes
base and added pairs. The ledger is retained across restarts.

Connectivity alone does not prove successful SfM. Mapping, all-image registration
and alignment checks still run before GS. If no valid bridges are found, the
component sizes and isolated names remain in `verification_report.json`; the
pipeline does not merge them using invented correspondences or drop images.

### Additional learnable-geometry controls (GPUs 2 and 3)

Keep the two fixed-LiDAR controls. These two **additional** runs use the same
LiDAR initialization, images, masks and 90k schedule, with position optimization
at 0.05 times the native LR, densification and native pruning. Position LR
endpoints before scene-extent scaling are `8e-6` and `8e-8`; updates start at
step 1. Native clone/split/prune runs every 100 steps after 500 and before 15k.
Scale LR stays 0.001, scales remain unconstrained, culling and opacity resets
stay disabled. Previews run every 500 steps; snapshots and regional train/held-out
ROI evaluation run every 10k, plus the exposure-matched checkpoint.

```bash
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate 3dgs &&
cd /home/sequor_vfm/gaussian-splatting && git pull --ff-only
REGION=/data/3dgs_outputs/orange_4F/1_full/3dgs_finetuning/regions/wall_06663_v1
# Original poses: can start independently of COLMAP.
bash scripts/run_region_adaptive.sh 2 "$REGION" original
# Separate terminal, after the SuperGlue COLMAP dataset export finishes:
bash scripts/run_region_adaptive.sh 3 "$REGION" colmap
```

Models are `$REGION/runs/crop_original_adaptive_lr005_v1` and
`$REGION/runs/crop_colmap_superglue_seq20_adaptive_lr005_v1`. The second reads the
accepted `colmap_superpoint_superglue_seq20_v1/dataset` export, rechecks its audit,
and still initializes from the original LiDAR, never the new SfM sparse cloud.
It does not start before that export exists. Camera poses stay fixed within each
GS run; the learnable positions here are **Gaussian centers**, not camera poses.
Set `COLMAP_VERSION`/`COLMAP_SEQUENTIAL_OVERLAP` if using another COLMAP workspace.

### Fifth run: LiDAR adaptation of the self-constrained prior (GPU 4)

The five comparisons are:

| Run | GPU | Camera poses | Gaussian geometry | Extra LiDAR prior |
| --- | --- | --- | --- | --- |
| 1 | 0 | Original | Fixed centers, constant population | None |
| 2 | 1 | SuperPoint/SuperGlue + COLMAP | Fixed centers, constant population | None |
| 3 | 2 | Original | Learnable centers, densification and pruning | None |
| 4 | 3 | SuperPoint/SuperGlue + COLMAP | Learnable centers, densification and pruning | None |
| 5 | 4 | Original | Same adaptive settings as run 3 | Distance-conditioned opacity and periodic surface pull |

All initialize from the same cropped LiDAR. Runs 2 and 4 use the accepted COLMAP
export; held-out poses remain original. Run 5 is independent of that export.

GSPrior ([paper](https://arxiv.org/abs/2603.19682)) builds a periodically updated
TSDF from rendered depths, penalizes opacity according to surface distance, and
projects Gaussian centers onto the estimated surface. **This run is a LiDAR
adaptation, not a reproduction of the complete method.** It replaces the
self-generated TSDF with a fixed, measured-LiDAR local plane field. It retains
native volumetric 3D Gaussians and omits the paper's depth/NCC/normal losses and
hard prior-based outlier deletion.

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate 3dgs &&
cd /home/sequor_vfm/gaussian-splatting &&
git pull --ff-only &&
bash scripts/run_region_scp.sh 4 \
  /data/3dgs_outputs/orange_4F/1_full/3dgs_finetuning/regions/wall_06663_v1
```

Output: `$REGION/runs/crop_original_adaptive_lr005_lidar_scp_v1`. Supply a third
argument, e.g. `v2`, for a fresh model; the fixed prior cache is reused. The
launcher refuses to overwrite a model or stop another GPU process.

Preprocessing uses eight CPU workers and saves `$REGION/lidar_scp_v1/field.npz`
and `metadata.json`. It computes the median RMS distance to three nearest
neighbors on a deterministic sample, sets cell width `h = 2 * median`, and
retains an actual measured point per occupied voxel. Local PCA estimates normals
with 24 neighbors and rejects unreliable or distant support. Source and cache
hashes prevent reuse with different LiDAR. This cache is a surface reference;
it does not downsample or replace Gaussian initialization. Training loads its
sparse constants into GPU memory once; neighborhood queries use CUDA PyTorch
operations in bounded batches. Preprocessing itself uses CPU SciPy, not CUDA.

For a currently supported Gaussian, signed local plane distance is
`d = dot(x - anchor, normal)`. With band width `b`, normalized distance is
`s = min(abs(d)/b, 1)`. The opacity target is 1 when `s <= 0.3` and 0 otherwise,
weighted by PCA confidence and `1/(1+s)^2`. A uniform 16,384-point sample
estimates the scene-wide opacity penalty each step; unknown/unreliable
neighborhoods contribute zero. No gradient through the discrete correspondence
or opacity target is used to move XYZ.

The prior starts after 5k steps and ramps to opacity weight 0.01 over 2k steps.
The physical band shrinks from `2h` to `h` by 20k, without scene-extent scaling.
For illustration only, median spacing 0.018 in meter-valued inputs gives
`h=0.036 m`, band 0.072 -> 0.036 m, and on-surface band 0.0216 -> 0.0108 m.
These are distances for the prior, **not Gaussian scale caps**.

After each 100-step interval, a cycling sample of up to 262,144 centers receives
a normal-direction pull of at most 25% of its supported surface distance,
bounded to 25% of the current band per update. It runs after Adam and population
changes. Adam moments are cleared only for moved XYZ rows. Correspondences are
recomputed at current positions, including newly cloned/split children.
Centers outside reliable local support are unchanged; this is not a global
nearest-surface attraction or a guarantee of recovering distant floaters.

Training stays at 90k for this export, with position LR 0.05x native,
unconstrained scales, native densification/pruning, AA, and opacity resets off.
There is no ray culling during training or inference. RGB evaluation renders
the saved Gaussians normally and needs no prior lookup. Previews are saved at
step 1/every 500/final; snapshots and all **4,039 regional frames' ROI PSNR** run
every 10k in the existing parallel evaluator, plus the matched checkpoint.

`lidar_scp_training.jsonl` records sample support, surface distance, opacity
gradient strength relative to RGB, pull count and maximum displacement.
`lidar_scp_cuda_smoke.json` records the startup CUDA checks for tilted-plane
pulls, opacity gradients, unknown geometry, Adam state and densified children.
CPU tests do not substitute for this CUDA check or demonstrate real-scene
quality. Inspect these diagnostics and the unchanged 500-step GT/render/error
previews before attributing any quality change to the prior.

```bash
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate 3dgs &&
cd /home/sequor_vfm/gaussian-splatting &&
python scripts/serve_training_previews.py \
  --model /data/3dgs_outputs/orange_4F/1_full/3dgs_finetuning/regions/wall_06663_v1/runs/crop_original_adaptive_lr005_lidar_scp_v1 \
  --port 8095
```

Open `http://211.172.247.108:8095` after the first preview is saved.

Once Gaussian training starts, open its live preview in another terminal:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate 3dgs &&
cd /home/sequor_vfm/gaussian-splatting &&
python scripts/serve_training_previews.py \
  --model /data/3dgs_outputs/orange_4F/1_full/3dgs_finetuning/regions/wall_06663_v1/runs/crop_colmap_superglue_seq20_fixed_v1 \
  --port 8094
```

Browse `http://211.172.247.108:8094`. For comparison with the original-pose
control, use `compare_region.py --different-poses` as below, replacing the
second model with `crop_colmap_superglue_seq20_fixed_v1`.

## Rerun COLMAP on the selected training images (GPU 1)

This ablation reruns feature extraction, exhaustive matching, incremental SfM,
and bundle adjustment using **only the existing region's training images**.
The mapper starts from feature matches, without the old poses as an initial
reconstruction. Existing undistorted PINHOLE intrinsics stay fixed. An existing
COLMAP executable must be on PATH, or set `COLMAP_BIN=/path/to/colmap`; the
launcher does not install or change environments.

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate 3dgs &&
cd /home/sequor_vfm/gaussian-splatting &&
git pull --ff-only &&
bash scripts/run_region_colmap.sh 1 \
  /data/3dgs_outputs/orange_4F/1_full/3dgs_finetuning/regions/wall_06663_v1
```

For this older SIFT workflow, `COLMAP_SIFT_DEVICE=auto` uses GPU SIFT only if the
installed COLMAP build banner says `with CUDA`; otherwise extraction/matching
use CPU, capped at eight threads where supported. `QT_QPA_PLATFORM=offscreen`
alone does not supply the OpenGL context required by non-CUDA GPU SIFT.
`COLMAP_SIFT_DEVICE=cpu` explicitly forces CPU; `cuda` requires an advertised
CUDA build. GPU SIFT is limited to physical GPU 1 (logical device 0 inside
COLMAP). GS still trains/evaluates on physical GPU 1. Mapping and bundle
adjustment use COLMAP's default CPU solver. CLI option names are read
from the installed COLMAP help, supporting the older `SiftExtraction` /
`SiftMatching` and newer `FeatureExtraction` / `FeatureMatching` option groups.
Alignment likewise selects the installed `alignment_max_error` or legacy
`robust_alignment_max_error` flag; the legacy robust-alignment switch is enabled
when exposed. A failed option check stops before feature extraction or training.
After pulling a compatibility fix, rerun the same command and version to reuse
the prepared workspace; no deletion or new export is needed.
Exhaustive matching tries all selected-image pairs and can take substantial
time. Per-stage logs and successful stage markers are saved; interrupted
preprocessing reuses completed stages. The launcher never stops another run.

For the completed `wall_06663_v1` export, it uses 1,349 training images; the
2,690 held-out images are excluded from COLMAP. It requires all selected train
images in one reconstructed component; missing registrations are listed in
`registration_report.json` and block training, rather than silently changing
the ablation. A robust similarity alignment to the original training camera
centers puts the new poses back into the LiDAR coordinate frame. Alignment
uses a RANSAC threshold of 5% of the maximum training-camera distance from the
centroid, in original scene units. Fewer than 70% inlier centers or median
rotation discrepancy above 15 degrees blocks export. These checks only catch
gross alignment failures; they do not establish that the new poses are better.
Per-frame center/rotation differences are saved in `alignment_report.json`.

Outputs, with `REGION` denoting that existing regional dataset:

- `$REGION/colmap_train_v1/`: training-only image links, calibrated database,
  COLMAP stage logs, sparse models, alignment transform and pose audit.
- `$REGION/colmap_train_v1/dataset/`: new training poses, **unchanged held-out
  poses**, identical RGB and ROI masks, and the same cropped LiDAR initialization.
  The new COLMAP sparse points are not used to initialize Gaussians. Original
  dataset files are never overwritten.
- `$REGION/runs/crop_colmap_fixed_v1/`: 3DGS + AA with the same settings as
  `crop_v2`: fixed LiDAR centers, scaling LR 0.001, unconstrained scales, no
  densification/pruning/opacity reset/culling. With 1,349/1,670 selected frames,
  training lasts 90k steps and includes the matched checkpoint at 80,778.

Checkpoints and a separate same-GPU evaluator run every 10k steps, plus the
matched checkpoint. Evaluation covers all 4,039 exported regional frames and
their original ROI pixels. Training PSNR uses the **new training poses**;
held-out PSNR uses the unchanged original held-out poses. This tests corrected
training supervision, not held-out COLMAP relocalization or all-5,009-frame PSNR.
The original and new training poses therefore differ in train comparisons.

GT/render/error previews start at step 1, repeat every 500 steps, and include
fixed random training and held-out frames. Once training reaches its first
preview, run in another terminal:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate 3dgs &&
cd /home/sequor_vfm/gaussian-splatting &&
python scripts/serve_training_previews.py \
  --model /data/3dgs_outputs/orange_4F/1_full/3dgs_finetuning/regions/wall_06663_v1/runs/crop_colmap_fixed_v1 \
  --port 8094
```

Open `http://211.172.247.108:8094`. COLMAP preprocessing itself does not produce
Gaussian previews. Use a third launcher argument such as `v2` for a fresh
COLMAP workspace and training model; an existing training model is not reused.

Once both runs have the same checkpoint and GPU 1 is free, compare them using
their own dataset poses and identical images/masks:

```bash
REGION=/data/3dgs_outputs/orange_4F/1_full/3dgs_finetuning/regions/wall_06663_v1
python -u scripts/compare_region.py --region "$REGION" \
  --models "$REGION/runs/crop_v2" "$REGION/runs/crop_colmap_fixed_v1" \
  --iterations 30000 --different-poses --gpu 1 \
  --output "$REGION/comparisons/original_vs_colmap_30000"
```

`--different-poses` is mandatory for this comparison. Without it, the comparator
continues to reject pose differences. Neither mode permits different source
images or intrinsics, and both use the original regional ROI masks.

## Original-pose regional videos at 30k

Render the current `crop_v2` control on GPU 0 after its 30k snapshot has been
published. This reads the checkpoint only; it does not restart or stop training.

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate 3dgs &&
cd /home/sequor_vfm/gaussian-splatting &&
git pull --ff-only &&
bash scripts/render_region_videos.sh 0 \
  /data/3dgs_outputs/orange_4F/1_full/3dgs_finetuning/regions/wall_06663_v1/runs/crop_v2 \
  30000
```

The two videos render sequentially on the specified GPU, with AA on and
external culling off. They use the original regional dataset poses, not the
new COLMAP/SuperGlue reconstruction. Both plans are checked before rendering.
The launcher requires `point_cloud/iteration_30000/ready.json`, so it cannot
read a partly written snapshot. It activates the `3dgs` environment itself.

Under `$REGION/runs/crop_v2/videos/iteration_30000/`:

- `actual/side_by_side.mp4`: **GT | 30k render**, natural filename order, every
  exported regional camera (4,039 for this export: 1,349 train + 2,690 held-out).
  No frame subsampling; playback is 30 fps rather than original capture timing.
- `random/render.mp4`: a 40-second seeded novel trajectory, derived from a
  continuous 180-frame regional segment with modest translation and rotation.
  Novel views have no matching GT images. Camera-support bounds are applied,
  but this is not a mesh collision check.
- Each subfolder contains `trajectory.json` with poses, checkpoint, AA/culling
  settings, frame count and completion status; actual also retains separate
  `original.mp4` and `render.mp4` files.

These are full-frame visualizations, not ROI-masked PSNR evaluation. Areas
outside the cropped geometry and supervised ROI can be incomplete. Panel
width is 1,280 pixels. No all-5,009-frame evaluation is implied by these videos.
Supply a different GPU as the first argument if needed. Existing nonempty
video folders are preserved; pass a fresh fourth argument to render again.

## Automatic selection and adaptive geometry

For an immediate automatic test around `06663.jpg`, this command **exports the
region first**, then prepares and starts training on GPU 0:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate 3dgs &&
cd /home/sequor_vfm/gaussian-splatting &&
git pull --ff-only &&
python -m pip install -r requirements-region.txt &&
bash scripts/run_region_auto.sh 0
```

It uses a 4x4x4 scene-unit box centered on near central LiDAR returns in the
anchor image, plus a 1-unit halo. Automatic candidate filters use the same
approximate LiDAR depth masks described below. This bypasses manual review
explicitly; automatic candidates are not certified as unoccluded.

Outputs are under
`3dgs_finetuning/regions/wall_06663_auto_v1/runs/crop_adaptive_lr05_v1`.
The wrapper creates the missing regional dataset, reuses it only if its inputs,
box, selection settings and sampling match, and refuses to overwrite a model.
It does not require the earlier `wall_06663_v1` manual export. For a subsequent
fresh run with the same automatic dataset, pass `GPU REGION v2`.
`REGION_SOURCE`, `REGION_BASELINE` and `REGION_OUTPUT_ROOT` optionally override
the scene defaults. Existing GPU jobs are never stopped by this wrapper.

This mode uses 3DGS + AA with:

- **Position updates from step 1 at 0.05x native LR** (divisor 20): initial
  `8e-6`, final `8e-8`, before native camera-extent scaling. No fixed-position warmup.
- Native adaptive cloning/splitting and opacity/size pruning: start 500,
  end 15,000, interval 100, gradient threshold `0.0002`, split boundary
  `0.01 * regional_scene_extent`. The small regional camera extent is recomputed;
  this is not the earlier whole-scene `0.0001` override.
- Scaling LR `0.001`, unconstrained scales, opacity resets disabled, original
  camera poses, and no external culling. Initialization uses all crop+halo LiDAR.
- At least 30k steps with the exposure-matched checkpoint described below,
  10k checkpoints and regional ROI evaluation. GT/render/error previews remain
  enabled at step 1, every 500 steps and the final step.

Open the training previews from a second terminal:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate 3dgs &&
cd /home/sequor_vfm/gaussian-splatting &&
python scripts/serve_training_previews.py \
  --model /data/3dgs_outputs/orange_4F/1_full/3dgs_finetuning/regions/wall_06663_auto_v1/runs/crop_adaptive_lr05_v1 \
  --port 8092
```

Browse `http://211.172.247.108:8092`. The evaluator measures only the exported
regional frames and ROI pixels, not all 5,009 full-scene images.

For a manually exported region, the same adaptive mode is available through
`scripts/run_region_ablation.py --region REGION --geometry adaptive
--position-lr-multiplier .05 --model MODEL --gpu 0`, followed by
`bash scripts/run_prepared_reconstruction.sh MODEL`.
For headless automatic export alone, use `scripts/select_region.py` with
the source, baseline, output and box arguments below plus `--automatic --export`;
add `--reuse` to validate and reuse an existing automatic export.

## Fixed-position control and manual selection

Question: does a fixed-position reconstruction become sharp when trained on
images of one physical region, and does removing the rest of the LiDAR matter?
The first diagnostic uses 3DGS + AA, original poses, fixed original LiDAR XYZ,
learned appearance/opacity/rotation/scales, scaling LR 0.001, no scale constraint,
no pruning, no densification, no opacity reset, and no external occlusion culling.
No pruning keeps this diagnostic comparable with the pre-pruning phase of the
200k fixed-position run. This is an explicit diagnostic, not a new default for
the other experiments.

## Select images, poses and LiDAR together

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate 3dgs &&
cd /home/sequor_vfm/gaussian-splatting &&
git pull --ff-only &&
python -m pip install -r requirements-region.txt

ROOT=/data/3dgs_outputs/orange_4F/1_full
SOURCE=/data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full
BASE=$(python -c 'from scripts.render import resolve_models; import sys; print(resolve_models(None, sys.argv[1], ["sharp_keyframes_stride3_aa"])[0])' "$ROOT")
REGION="$ROOT/3dgs_finetuning/regions/wall_06663_v1"

python -u scripts/select_region.py \
  --source "$SOURCE" --baseline "$BASE" --output "$REGION" \
  --anchor 06663.jpg --size 4 4 4 --halo 1 --port 8091
```

Open `http://211.172.247.108:8091`. Adjust **Box center / Box XYZ size** to
include the textured wall and floor. Yellow points are inside the core; blue
points form the surrounding LiDAR halo. Gray points show the rest of the scene.
Use **Find candidate views**, then review the **actual unmasked RGB**:

- **Keep and next** includes the image; **Reject and next** excludes it, for
  occlusion or any other reason. **Previous image / Next image** only navigate.
- **Review filter** shows all, unreviewed, kept or rejected images. Select a
  rejected image and use **Keep and next** to restore it. **Mark unreviewed**
  clears a decision.
- **Export kept images** exports only explicitly kept cameras and RGB images.
  Rejected and unreviewed frames are excluded from **both** train and evaluation;
  kept frames retain their original training/held-out assignment. At least two
  kept training views are needed. Counts update after every decision.
- The optional **Show approximate ROI mask** overlay is separate from the actual
  RGB. Keeping an image does not certify that every point-derived mask pixel has
  correct visibility. Review the mask too before using it for supervision.

Decisions and box/search settings autosave to `OUTPUT.review.json` outside the
fresh export directory. Restart with the same command/output to resume. The
export also contains `review.json` with candidate names and all decisions.
Changing the box or search bounds invalidates the review; finding candidates
again starts a new one. Repeating an unchanged search preserves decisions.
The source files are never overwritten. The output must be a fresh directory.

By default a view needs at least 2% region overlap and its camera must be within
8 scene units of the box. Set camera distance to 0 to search every camera.
The box dimensions, halo, and distances use original scene units (meters only
if the input was registered in meters). The anchor seeds a box using near
central LiDAR returns; it is not an automatic semantic room segmentation.

Selection uses a deterministic sample of up to 1M points, projected with COLMAP
intrinsics into a 256px nearest-depth buffer. Known nearer non-region hits are
excluded. Only tiny unsampled holes are closed; boundaries are eroded. These
masks are approximate measured-LiDAR visibility, not a complete scene surface
or ground-truth visibility. Inspect RGB overlays, particularly near depth edges
and sparse returns. `--sample-points 2000000` can increase sampling density.
Training itself uses no such depth buffer or ray clipping.

Projection uses a direct three-coordinate NumPy contraction rather than a
multithreaded BLAS matrix multiply. Packed per-view masks are saved immediately
under `OUTPUT.projection_cache/`, outside the final export directory. The cache
is keyed by the original cloud/pose file stamps, sampling, and box geometry;
changing these does not reuse stale masks. A matching restart reuses completed
masks. Selection, export validation and overlay generation share these masks;
an unchanged selection is also reused within the running process. Masks are
loaded one view at a time, not all into RAM or VRAM. No extra package is needed.

The terminal prints the exact destination and cache paths before loading the
cloud, then camera counts, projected/cached counts and elapsed time during the
search. Export reports its temporary directory, image count, LiDAR crop and
completion. The viewer receives export progress too. The final region directory
still appears only on successful completion; its parent and cache appear early.
If an older selector is running, pulling changes does not update that process;
restart it to use this implementation. The old process's elapsed lifetime is
not necessarily time spent exporting (it also serves the interactive viewer).

Exports:

- `images/`: symlinks to original RGB, without resampling.
- `sparse/0/`: subset cameras and original world-to-camera poses. Intrinsics are
  unchanged; SIMPLE_PINHOLE is expressed equivalently as PINHOLE. Feature tracks
  are omitted because initialization uses the LiDAR PLY.
- `input.ply`: **every original point** inside box + halo, in original order and
  coordinates; `original_lidar_indices.npy` maps to the input cloud rows.
- `keyframes.json`: manually kept candidates intersected with the original
  training manifest. Other **kept** frames remain held-out.
- `masks/`, `overlays/`, `index.html`: loss/metric masks and review gallery.
- `selection.json`, `poses.json`: provenance, camera audit, counts, per-view
  mask fractions and exact box/filter settings.
- `review.json`: kept/rejected/unreviewed decisions for every automatic candidate.

RGB aspect mismatches fail export. Same-aspect pre-resized images are accepted
and their dimensions recorded. Principal-point offsets from the image center
are reported: native 3DGS's centered-FOV projection does not incorporate these
offsets. Large offsets need a separate camera-model correction experiment.

For reproducibility or a second temporal-window ablation, use explicit box
coordinates with `--center X Y Z --size SX SY SZ` and a fresh output, then review
the resulting candidates. Without `--automatic`, `--export` requires a saved manual review and
exports its kept images using the saved box/search settings without starting
Viser. Use `--review-file PATH` to explicitly select a saved review. Optional
`--frame-start NAME --frame-end NAME` restricts the lexical image-name range
(appropriate for this scene's zero-padded names). Keep the same box and halo
when comparing all local visits with one contiguous visit.

## Train and inspect

In another terminal (or after stopping only the selector with Ctrl-C):

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate 3dgs &&
cd /home/sequor_vfm/gaussian-splatting
REGION=/data/3dgs_outputs/orange_4F/1_full/3dgs_finetuning/regions/wall_06663_v1
bash scripts/run_region_ablation.sh 0 "$REGION" crop
```

Default model: `$REGION/runs/crop_v1`. No existing GPU process is stopped. An
occupied GPU or existing plan fails before training. Supply a fourth argument
such as `v2` for a fresh run directory.

The schedule is at least 30k steps. It saves every 10k plus a checkpoint at
`round(100000 * regional_train_images / original_train_images)`, matching the
expected number of image draws in a 100k full-scene run. If that checkpoint is
above 30k, the run extends to the next 10k boundary. The exact schedule is in
`region_ablation.json`. This matches expected image exposure, not per-surface
gradient exposure; the native SH warmup still depends on global steps.
Use `scripts/run_region_ablation.py --reference-steps N --iterations M ...`
followed by `bash scripts/run_prepared_reconstruction.sh MODEL` for a different
reference checkpoint. Preparation supports `--dry-run`.

An independent process on the same GPU evaluates every exported regional frame
at each checkpoint. It reports **ROI PSNR**, separately for original training
keyframes and their held-out complement; it does not claim 5,009-frame full-scene
evaluation. Pixel masks normalize L1/PSNR by supervised area; SSIM uses only
11x11 windows completely inside the mask. Outside-ROI pixels have zero loss
gradient, but the ordinary renderer can still receive contributions from any
Gaussian. The CUDA startup smoke tests this loss path before the real run.

Fixed random train/held-out GT/render/error previews run at step 1, every 500
steps, and the final step. A sampled `scripts/render.py --split both` gallery
also follows each evaluation checkpoint. Outside-ROI pixels are dimmed for
display; PSNR is computed from unmodified tensors inside the ROI.

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate 3dgs &&
cd /home/sequor_vfm/gaussian-splatting
python scripts/serve_training_previews.py \
  --model /data/3dgs_outputs/orange_4F/1_full/3dgs_finetuning/regions/wall_06663_v1/runs/crop_v1 \
  --port 8092
```

Open `http://211.172.247.108:8092` after the first preview is saved.

## Full-cloud control and direct comparison

After GPU 0 is free, train **the same images, original poses and masks** with the
entire original LiDAR cloud:

```bash
bash scripts/run_region_ablation.sh 0 "$REGION" full
```

This writes `$REGION/runs/full_v1`. Fixed XYZ and disabled growth/pruning remove
the global-extent dependence of those operations. KNN scale initialization is
recomputed on each input cloud; the crop halo reduces, but does not eliminate,
boundary initialization differences. All controls use the same other LRs.

```bash
python -u scripts/compare_region.py --region "$REGION" \
  --models "$REGION/runs/crop_v1" "$REGION/runs/full_v1" \
  --iterations 30000 --gpu 0 --output "$REGION/comparisons/crop_vs_full_30000"
```

You can add the earlier full-scene model to `--models` if it has a 30k checkpoint.
The comparator verifies matching calibration, original poses and RGB paths,
then renders every regional frame with identical ROI masks at full resolution.
It does not overwrite the model's ordinary full-frame metrics. For exposure
matching, pass one checkpoint per model and explicitly use `--different-steps`.
The checkpoint steps are shown in the gallery and summary.

## Interpret the outcomes

| Observation | Next hypothesis to test |
| --- | --- |
| Region becomes sharper only after far more updates per image | Optimization exposure / coverage imbalance. Compare loss and PSNR at the matched checkpoint before changing geometry. |
| Regional images + full LiDAR are sharp, but full-scene training is blurry | Additional views introduce conflicting pose/intrinsic/appearance supervision, or sampling gives this region too little attention. |
| Cropped LiDAR is sharp but the full-LiDAR control is blurry | Inspect distant Gaussian contributions and KNN boundary effects. This does not by itself prove ray-culling errors. |
| One contiguous local visit is sharp, all visits of the same region are blurry | Test relative pose drift, rolling shutter, exposure changes, moving content and non-Lambertian surfaces. |
| Even the region stays blurry on training views | Inspect projected Gaussian footprints, local coverage, calibration and residual maps; then try a single-view fitting diagnostic. |

Do not compare regional ROI PSNR numerically with the old full-image PSNR table.
Use the comparator to score both reconstructions on the same pixels.
