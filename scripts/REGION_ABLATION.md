# Regional blur ablation

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
Use **Find overlapping views**, then **Selected image review**. Bright image
pixels are supervised; dim ones are excluded. Exclude unwanted images by name
(e.g. displays with changing contents). **Export region** writes the dataset.
Changing the box/filter invalidates the selection until it is recalculated.
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

Exports:

- `images/`: symlinks to original RGB, without resampling.
- `sparse/0/`: subset cameras and original world-to-camera poses. Intrinsics are
  unchanged; SIMPLE_PINHOLE is expressed equivalently as PINHOLE. Feature tracks
  are omitted because initialization uses the LiDAR PLY.
- `input.ply`: **every original point** inside box + halo, in original order and
  coordinates; `original_lidar_indices.npy` maps to the input cloud rows.
- `keyframes.json`: geometric view selection intersected with the original
  training manifest. Other overlapping frames remain held-out.
- `masks/`, `overlays/`, `index.html`: loss/metric masks and review gallery.
- `selection.json`, `poses.json`: provenance, camera audit, counts, per-view
  mask fractions and exact box/filter settings.

RGB aspect mismatches fail export. Same-aspect pre-resized images are accepted
and their dimensions recorded. Principal-point offsets from the image center
are reported: native 3DGS's centered-FOV projection does not incorporate these
offsets. Large offsets need a separate camera-model correction experiment.

For reproducibility or a second temporal-window ablation, use explicit box
coordinates with `--center X Y Z --size SX SY SZ --export`. Optional
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
