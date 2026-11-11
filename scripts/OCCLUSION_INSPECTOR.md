# Inspect actual occlusion decisions

The inspector loads a saved **native 3DGS** checkpoint, its original camera
calibration and training resolution, and its saved LiDAR or DA3 culling settings.
It uses the exact `occlusion_keep` returned by `gaussian_renderer.render`.
It does not optimize weights, modify checkpoints, or interrupt training jobs.

## Compare LiDAR patches with LiDAR-calibrated dense depth

The old LiDAR-only run uses isolated planar patches, not DA3 depth. Its blue
points visualize only those accepted patches' first-hit depths; green points
are retained raster candidates, not physically verified visible points. A small
patch-retention fraction does not establish adequate image coverage. One unknown
pixel invalidates its 16x16 tile, and a Gaussian is kept if its conservative
footprint includes any such unknown tile. Sparse wall coverage can therefore
make culling ineffective even when the gradient masking mechanism is correct.

To check this on identical saved weights, stop the previous inspector with
Ctrl+C and run (use an available GPU):

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate 3dgs
cd /home/sequor_vfm/gaussian-splatting
git pull --ff-only origin experiment/my-3dgs-changes
bash scripts/inspect_da3_occlusion.sh 5 \
  /data/3dgs_outputs/orange_4F/1_full/3dgs_stride3_aa_lidar_occlusion
```

The script explicitly activates `da3-depth`, prepares three distributed training
views with their usual neighboring context, then activates `3dgs` for the viewer.
The depth prediction is calibrated to the checkpoint's exact initial LiDAR PLY,
using nearest projected anchors, held-out calibration residuals, confidence,
depth-boundary and multiview consistency checks. These checks reduce visibility
errors in the merged cloud; they do not replace per-frame LiDAR visibility.
`DA3_DEPTH_DIR` overrides the cache directory. An optional third positional
argument selects one exact training image instead of three distributed views.
The current default cache is `da3_lidar_depth_stride3_v2`. It validates LiDAR
alignment locally instead of erasing the whole frame when one global alignment
criterion fails. Old rejected caches are not reused by this preparation. For the
reported wall view, append `07519.jpg` as the third argument to the wrapper.

Open `http://211.172.247.108:8082` from the remote client. The wrapper binds
`0.0.0.0`; `OCCLUSION_INSPECTOR_PORT` changes the port. It does not start training.
Inspect-only caches remain incomplete and are rejected by training. Running
the ordinary preparation without `--inspect-views`/`--inspect-image` later resumes
the same cache and processes all keyframes.

If the calibrated cache already exists, the direct command is:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate 3dgs
cd /home/sequor_vfm/gaussian-splatting
python -u scripts/inspect_occlusion.py \
  --model /data/3dgs_outputs/orange_4F/1_full/3dgs_stride3_aa_lidar_occlusion \
  --depth-dir /data/3dgs_outputs/orange_4F/1_full/da3_lidar_depth_stride3_v2 \
  --gpu 5 --host 0.0.0.0 --port 8082 --check-gradients
```

With `--depth-dir`, the inspector computes both references on the **same saved
checkpoint and exact camera**. The default displayed reference is DA3; the
dropdown switches between it and the saved LiDAR patches without reloading
weights. Reports identify DA3 as a diagnostic override, not a change to the
checkpoint's training history. Image gradients are checked for the selected
reference; choose another reference and click **Check image gradients** to test
that source too. Culling decisions and coverage are computed for both sources.

The comparison table reports trusted pixel coverage, usable tile coverage, and
the fraction of OFF raster candidates culled. Retained candidates are separated
into missing depth coverage, overlap with depth/margin, and those outside the
conservative test. **Show reference surface only** hides Gaussian markers.
Additional 2D panels show raw depth, trusted coverage over GT, usable tiles, and
LiDAR anchor agreement. Version-2 caches also show the separate confidence and
local-calibration stages. Anchor agreement is measured against the depth prediction
before masking, so a failed calibration no longer hides every residual as gold.
The status explicitly says `EMPTY REFERENCE` when culling has no usable depths.
The coverage image uses green for usable tiles, blue for
accepted pixels lost to whole-tile checks, and black for unknown/rejected pixels.
This panel's colors are separate from the 3D Gaussian legend.

The depth lookup is now `exact_tile_rectangle_v2` (shown in the viewer and JSON).
The previous strided mip lookup could include unknown tiles outside a Gaussian's
tested rectangle, incorrectly retaining it even when that rectangle was fully
covered by wall depth. The replacement returns the exact maximum over that tile
rectangle. Real holes inside it still retain the Gaussian. Existing depth caches
are compatible; they do not need rebuilding for this lookup correction. Restart
the inspector after pulling the fix. A running training process retains its
loaded code; inspecting an old checkpoint uses the current culling implementation
and does not reproduce the historical training mask.

The DA3 inspection wrapper runs `scripts/smoke_da3_occlusion.py` in `3dgs` before
opening the viewer. This tests a local wall surrounded by unknown depth, confirms
zero hidden gradients in the native CUDA rasterizer, and verifies that a hole
inside the actual footprint preserves the potentially visible Gaussian.

## Inspect only the reference saved with a run

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate 3dgs
cd /home/sequor_vfm/gaussian-splatting
git pull --ff-only origin experiment/my-3dgs-changes
python -m pip install 'viser==1.0.30'
python -u scripts/inspect_occlusion.py \
  --model /data/3dgs_outputs/orange_4F/1_full/3dgs_stride3_aa_lidar_occlusion \
  --gpu 5 --port 8082 --check-gradients
```

Use a free GPU for the viewer, especially for the backward check. GPU 5 is an
example; GPU 2 may already be training the tangent ablation. Open
`http://127.0.0.1:8082` on the GPU host. From another computer, forward port 8082
over SSH, then open that same URL locally. Bind address defaults to localhost.

The latest saved PLY is selected by default. Use `--iteration 7000` to inspect an
early checkpoint, or `--image 'exact_registered_name.jpg'` to choose an image.
The dropdown defaults to training images. LiDAR supports `--split all`; DA3 only
supports the training images for which its depth cache was prepared. To inspect a
DA3 run, simply change `--model`; the saved depth directory is detected.

## What the layers mean

| Layer | Exact definition |
| --- | --- |
| Green, active | Kept by external depth culling and native rasterizer radius > 0 |
| Red, depth-culled | `occlusion_keep == False` |
| Gray, other | Kept, but no raster footprint for this camera |
| Blue surface | Raw first-hit LiDAR or trusted calibrated DA3 depth, backprojected with its own intrinsics |

All Gaussian centers in a selected layer are displayed by default. Gray is
initially hidden and loaded only when requested. Set the display cap to a positive
number if browser bandwidth/memory is limiting; displayed/total counts remain
explicit. Masks, statistics and PLY exports always include the full populations.
Depth surface points are sampled every three pixels for display.

Green is **not proof of actual pixel contribution**: native alpha compositing can
still hide a kept Gaussian. Red is a decision of our conservative depth test,
not independently measured ground-truth visibility. Markers deliberately ignore
learned opacity. Hide green/blue and orbit to inspect the red population; return
to the image camera to check projection alignment.

Shift-click near a colored point (nearest displayed center to the click ray), or enter its
zero-based checkpoint row ID. The yellow ellipsoid rings show its configured sigma scale
and orientation. The details show camera-z, nearest tested support depth, the
conservative surface bound, pixel radius, tile footprint, margin, and the reason
for retention/culling. Unknown footprint coverage keeps a Gaussian. A center
behind a wall can also remain if its large support overlaps the wall/margin.

## Verify image and gradient effects

The viewer shows actual RGB, culling OFF/ON, their absolute difference (×10), a
projected center overlay, and conservative occlusion tile depths (black = unknown).
The projected center overlay intentionally has no z-buffer; it is a diagnostic,
not a rendered segmentation image. PSNR uses the same image/pose in both cases.

The optional **Check image gradients** action runs the actual native AA forward
and backward passes, using L1+SSIM against that image. It checks xyz, opacity,
scale, rotation, DC color and higher SH coefficients for every culled Gaussian.
It then repeats with culling OFF and reports how many of those Gaussians regain
nonzero image gradients. A valid masking result has exactly zero culled image
gradients with culling ON; when OFF also yields zero, native compositing or this
image loss may already suppress those points. All-zero probes and views with no
culled Gaussians are marked inconclusive. No Adam update is performed. This is a
saved-state image-gradient check, not a measurement of a live training process
or of separate depth/surface regularizers.

The inspector reports both **culled / all Gaussians** and **culled / raster
candidates without culling**. A small percentage of a large scene can still be a
substantial fraction of the Gaussians intersecting the selected view.

## Saved results

Each inspected view writes into
`MODEL/occlusion_diagnostics/iteration_N/IMAGE_HASH/`:

- Separate `saved/` and, when supplied, `da3/` subdirectories. Each contains
  `index.html`, full-resolution comparison/coverage PNGs and `summary.json`.
  Neither source overwrites the other's report.
- `labels.npy`: one uint8 label per original checkpoint PLY row (0 other,
  1 active, 2 culled); no sampling. `summary.json` records the checkpoint SHA256.
- `occlusion_tiles.npy`: the actual bounds used for the selected camera.
- `reference.npz`: numerical prediction, masks, camera matrices and projected
  LiDAR anchors; v2 also includes raw DA3 prediction and confidence. This supports
  reproducing calibration failures instead of inferring depth from a PNG.
- A shared `occlusion_debug.zip` one directory above `saved/` and `da3/` contains
  both reference reports, PNGs, tile bounds and numeric reference snapshots.
  The viewer and terminal print its path. Large Gaussian PLY/label files are excluded.
- **Export all green + red centers to PLY** writes `active_and_culled.ply`,
  including labels and original `gaussian_id` values. It exports every green/red
  center even if a browser display cap is enabled.

Append `--export-only` to the command to save a report, exact labels and the full
colored center PLY without opening a viewer. `--output` overrides the report root.
