# Inspect actual occlusion decisions

The inspector loads a saved **native 3DGS** checkpoint, its original camera
calibration and training resolution, and its saved LiDAR or DA3 culling settings.
It uses the exact `occlusion_keep` returned by `gaussian_renderer.render`.
It does not optimize weights, modify checkpoints, or interrupt training jobs.

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

- `index.html`, six full-resolution PNGs and `summary.json`.
- `labels.npy`: one uint8 label per original checkpoint PLY row (0 other,
  1 active, 2 culled); no sampling. `summary.json` records the checkpoint SHA256.
- `occlusion_tiles.npy`: the actual bounds used for the selected camera.
- **Export all green + red centers to PLY** writes `active_and_culled.ply`,
  including labels and original `gaussian_id` values. It exports every green/red
  center even if a browser display cap is enabled.

Append `--export-only` to the command to save a report, exact labels and the full
colored center PLY without opening a viewer. `--output` overrides the report root.
