# Fixed Scaffold parents with per-view updates

For the red LiDAR points in the 3D sparse-observation viewer, run:

```bash
bash scripts/run_orange4f_scaffold.sh 4 indexed
```

The indexed run takes the viewer's full per-image PLY row list. It uses explicit row indices when present and otherwise the same unique XYZ matches for stored COLMAP tracks. Set `SCAFFOLD_INDEXED_PLY` or `SCAFFOLD_OBSERVATIONS` only if you used the viewer's `--ply` or `--manifest` overrides. The launcher voxelizes the viewed cloud at 2 cm by default and freezes the parent positions and population. For each training image, a parent qualifies if any of its LiDAR rows is red. All normally emitted Gaussians composite the image. Only eligible rendered children contribute gradients to their local parameters and the shared decoders.

The stride-3 run trains for 30,000 iterations with full-resolution RGB and AA off. `observations/metadata.json` records the source, red-row count and eligible-parent count for every training image. The terminal reports updated parents and eligible children. To view train and held-out RGB previews after step 1, run:

```bash
python scripts/serve_training_previews.py --model "${ORANGE4F_OUTPUT_ROOT:-/data/3dgs_outputs/orange4f/3dgs_finetuning}/stride3_noaa_v1/scaffold_indexed_anchors_v1" --host 0.0.0.0 --port 8096
```

## Indexed-only rendering and hit opacity

Run either variant on an available GPU. The two commands use distinct output directories.

```bash
bash scripts/run_orange4f_scaffold.sh 5 indexed-opacity
bash scripts/run_orange4f_scaffold.sh 6 indexed-only
```

Both runs intersect each training camera’s indexed parent set with its frustum before decoding any children. The opacity variant also projects the original red points using the trainer’s centered pinhole convention and stores up to 8,192 distinct in-image pixels per training view. A second differentiable render measures their accumulated opacity with fixed footprints. The auxiliary loss is `0.05 * mean(relu(0.9 - alpha_at_hit)^2)`. Geometry and RGB still learn from the image objective. Empty hit sets skip the auxiliary pass. The target and weight are starting values for the comparison. The term measures coverage along a ray and does not supervise its depth.

Training previews display GT, full-scene RGB and error, and an additional indexed-only RGB with its PSNR. Held-out previews and periodic evaluation always render the full model. Compare the same held-out views and training step across both runs. The training log records mean opacity at supervised pixels (`hit_alpha`), sample count and the unweighted loss.

```bash
python scripts/serve_training_previews.py --model "${ORANGE4F_OUTPUT_ROOT:-/data/3dgs_outputs/orange4f/3dgs_finetuning}/stride3_noaa_v1/scaffold_indexed-opacity_anchors_v1" --host 0.0.0.0 --port 8097
```

## Measured-return and mesh variants

Inspect the capture with `python scripts/inspect_lidar_observations.py`. Use `--source /path/to/raw/capture` when the recording lives outside the prepared COLMAP scene. This lists candidate files, PLY properties, NPZ keys and recognized image associations. Acquisition timestamps require a sensor-specific synchronization or explicit image association.

Run `bash scripts/run_orange4f_scaffold.sh 4 sensed` and compare against `bash scripts/run_orange4f_scaffold.sh 5 fixed-static`. Both use 2 cm fixed voxel centers, a constant parent population, learned offsets, the same shared MLP architecture and identical train/held-out views. Both composite every normally emitted Gaussian. In the sensed run, only eligible children contribute RGB and scale gradients to the shared networks. Local Adam parameters and both moment buffers update only for eligible parents with a rendered child. Shared decoder changes can affect future predictions for any parent. Empty return sets skip optimization for that image. Missing associations are reported before starting training.

Set `SCAFFOLD_OBSERVATIONS=/path/to/manifest.json` or place one of the following in the source directory.

* `lidar_observations.json` identifies measured returns by zero-based vertex rows of the input PLY. Each NPY is a one-dimensional integer array. Names must match the dataset's image names, including extensions.
* `lidar_scans.json` uses the PGSR `scanner_xyz_v1` format with an additional `image` or `images` field on each associated scan. XYZ returns must be in metres in the listed scanner frame. Multiple scans assigned to one image are combined. Transformed returns match occupied anchor voxels using the same rounding as initialization. Unmatched returns receive no parent. The camera frustum still applies when the scanner has a wider field of view.

Example row manifest:

```json
{"format":"lidar_rows_v1","point_cloud_sha256":"<sha256 of the input PLY>","frames":{"09882.jpg":"rows/09882.npy"}}
```

Example scan entry:

```json
{"format":"scanner_xyz_v1","scans":[{"images":["09882.jpg"],"xyz":"scans/0012.npy","scanner_to_world":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}]}
```

The run saves per-view parent counts and aggregate coverage in `observations/metadata.json`, actual eligible render counts in `scaffold_training.jsonl`, and RGB comparisons in `debug_previews/index.html`. Evaluate on the same held-out views at 10k, 20k and 30k. A low ever-sensed fraction indicates that many parents receive no local updates.

For a separate estimated-visibility comparison, set `SCAFFOLD_OBSERVATIONS` to the existing mesh cache’s `visibility/metadata.json` and use the `mesh-visible` variant. This reads the cached `visible_ids` and verifies each training camera signature and the cloud row identity. Its eligibility derives from first-hit mesh agreement.
