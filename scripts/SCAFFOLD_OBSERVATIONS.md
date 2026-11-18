# Fixed Scaffold parents with per-view measured updates

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
