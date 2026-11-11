# Optional DA3 scene fusion and per-view LiDAR visibility

**The default occluder is now the full LiDAR mesh described in
[RAY_OCCLUSION.md](RAY_OCCLUSION.md).** The DA3-only path below is retained for
comparison and optional gap supplementation. Its LiDAR support filter does not
reconstruct missing measured walls; low trusted DA3 coverage can therefore leave
large holes. The LiDAR-primary builder does not use these acceptance masks.

For per-pixel CUDA clipping and a multi-frame preview before full preparation,
use [RAY_OCCLUSION.md](RAY_OCCLUSION.md). The older whole-Gaussian culler remains
available for comparison.

This path addresses missing wall coverage in per-image occluders. It builds one
fixed mesh from all selected training views, then projects that same scene into
every registered camera. It is independent of Proxy-GS and learned Gaussian
opacity. It does not claim that fusing incomplete depths guarantees complete
surface coverage: unseen or unsupported surfaces stay unknown and are reported.

## Construction and visibility

1. Finish the **complete DA3 trajectory-group training cache** (format v2),
   including shared group metric alignment, local LiDAR donor checks,
   confidence/boundary checks, and agreement with at least two other views.
   No held-out RGB images enter depth prediction or surface fusion.
2. Fuse trusted camera-z depths in world coordinates with a scalable TSDF
   (5 cm voxels, 15 cm truncation by default). Integrate only trusted pixels;
   other views can supply surfaces missing in a particular image.
3. Retain triangles only when all vertices are within 20 cm of the registered
   LiDAR cloud. This uses the full finite cloud, without the sparse planar-patch
   selection used by the original LiDAR occluder. The distance is a geometric
   support check, not a claim of exact surface accuracy. There is no explicit
   hole filling, Poisson closure, or convex hull; small details remain limited
   by voxel resolution and depth accuracy.
4. For **all registered frames** (default, including unselected frames), render
   first-hit depth from this mesh and cache conservative Gaussian-culling tiles.
5. On CUDA, project every original LiDAR row and compare its camera z with the
   nearest pixel's first-hit mesh depth. `--visibility-backend exact` retains
   CPU BVH rays through the exact point projections as a reference. Depth is
   **camera z**, not Euclidean range. Save original PLY row IDs in four disjoint sets:

| Array | Meaning | Viewer color |
| --- | --- | --- |
| `visible_ids` | Point agrees with the first surface within tolerance | Green |
| `occluded_ids` | Point is farther than the first surface plus tolerance | Red |
| `unknown_ids` | Ray has no valid mesh intersection | Gray |
| `in_front_ids` | Point lies before the mesh; investigate geometry disagreement | Orange |

Tolerance is 5 cm plus 10% of surface depth by default. Out-of-view and invalid
rows are omitted and counted separately. IDs index the original input PLY,
including its original ordering; they are not indices into a filtered cloud.
The blue transparent geometry is the fixed occluding mesh.

The cached point sets support inspection and visible-only LiDAR supervision.
They are **not reused as permanent Gaussian IDs** after movement or densification.
Native training tests the current Gaussian support against mesh depth on each
iteration and freezes noncontributing optimizer rows/momentum. Novel and held-out views
raycast the same mesh; no new DA3 inference or opacity warmup is required for
the reference. The legacy `whole` backend retains partial/unknown footprints;
the `ray` backend blocks each covered pixel fragment independently.

CPU preprocessing uses Open3D 0.19 scalable TSDF fusion. CUDA handles per-camera
mesh depth with nvdiffrast, point projection/depth lookup, and the custom
forward/backward splat cutoff when `--occlusion-backend ray` is selected.
The optional exact point-ray reference still uses a CPU triangle BVH. APIs:
[TSDF integration](https://www.open3d.org/docs/release/tutorial/pipelines/rgbd_integration.html),
[ray casting](https://www.open3d.org/docs/release/tutorial/geometry/ray_casting.html).

## Prepare the scene and open the LiDAR viewer

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate 3dgs
cd /home/sequor_vfm/gaussian-splatting

bash scripts/prepare_scene_occlusion.sh 5 \
  /data/3dgs_outputs/orange_4F/1_full/3dgs_stride3_aa_lidar_occlusion \
  07519.jpg
```

The wrapper uses [trajectory-spanning groups](DA3_TRAJECTORY.md), with a 40-image
budget, eight shared references, two local companions per target, and processing
resolution 504. Preview a failing frame using `inspect_da3_trajectory.sh` first.
The wrapper activates `da3-depth` for complete depth preparation and `3dgs` for
fusion, visibility, and viewing. Existing matching DA3 frames are reused. It
runs the new mesh CUDA smoke test before scene fusion. All registered frames are
preprocessed, then the original-LiDAR viewer opens on port **8083**. This stage
does not start training. `SCENE_VISIBILITY_PORT`, `SCENE_OCCLUSION_DIR`,
`DA3_TRAJECTORY_DEPTH_DIR`, and `DA3_CONDA_SH` override defaults. If only the viewer was
stopped, run `scripts/inspect_scene_visibility.py` directly to reopen it.

Default scene output:
`/data/3dgs_outputs/orange_4F/1_full/scene_occlusion_stride3_trajectory40_v1`

- `scene.json`, `mesh.npz`, `mesh.ply`: immutable scene input identity and surface.
- `occlusion_depth_cache/<scene-hash>/`: native-camera first-hit depth and tiles.
- `visibility/frames/*.npz`: four point-ID sets, camera matrices and counts.
- `visibility/index.html`, `summary.json`: coverage and visibility for every view.
- `visibility/previews/`: actual RGB, mesh depth and coverage for sampled views,
  including `07519.jpg` when present.

Independent commands for inspection/resuming:

```bash
python -u scripts/prepare_scene_occlusion.py fuse \
  --depth-dir /data/3dgs_outputs/orange_4F/1_full/da3_lidar_depth_stride3_trajectory40_v1 \
  --point-cloud /data/3dgs_outputs/orange_4F/1_full/3dgs_stride3_aa_lidar_occlusion/input.ply \
  --output /data/3dgs_outputs/orange_4F/1_full/scene_occlusion_stride3_trajectory40_v1

python -u scripts/prepare_scene_occlusion.py views \
  --scene /data/3dgs_outputs/orange_4F/1_full/scene_occlusion_stride3_trajectory40_v1/scene.json \
  --point-cloud /data/3dgs_outputs/orange_4F/1_full/3dgs_stride3_aa_lidar_occlusion/input.ply \
  --gpu 5 --image 07519.jpg
```

Omit `--image` to finish all registered views. `--device cpu` provides the same
exact BVH semantics without CUDA, for testing or CPU preprocessing. A partial
visibility run remains marked incomplete.

## Inspect actual optimized Gaussians against the scene mesh

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate 3dgs
cd /home/sequor_vfm/gaussian-splatting
python -u scripts/inspect_occlusion.py \
  --model /data/3dgs_outputs/orange_4F/1_full/3dgs_stride3_aa_lidar_occlusion \
  --scene-geometry /data/3dgs_outputs/orange_4F/1_full/scene_occlusion_stride3_trajectory40_v1/scene.json \
  --image 07519.jpg --gpu 5 --host 0.0.0.0 --port 8082 --check-gradients
```

Select **Fixed whole-scene mesh (diagnostic override)**. This compares the same
checkpoint/camera with its old saved reference. Counts and the gradient check
measure actual Gaussian culling; the standalone LiDAR viewer measures original
point visibility. Inspect both, including unknown coverage and doorways.

## Training after inspecting the preprocessing

`scripts/run_occlusion.py` accepts `--scene-geometry scene.json`. It requires
complete matching per-view preprocessing, uses native 3DGS + AA, and retains
all-frame PSNR, train/eval actual-image comparisons, and both trajectory videos.
Pass the existing AA baseline explicitly for video comparisons:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate 3dgs
cd /home/sequor_vfm/gaussian-splatting
scene_root=/data/3dgs_outputs/orange_4F/1_full
scene_source=/data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full
scene_base=$(python -c 'from scripts.render import resolve_models; import sys; print(resolve_models(None, sys.argv[1], ["sharp_keyframes_stride3_aa"])[0])' "$scene_root")
python -u scripts/run_occlusion.py \
  --source "$scene_source" --baseline "$scene_base" --keyframes "$scene_base/keyframes.json" \
  --model "$scene_root/3dgs_stride3_aa_scene_occlusion_warm7k_lr50" \
  --scene-geometry "$scene_root/scene_occlusion_stride3_trajectory40_v1/scene.json" \
  --position-warmup-steps 7000 --position-ramp-steps 2000 --position-lr-scale .02 \
  --gpu 5
```

This keeps the requested fixed-position warmup and 50x smaller position LR.
Both culling-on and culling-off all-frame scores are emitted. Actual scene
coverage and reconstruction improvements must be measured on the server;
synthetic geometry tests alone cannot establish them.
