# Full LiDAR surfaces and CUDA ray clipping

The primary occluder is now a connected triangle mesh built directly from the
**entire initialization LiDAR cloud**. DA3 confidence, local calibration, and
multiview acceptance masks cannot delete measured walls. The earlier TSDF
preview used only accepted DA3 pixels, with LiDAR as a support filter; displaying
the gray LiDAR cloud did not make those points occlude anything.

`prepare_lidar_scene.py` selects measured representatives at 5 cm spacing, then
meshes overlapping spatial blocks using two-sided ball pivoting. Vertices remain
in their original coordinates. Triangle edges are capped at 20 cm; local tangent
agreement and measured support at midpoints/centroids reject unsupported bridges.
Both pivot orientations are used so a nearby parallel wall does not prevent the
front sheet from being reconstructed. Duplicate triangles are removed. Doorways
are not closed by a convex hull, Poisson completion, or global voxel dilation.
Features below the selected spatial scale and sparsely measured surfaces can
still be missing; image coverage is measured separately from mesh vertex coverage.

Meshing is one-time **CPU preprocessing** with bounded spatial blocks. Fixed-mesh
depth rasterization, point projection/depth comparisons, and per-pixel Gaussian
forward/backward clipping are **CUDA**. This remains native 3DGS + AA, independent
of Proxy-GS. The mesh also supplies depth for novel camera poses.

## Inspect the failing wall

```bash
source /path/to/anaconda3/etc/profile.d/conda.sh
conda activate 3dgs
cd /path/to/gaussian-splatting
git pull --ff-only
bash scripts/inspect_ray_surface.sh 5 \
  /path/to/native_checkpoint \
  FRAME.jpg
```

Replace the placeholder paths and `FRAME.jpg` with the checkpoint and camera
being inspected. The wrapper activates `3dgs`, builds the CUDA extension, runs native/ray AA
forward/backward checks, and runs `smoke_occlusion.py --lidar-mesh --backend ray`.
The latter reconstructs an actual measured-point fixture before testing CUDA
depth, a novel camera, rendering, and zero gradients for a hidden Gaussian.
Next it meshes the full cloud once, caches original LiDAR row IDs for the selected
view, and opens the viewer at **http://HOST:8085**. It does not launch
training or run DA3. Existing checkpoints and DA3 caches are preserved.

Default output: `MODEL_PARENT/lidar_scene_surface_v1`.
This global mesh is reused when selecting another camera. `RAY_SURFACE_OUTPUT`
overrides that directory; use a new directory when changing meshing settings.
`RAY_SKIP_INSTALL=1` skips dependency rebuilding on subsequent launches.
`RAY_INSPECTOR_PORT` overrides 8085. `DA3_CONDA_SH` overrides the conda profile path.

| Viewer layer | Color | Meaning |
|---|---|---|
| Outside reference frustum | Gray | Scene context outside the selected camera |
| All scene points (optional) | Gray | Raw-cloud overlay; may cover colored labels |
| All frustum points (optional) | Cyan | Full in-view point set |
| Culled point centers | Red | Behind first-hit depth plus the culling margin |
| Near surface | Yellow | Within 3 cm of the first hit, independent of culling tolerance |
| Rear tolerance band | Orange | Behind the narrow band but not beyond the cutoff |
| Unknown depth | Purple | In-frustum ray with no mesh hit |
| In front of mesh | Green | Inspect missing foreground geometry or disagreement |
| Primary occluding surface | Blue mesh | Connected LiDAR triangles |
| Actual Gaussian fragments | Separate toggles | Fully blocked, partly blocked, contributing |

Default point classes are disjoint; they collectively show the whole scene.
No display subsampling occurs unless `--max-points` is provided. The small RGB
plane is a camera visualization at an arbitrary distance, not measured wall depth.
Point-center labels and actual Gaussian fragment flags are distinct.

The **foreground audit** projects the nearest measured LiDAR point into each
native camera pixel and compares it with mesh first-hit depth. Its RGB overlay
uses green for agreement, red for a mesh hit behind measured foreground, purple
for missing mesh depth, and cyan for a mesh hit in front of the measured sample.
These are sparse samples, not ground-truth dense depth: mixed pixels, outliers,
and calibration errors can cause disagreements. The sparse projection is never
used as the occluder. A passing gradient check alone does not prove wall coverage.

`ray_inspection/<image-id>/` saves GT, culling OFF/ON renders, difference and depth
images, the foreground overlay, point-ID sets, `depth_audit.npz`, and HTML/JSON.
`unmeshed_lidar.npz` lists original PLY rows for representatives without triangles.
Culling uses a 5 cm absolute margin and **zero relative margin** by default; the
former 10% depth tolerance is not inherited by the new mesh. All distances are
in the input scene's coordinate units (meters for the registered metric LiDAR).

## Cache every registered view

Set `RAY_PRECOMPUTE_ALL=1` when running the wrapper, or finish the resumable cache
separately after inspecting the selected wall:

```bash
source /path/to/anaconda3/etc/profile.d/conda.sh
conda activate 3dgs
cd /path/to/gaussian-splatting
ray_root=/path/to/model_parent
python -u scripts/prepare_scene_occlusion.py views \
  --scene "$ray_root/lidar_scene_surface_v1/scene.json" \
  --point-cloud /path/to/native_checkpoint/input.ply \
  --split all --resolution -1 --device cuda --visibility-backend cuda --gpu 5
```

This saves visibility for all registered frames, including unselected frames,
using stable original PLY row IDs. A single-view cache remains explicitly partial.
The mesh is complete independently of that cache; training requires the full
visibility cache and matching training keyframes/camera signatures.

## Optional DA3 supplement

`prepare_lidar_scene.py --supplement-scene PATH/scene.json` accepts a complete,
validated DA3-only TSDF scene made by the existing fusion pipeline. The source,
initialization cloud, and training keyframes must match. Diagnostic partial
meshes are rejected. The wrapper accepts `RAY_DA3_SUPPLEMENT_SCENE` for this option.

The two meshes remain separate. A finite LiDAR depth always wins; the DA3 depth
is used only on rays where the LiDAR mesh has no hit. This priority is shared by
CUDA depth, exact CPU point-ray checks, training, and novel views. DA3 candidate
geometry is an optional orange mesh layer in the viewer, off by default.

## Training after the coverage check

```bash
source /path/to/anaconda3/etc/profile.d/conda.sh
conda activate 3dgs
cd /path/to/gaussian-splatting
ray_root=/path/to/model_parent
ray_base=$(python -c 'from scripts.render import resolve_models; import sys; print(resolve_models(None, sys.argv[1], ["sharp_keyframes_stride3_aa"])[0])' "$ray_root")
python -u scripts/run_occlusion.py \
  --source /path/to/colmap_scene \
  --model "$ray_root/3dgs_stride3_aa_lidar_mesh_ray_warm7k_lr50" \
  --baseline "$ray_base" --keyframes "$ray_base/keyframes.json" \
  --scene-geometry "$ray_root/lidar_scene_surface_v1/scene.json" \
  --occlusion-backend ray --position-warmup-steps 7000 --position-ramp-steps 2000 \
  --position-lr-scale .02 --gpu 5
```

The launcher retains train/eval GT comparisons, all-frame PSNR, and actual/novel
trajectory videos. Current Gaussian fragments are clipped every iteration;
precomputed LiDAR IDs are not reused as Gaussian IDs after movement/densification.
Noncontributing Gaussian optimizer rows and momentum remain frozen for that view.
