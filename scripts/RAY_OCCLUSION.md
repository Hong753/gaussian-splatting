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
bash scripts/inspect_ray_surface.sh 0 \
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
  --split all --resolution -1 --device cuda --visibility-backend cuda --gpu 0
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

## Training with mesh-free inference

```bash
source /path/to/anaconda3/etc/profile.d/conda.sh
conda activate 3dgs
cd /path/to/gaussian-splatting
ray_root=/path/to/model_parent
ray_base=$(python -c 'from scripts.render import resolve_models; import sys; print(resolve_models(None, sys.argv[1], ["sharp_keyframes_stride3_aa"])[0])' "$ray_root")
python -u scripts/run_occlusion.py \
  --source /path/to/colmap_scene \
  --model "$ray_root/3dgs_stride3_aa_lidar_mesh_trainonly_warm7k_lr50" \
  --baseline "$ray_base" --keyframes "$ray_base/keyframes.json" \
  --scene-geometry "$ray_root/lidar_scene_surface_v1/scene.json" \
  --prepare-visibility --inference-occlusion off --occlusion-backend ray \
  --position-warmup-steps 7000 --position-ramp-steps 2000 \
  --position-lr-scale .02 --gpu 0
```

The launcher retains train/eval GT comparisons, all-frame PSNR, and actual/novel
trajectory videos. Current Gaussian fragments are clipped every iteration;
precomputed LiDAR IDs are not reused as Gaussian IDs after movement/densification.
Noncontributing Gaussian optimizer rows and momentum remain frozen for that view.

`--prepare-visibility` finishes/resumes the all-frame CUDA visibility cache before
training; a partial diagnostic cache is not silently treated as complete.
GPU 0 is the default for the occlusion training and inspection workflow.

`--inference-occlusion off` saves `occlusion_inference_mode=off` in `cfg_args`,
while `occlusion_mode=mesh` remains enabled during optimization. Checkpoint
validation, image comparisons, all-frame PSNR, and both videos use native 3DGS
with AA and no mesh access. Training-only culling does not force learned
foreground opacity to become opaque; mesh-free results measure whether the
learned model retains the improvement without the external gate.

The default mesh-free PSNR is written to `psnr_all_frames/iteration_30000/`
with training mode and inference policy recorded in `summary.json`. This differs
from an explicit off override on an older model, which uses `occlusion_off/`.
`--occlusion on` remains available for a separate diagnostic render; it requires
the mesh and writes a distinct result variant. `--evaluate-only` recovers the
saved inference policy and does not require visibility preprocessing when off.

## Tangent densification on the same mesh

The plain training command above uses ordinary adaptive densification. Add
`--view-tangent` to gate growth with the same mesh's **raw first-hit depths**:
visible, near-surface parents accumulate image gradients; high-gradient parents
clone or split inside a locally supported tangent disk. New centers preserve the
parent's signed distance to that plane. Holes, depth discontinuities, unsupported
regions, and occluded parents do not seed growth. This changes the growth rule;
the ordinary reduced-LR position/covariance optimizer remains in place.

The dedicated launcher defaults to this mesh run on **GPU 1**:

```bash
source /path/to/anaconda3/etc/profile.d/conda.sh
conda activate 3dgs
cd /path/to/gaussian-splatting
DA3_OUTPUT_ROOT=/path/to/model_parent \
DA3_SCENE_ROOT=/path/to/colmap_scene \
DA3_BASELINE=/path/to/3dgs_aa_baseline \
VIEW_TANGENT_SCENE=/path/to/model_parent/lidar_scene_surface_v1/scene.json \
bash scripts/run_view_tangent.sh 1 mesh
```

It reuses the existing depth/visibility files, fills missing visibility frames,
and creates a separate `3dgs_stride3_aa_lidar_mesh_trainonly_viewtangent_warm7k_lr50`
model. Defaults: 30k iterations, fixed positions for 7k, 2k LR ramp, 50x smaller
position LR, growth at steps 7,600 through 21,900, at most 50k growing parents per
event and 12M total Gaussians. AA stays enabled. Validation, all-frame PSNR, GT
comparisons, and both videos render without external occlusion surfaces.

`view_tangent_cuda_smoke.json` records the actual GPU preflight (mesh rasterization,
blocked gradients, tangent clone/split, optimizer update after growth, and ordinary
inference). During training, `view_tangent_densification.jsonl` records supported,
selected, cloned and split parent counts, plus the measured maximum child normal
displacement and tangent-radius fraction. The displacement should be near floating
point zero and the fraction at most 0.9. These check new child placement, not later
optimizer motion. Disk depth caches persist; depth and derived surface-frame GPU
caches each retain at most four views. Surface frames are recomputed from disk
depth when evicted, rather than keeping all scene views in memory.

## Inspect each saved camera after densification

The inspector loads every Gaussian from the selected checkpoint and recomputes
CUDA fragment flags for each camera, including cloned and split children. It
renders both culling ON and OFF alongside GT even when the saved inference policy
is OFF. This diagnostic does not change the training or saved inference policy.

```bash
conda activate 3dgs
python -u scripts/inspect_ray_occlusion.py \
  --scene /path/to/model_parent/lidar_scene_surface_v1/scene.json \
  --point-cloud /path/to/model/input.ply --model /path/to/model \
  --iteration -1 --image 06663.jpg --point-set Both --mesh-style hidden \
  --gpu 0 --host 0.0.0.0 --port 8086
```

With a checkpoint, the default 3D colors are actual CUDA states: red = fully
blocked, orange = partly blocked and contributing, green = contributing with no
blocked fragments, gray = no encountered contribution. Gray does not imply
geometric visibility or occlusion. The four groups are disjoint, so green cannot
hide the partly blocked orange group. These are Gaussian centers, not rendered
ellipsoids. **Show LiDAR** and **Show Gaussians** are independent toggles, both on
by default with a checkpoint. LiDAR has its own point size and either original
RGB (neutral blue-gray if absent) or depth-class colors. The original PLY rows
remain the identities used by the classification exports.

The blue mesh is initially hidden in the 3D display. **Surface display (visual
only)** can show its wireframe or filled triangles without changing CUDA culling.
The LiDAR/Gaussian clouds can also overlap: **Show contributing Gaussians only**
hides LiDAR, both mesh overlays and noncontributing Gaussian groups, leaving the
green and orange centers. Display controls reuse point buffers instead of
rerendering or reclassifying. The status reports actual post-culling contribution
counts; missing green markers alone do not establish excessive culling. Inspect
the GT and culling ON/OFF images alongside those counts to check the effect.

Use the Camera dropdown or Previous/Next frame buttons to change the reference
view. Merely orbiting the 3D scene does not change the reference camera used for
classification.

Blocked percentages use pre-culling image candidates, not all scene Gaussians.
Per-frame results are written below `MODEL/ray_inspection/iteration_N/`.
The viewer shows a fixed saved checkpoint, with its iteration displayed; restart
to load newer weights. With the default 7k/30k save schedule, the 7k checkpoint
predates adaptive densification, whereas the 30k checkpoint contains its result.
