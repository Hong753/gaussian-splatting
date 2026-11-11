# CUDA ray clipping and a multi-frame surface preview

The former whole-Gaussian test kept every splat whose tile footprint touched any
unknown depth. The new opt-in `occlusion_backend=ray` tests every pixel fragment
inside the CUDA forward/backward compositor. It retains native 3DGS + AA and
uses the same Gaussian center camera-z convention as native splat depth sorting.
No derivative passes through the fixed visibility decision.

The full pixel depth comes from CUDA triangle rasterization of a fixed TSDF
mesh. Unknown pixels affect only their own ray (plus one pixel of conservative
boundary padding). There is no 16x16 tile coverage gate. Gaussians without any
contributing fragments have frozen optimizer rows/momentum and do not add
densification statistics for that image. Actual fragment flags distinguish
fully blocked and partially blocked splats. Point-center labels are separate.

## First preview: many input views, one inspected camera

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate 3dgs
cd /home/sequor_vfm/gaussian-splatting
git pull --ff-only
bash scripts/inspect_ray_surface.sh 5 \
  /data/3dgs_outputs/orange_4F/1_full/3dgs_stride3_aa_lidar_occlusion \
  06663.jpg
```

The wrapper builds the separate CUDA extension and runs the real GPU wall/hole
and gradient checks **before** preprocessing. It reuses the cached DA3 owning
group (up to 38 views with the current 40-view budget), validates every group
image, and fuses those surfaces into a diagnostic TSDF mesh. Existing single-view
previews did not fuse the other group views. No extra DA3 forward is necessary
when its raw group cache exists. Surface fusion is CPU preprocessing; mesh depth
rasterization, point visibility tests, and splat forward/backward clipping are
CUDA. This workflow does not launch training.

The smoke test includes native/ray RGB and inverse-depth parity with AA off/on,
finite gradients for every parameter, and direct CUDA binding checks using
expanded and transposed inverse-depth gradients. A reduction such as
`depth.sum()` supplies a zero-stride gradient; the binding materializes it before
CUDA reads a dense pixel array. The shared render wrapper protects the installed
native baseline the same way. If updating after the original NaN parity failure,
rerun the wrapper after `git pull --ff-only`; it rebuilds the extension and retains
the existing DA3 predictions. A new smoke failure names the backend, AA setting,
and offending parameter instead of only reporting an anonymous tensor mismatch.

Open `http://211.172.247.108:8085`. The point-set selector switches between
original LiDAR rows and checkpoint Gaussian centers. Toggle:

| Layer | Color | Meaning |
|---|---|---|
| All scene points | Gray | Every finite point, including outside the camera |
| Camera frustum points | Cyan | Positive depth and projected inside the image |
| Culled points | Red | Point center behind the first surface plus margin |
| Points on surface | Yellow | Within the surface tolerance band |
| Occlusion surface | Blue mesh | Connected triangles, not point markers |
| Unknown depth | Purple | In-frustum point without a reliable surface hit |
| In front of surface | Green | Geometry ahead of the reconstructed surface |
| CUDA blocked / partial / contributing | Red / orange / green | Actual Gaussian fragment flags |

All points are displayed by default; `--max-points` is an explicit display-only
cap. The panel reports exact counts, the number of fused frames, depth coverage,
GT / culling OFF / culling ON images, a difference image, and a gradient probe.
Labels, reports and PNGs are saved beside the diagnostic mesh under
`ray_inspection/`. Model parameters are not optimized or overwritten.

The preview mesh is explicitly marked diagnostic and rejected for training.
It may contain multiple surfaces; the inspected ray uses only its first hit.
Unsupported depth regions remain unknown. A visible wall with no supported
surface is a reconstruction/calibration failure, not something hidden by
inflating a voxel to close all openings.

## Full-scene training

Finish the full trajectory depth cache and scene fusion as described in
[SCENE_OCCLUSION.md](SCENE_OCCLUSION.md). Then select the new compositor explicitly:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate 3dgs
cd /home/sequor_vfm/gaussian-splatting
ray_root=/data/3dgs_outputs/orange_4F/1_full
ray_base=$(python -c 'from scripts.render import resolve_models; import sys; print(resolve_models(None, sys.argv[1], ["sharp_keyframes_stride3_aa"])[0])' "$ray_root")
python scripts/run_occlusion.py \
  --source /data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full \
  --model /data/3dgs_outputs/orange_4F/1_full/3dgs_stride3_aa_ray_occlusion_warm7k_lr50 \
  --baseline "$ray_base" --keyframes "$ray_base/keyframes.json" \
  --scene-geometry /data/3dgs_outputs/orange_4F/1_full/scene_occlusion_stride3_trajectory40_v1/scene.json \
  --occlusion-backend ray --position-warmup-steps 7000 --position-ramp-steps 2000 \
  --position-lr-scale .02 --gpu 5
```

Use the resolved baseline path for this experiment, with the same keyframe
manifest as fusion. The launcher retains GT comparisons for both saved splits,
all-frame PSNR, and actual/novel trajectory comparison videos. The fixed mesh is
available at novel poses; target RGB is not needed for its rendering-time depth.
`--occlusion-backend ray` is also supported by rendering/evaluation overrides.
Legacy checkpoints default to their original `whole` backend.
