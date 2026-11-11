# Trajectory-spanning geometry priors

The old preparation uses four output frames plus two context frames on each
side (up to eight adjacent images). Its per-image local LiDAR validation can
erase a prediction before multiview checks are attempted. Around 80% "confident"
coverage mainly reflects the top-80% confidence selection, not measured accuracy.

PlanarGS's [grouping code](https://github.com/SJTU-ViSYS-team/PlanarGS/blob/main/run_geomprior.py)
distributes sequence images across groups of about 40. Its
[DUSt3R path](https://github.com/SJTU-ViSYS-team/PlanarGS/blob/main/geomprior/run_dust3r.py)
uses image pairs and global alignment within each group, followed by alignment
to COLMAP geometry. It does not infer all scene frames in one model batch.

Our adaptation keeps supplied COLMAP poses/intrinsics and uses:

1. A **40-image budget** per group: targets distributed across the full ordered
   trajectory, eight shared references, and two nearby companions per target.
   Defaults yield up to 38 images after reserving these roles; deduplication can
   reduce that count. Each output has exactly one owning group.
2. Pose-conditioned DA3, `saddle_balanced` reference selection, and processing
   resolution 504. Local companions preserve overlap within the broad context.
3. One multiplicative metric scale per group from equal-view LiDAR proposals.
   At least three proposals and 50% consensus within 10% are required. This is
   only a scale proposal: held-out local LiDAR checks still validate donors.
   Camera poses remain fixed and no additive depth shift is fitted.
4. Target pixels without their own LiDAR anchors may survive if **two other
   locally LiDAR-validated views** agree geometrically. Reprojection agreement
   ranks overlapping donors; up to eight are checked at full resolution.
5. Borrowed pixels never become donors, preventing circular validation.
   Confidence/depth-boundary holes stay unknown. Borrowed depths receive at
   least a 10% relative culling margin. Unsupported regions remain unknown.

Shared images, fixed poses and LiDAR metric alignment connect groups. This is
not one transformer pass or joint bundle adjustment over all frames, and does
not reproduce PlanarGS's DUSt3R optimizer. Actual coverage and memory use still
require a server preview.

## Audit the existing run without repeating inference

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate 3dgs
cd /home/sequor_vfm/gaussian-splatting
python scripts/audit_da3_occlusion.py \
  --depth-dir /data/3dgs_outputs/orange_4F/1_full/da3_lidar_depth_stride3_v2 \
  --images 06663.jpg 06705.jpg 06821.jpg 06851.jpg --collect
```

Works on partial caches. Reports rejection stages, anchor counts/errors, local
rejection counts, and coverage. `alignment_debug.zip` includes numeric depths
and anchors for the chosen frames. These reasons do not by themselves establish
whether the LiDAR projection, predicted depth, or camera registration is wrong.

## Preview a failing frame

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate 3dgs
cd /home/sequor_vfm/gaussian-splatting
bash scripts/inspect_da3_trajectory.sh 5 \
  /data/3dgs_outputs/orange_4F/1_full/3dgs_stride3_aa_lidar_occlusion \
  06663.jpg
```

The wrapper activates both environments, infers the full owning group, writes
the requested frame, prints the audit, and opens the native Gaussian inspector
on port 8082 with the backward check. All group predictions are saved in
`predictions/`; later full preparation reuses that inference.

New default cache: `da3_lidar_depth_stride3_trajectory40_v1` (cache format 2).
The old window cache stays separate. `DA3_TRAJECTORY_DEPTH_DIR` overrides this
path. `groups.json` records input lists and output owners. Cache identity includes
grouping, resolution and validation policy.

## Complete the scene after inspecting the preview

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate 3dgs
cd /home/sequor_vfm/gaussian-splatting
bash scripts/prepare_scene_occlusion.sh 5 \
  /data/3dgs_outputs/orange_4F/1_full/3dgs_stride3_aa_lidar_occlusion \
  07519.jpg
```

The scene wrapper now uses the same trajectory grouping/cache before fusion and
all-frame LiDAR visibility. Its mesh output is
`scene_occlusion_stride3_trajectory40_v1`. See [SCENE_OCCLUSION.md](SCENE_OCCLUSION.md)
for viewing and subsequent training. Both cache paths are separate from the
older eight-view experiment.
