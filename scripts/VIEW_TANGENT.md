# Visible-surface tangent densification

This native 3DGS + AA ablation combines independent depth culling with per-view
surface-guided adaptive densification. No Proxy-GS dependency or pose changes.

## Method

- Read **raw camera-z**, either first-hit rasterized fixed LiDAR surfels or the
  LiDAR-calibrated DA3 depth plus its multiview-trusted mask. Never fit planes to
  safety-inflated occlusion tile depths. LiDAR surfel coverage is an approximation
  to measured surface visibility, not a complete per-frame sensor return set.
- Require fully trusted 7×7 neighborhoods, limited depth variation, local plane
  consistency, and a nondegenerate viewing angle. Backproject with the depth
  map's own intrinsics; rotate normals using the exact training camera pose.
- Accumulate native screen-space densification gradients only for rendered
  Gaussian centers matching the front surface (3 cm + 1% of depth tolerance).
  Parents need at least two matched observations in the accumulation window.
  Inconsistent plane associations clear that parent's accumulated statistics.
- Rank eligible parents by their accumulated gradient. Clone small Gaussians
  with a nonzero tangent displacement; split large ones using opposite tangent
  displacements. Bound each child within a local measured disk (at most 10 cm,
  also limited by the neighborhood's image footprint). Preserve the parent's
  normal offset instead of snapping LiDAR initialization to possibly biased DA3.
- Unknown, discontinuous, occluded, or drifted parents do not seed new Gaussians.
  They still follow the existing rendering/culling and optimization rules. This
  experiment changes densification, not the positional optimizer's update rule.
  Covariances remain native 3DGS, not flattened or locked to the normal.
- Limit growth to 50,000 parents per event and 12 million total Gaussians. Each
  selected parent adds one net Gaussian; at most 100,000 child tensors are
  appended at once. About 360 MB of reference data is added for 10M float32 points.

The default run uses 30k iterations, fixed xyz for 7k, a 2k LR ramp, and 50× lower
position LR endpoints (3.2e-6 to 3.2e-8, before the existing scene-extent scaling).
Densification runs at 7600–21900; opacity resets at 10k/13k/16k/19k. No additional
depth/normal loss is enabled. Culling continues to suppress image gradients and
freeze Adam state for fully hidden Gaussians.

## Run on GPU 2

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate 3dgs
cd /home/sequor_vfm/gaussian-splatting
git pull --ff-only origin experiment/my-3dgs-changes
bash scripts/run_view_tangent.sh 2 lidar
```

Default model:
`/data/3dgs_outputs/orange_4F/1_full/3dgs_stride3_aa_lidar_occlusion_viewtangent_warm7k_lr50`.
The existing sharp stride-3 AA baseline supplies the exact keyframe manifest.
The native environment must already have `requirements-occlusion.txt`, as for
the existing LiDAR culling runs. The launcher runs native CUDA checks first.

Once the existing DA3 preparation has completed, use a separate output:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate 3dgs
cd /home/sequor_vfm/gaussian-splatting
bash scripts/run_view_tangent.sh 2 da3
```

This consumes `da3_lidar_depth_stride3` without starting another DA3 preprocessing
job. It fails early if that cache is absent/incomplete. Do not run both commands
simultaneously on GPU 2. `VIEW_TANGENT_MODEL_DIR`, `DA3_BASELINE`, `DA3_DEPTH_DIR`,
`DA3_OUTPUT_ROOT`, and `DA3_SCENE_ROOT` override the corresponding defaults.
Append `--dry-run` to inspect the plan or `--evaluate-only` to retry evaluation.

## Outputs and interpretation

- `view_tangent_densification.jsonl`: eligible/selected parents, clones, splits,
  additions, pruning and point counts per event. The progress bar and TensorBoard
  report the matched fraction among currently rendered Gaussians. If this stays
  low, insufficient depth/surfel coverage is limiting growth; no holes are filled.
- PSNR over **all source frames** under `psnr_all_frames/iteration_30000` (5,009
  for this scene), followed by actual-image/render/error galleries for both splits
  (up to 64 saved examples per split; metrics still cover all frames).
- Shared actual-camera video with GT and both renders, and the seed-42 random
  trajectory comparison against the original 3DGS + AA baseline in `comparisons`.
  Same iteration and source poses; no camera refinement is performed.

LiDAR runs evaluate with culling on and off, including the baseline four-way
comparison. DA3 runs always evaluate/render without target-image depth inputs.
Novel-view densification needs no new depths: only training uses these patches.
LiDAR culling itself can be recomputed from the fixed geometry for novel cameras.
DA3-only novel rendering is ordinary AA 3DGS, so no guarantee of hard occlusion
remains once its training-only mask is removed.

To isolate this densification change, compare with the same LiDAR/DA3 culling
source, warmup/LR, iteration, and rendering mode. The original AA baseline video
also differs in culling and schedule. This is a hypothesis test, not a measured
quality improvement; sparse trusted coverage can suppress useful growth.

CPU verification covers front/back surfaces, holes and jumps, rotated cameras,
bounded tangent children, growth budgets, pruning and reference bookkeeping.
`scripts/smoke_view_tangent.py --gpu 2` additionally exercises the real CUDA AA
rasterizer, masked gradients, clone/split mutations and rendering without depth.
