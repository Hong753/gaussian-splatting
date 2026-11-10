# LiDAR-calibrated Depth Anything 3 occlusion ablation

This experiment uses native 3DGS + AA, initialized from the same registered LiDAR
cloud and trained on the same sharp stride-3 keyframes. DA3 supplies fixed dense
depth maps for **training-view occlusion only**. Depth regularization and exposure
compensation are disabled. The default schedule matches the LiDAR warmup run:
30k steps, fixed xyz for 7k, a 2k position-LR ramp, 50x smaller LR endpoints, and
densification/opacity resets shifted by 7k.

## How calibration works

1. Run `depth-anything/DA3-LARGE` on overlapping windows of selected training
   images (four core images, two context images on each side), conditioned on the
   existing world-to-camera poses and the camera model used by native 3DGS.
2. Project the registered initialization cloud; retain frontmost camera-z samples
   per 4x4 image cell. This is approximate visibility, not guaranteed first-hit
   ground truth. No raw per-frame LiDAR associations are required.
3. Fit a positive multiplicative depth scale per image using robust log-ratios
   with spatially balanced anchors. DA3 predicts depth, so no inverse-disparity
   shift is fitted. Complete small image regions are held out from fitting.
4. Require enough spatially distributed anchors and acceptable held-out error.
   Reports include the median, p90 over all held-out anchors, p90 over held-out
   inliers, and inlier fraction. The error threshold is relative to camera z.
5. Use confident depth interiors that agree after reprojection into at least two
   other calibrated training views. Discontinuities and uncertain pixels remain
   unknown. A relative depth margin (at least 5%, increased by the held-out inlier
   p90) plus the native 5cm margin makes culling conservative.
6. Convert these maps to the existing conservative tile-depth representation.
   Gaussians whose complete support is behind trusted foreground are excluded
   before rasterization; their image gradients and Adam updates are blocked.

Scale calibration cannot fix a wrong camera pose, nonuniform depth distortions,
or systematically incorrect LiDAR visibility. Multiview predictions can also
agree on the same incorrect surface. Inspect the overlays; this is an ablation,
not a claim that predicted depth is ground truth. Whole-Gaussian culling remains
conservative; partially occluded splats are retained.

## Environment setup (once)

DA3 has different PyTorch dependencies; install it in a separate conda environment.
The native 3DGS environment is used unchanged for training and rendering.

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
cd /home/sequor_vfm/gaussian-splatting
git pull --ff-only origin experiment/my-3dgs-changes
conda create -n da3-depth python=3.11 -y
conda activate da3-depth
python -m pip install -r requirements-da3.txt
```

The DA3 source revision is pinned in `requirements-da3.txt`. The resolved
Hugging Face weight revision, source cloud digest, exact camera signatures,
image identities, and preparation settings are recorded in the cache metadata.
Partial preparations can be rerun with identical settings and reuse finished
frames. Change the output directory when changing inputs/settings.

## Prepare and run on GPU 3

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate 3dgs
cd /home/sequor_vfm/gaussian-splatting
ROOT=/data/3dgs_outputs/orange_4F/1_full
SOURCE=/data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full
BASE=$(python -c 'from scripts.render import resolve_models; import sys; print(resolve_models(None, sys.argv[1], ["sharp_keyframes_stride3_aa"])[0])' "$ROOT")
DEPTH="$ROOT/da3_lidar_depth_stride3"
MODEL="$ROOT/3dgs_stride3_aa_da3_occlusion_warm7k_lr50"

conda activate da3-depth
python -u scripts/prepare_da3_occlusion.py \
  --source "$SOURCE" --keyframes "$BASE/keyframes.json" \
  --point-cloud "$BASE/input.ply" --output "$DEPTH" \
  --process-res 756 --resolution -1 --gpu 3

conda activate 3dgs
python -u scripts/run_da3_occlusion.py \
  --source "$SOURCE" --model "$MODEL" --baseline "$BASE" \
  --depth-dir "$DEPTH" --gpu 3
```

`prepare_da3_occlusion.py` writes `index.html` with RGB / calibrated depth /
trusted-coverage overlays, `calibration.json` with per-frame quality and coverage,
and a complete cache. It refuses to mark preparations complete when fewer than
half of the training frames pass calibration or mean reliable tile coverage is
at most 1%. Rejected individual frames use no depth culling.

Training verifies that the calibration point cloud equals the copied
initialization cloud. An actual CUDA smoke test runs before training and asserts
zero hidden rendering gradients and frozen Adam momentum. No DA3 imports or
nvdiffrast are required in the training environment.

After training, the workflow automatically evaluates PSNR over all original
5,009 source frames, generates GT/render/error comparisons for train and eval
views (64 preview views per split by default), and creates both random-trajectory
and actual-trajectory comparison videos against the supplied 3DGS+AA baseline.
Use `--evaluate-only` with the same arguments to retry post-training stages.

Evaluation and novel-view videos **do not use target-image-derived depths**.
They render the learned Gaussians with normal alpha compositing. Training-only
culling does not guarantee that visibility errors disappear when culling is
removed; these outputs measure whether it improved the learned reconstruction.
Novel-view hard culling would require fusing trusted training depths into fixed
3D geometry and rendering its first-hit depth at the new pose, a separate test.

For a matched comparison against the LiDAR warmup run, compare its **culling-off**
evaluation with this run. Keep training schedule, keyframes, poses, resolution,
and checkpoint iteration identical. Comparing only against the original AA
baseline additionally includes the changed position/densification schedule.

Official API: https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md
