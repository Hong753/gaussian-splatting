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
4. Validate local image regions independently using spatial holdouts excluded
   from the shared scale fit. An image-wide inlier fraction below 70% no longer
   erases every pixel. Each accepted region still needs at least 70% inliers
   within 10% relative camera-z error and median error at most 5% in both fitting
   and held-out anchors. Require at least 24 anchors, 6 holdouts, and inliers in
   at least three of four quadrants. Regions overlap by half their size; depth
   stays unknown beyond half the shorter region dimension from the nearest inlier. No regional
   scale or depth shift is fitted. The global fit report remains available for
   auditing, including rejected global criteria and outlier error percentiles.
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
`requirements-da3.txt` pins PyTorch 2.6.0 + CUDA 12.4, torchvision 0.21.0,
and xFormers 0.0.29.post3 from the CUDA-12.4 wheel repository. The explicit
xFormers wheel URLs select the matching binary for Linux x86_64 Python 3.11/3.12;
using the same version from a different CUDA index can select a different build.
This supports the current host's CUDA-12.4 driver API without upgrading its driver.

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
cd /home/sequor_vfm/gaussian-splatting
git pull --ff-only origin experiment/my-3dgs-changes
conda create -n da3-depth python=3.11 -y
conda activate da3-depth
python -m pip install -r requirements-da3.txt
python scripts/check_da3_environment.py --gpu 3
```

If `da3-depth` already exists, skip `conda create`. The pinned install also
repairs the previous unbounded installation that could select a newer CUDA-major
wheel. `check_da3_environment.py` prints the interpreter and package/CUDA versions,
then executes an actual xFormers CUDA attention operation and imports the DA3 API.
The requirements also explicitly include `addict`, which DA3 imports but omits
from its upstream dependency list. To validate the real API import chain on a
CPU-only machine, run `python scripts/check_da3_environment.py --imports-only`;
the ordinary GPU check remains mandatory in the experiment launcher.

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
git pull --ff-only origin experiment/my-3dgs-changes
bash scripts/run_da3_ablation.sh 3
```

This shell launcher activates each conda environment explicitly, installs/repairs
the pinned DA3 dependencies, checks CUDA, prepares depths, and starts native
training only when every preceding command succeeds. It uses direct sequential
Python invocations and exits on the first failure. Existing valid depth caches
are retained. It does not stop other runs or change the system driver.

Defaults are the source `/data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full`,
output root `/data/3dgs_outputs/orange_4F/1_full`, baseline
`sharp_keyframes_stride3_aa` resolved below that root, depth cache
`da3_lidar_depth_stride3_v2`, and model `3dgs_stride3_aa_da3_v2_occlusion_warm7k_lr50`.
Override these with `DA3_SCENE_ROOT`, `DA3_OUTPUT_ROOT`, `DA3_BASELINE`,
`DA3_DEPTH_DIR`, `DA3_MODEL_DIR`, or `DA3_CONDA_SH`. Arguments after the GPU index
are forwarded to training, for example `bash scripts/run_da3_ablation.sh 3 --evaluate-only`.

`prepare_da3_occlusion.py` writes `index.html` with RGB / calibrated depth /
trusted-coverage overlays, `calibration.json` with per-frame quality and coverage,
and a complete cache. It refuses to mark preparations complete when fewer than
half of the training frames pass calibration or mean reliable tile coverage is
at most 1%. Rejected individual frames use no depth culling.

Cache version 2 records the local calibration policy and saves raw predictions,
confidence, confidence masks, local-calibration masks, multiview support counts,
and final trusted depth. Old version-1 caches can still be read for historical
inspection, but cannot be reused as new preparation outputs. The default output
and model directories are versioned so empty/rejected version-1 masks are not
silently reused. Each prepared view prints coverage after every stage and an
`EMPTY OCCLUSION REFERENCE` reason when no usable tiles survive.

Before full preparation or training, inspect one exact failing image:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate 3dgs
cd /home/sequor_vfm/gaussian-splatting
bash scripts/inspect_da3_occlusion.sh 5 \
  /data/3dgs_outputs/orange_4F/1_full/3dgs_stride3_aa_lidar_occlusion 07519.jpg
```

This prepares that image and its usual neighboring inference window using the
v2 cache, then opens the native checkpoint inspector. It does not retrain the
checkpoint. See [OCCLUSION_INSPECTOR.md](OCCLUSION_INSPECTOR.md) for the reference
comparison and the numeric debug archive emitted after each inspection.

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
