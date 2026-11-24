# orange_4F using fresh Spirula SfM

Use the [Spirula v2026.11.12 Linux Vulkan build](https://github.com/harry7557558/spirula-studio/releases/tag/v2026.11.12). Start with the native-resolution images from the existing stride-3 AA run. The new SfM estimates poses and intrinsics from the images, including views held out of splat training, as in the original reconstruction protocol.

## SfM only on all Orange4F images

```bash
conda activate 3dgs &&
cd ~/gaussian-splatting-orange4f &&
git fetch origin experiment/orange4f-joint-mesh &&
git switch --detach FETCH_HEAD &&
bash scripts/orange4f_spirula_sfm.sh
```

This estimates fresh extrinsics and pinhole intrinsics from the full image sequence at `/data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full/images`, including future rendering holdouts. The feature extractor uses a maximum long edge of 1,600 pixels and exports keypoints in the original image coordinates. The scene directory links to the original RGB files. Spirula writes the principal reconstruction to `/data/3dgs_outputs/orange4f/3dgs_finetuning/spirula_sfm_rgb_w1600/sparse/0`. Further numbered components have independent coordinate frames. The native summary in `sfm.log` records registration coverage, point count and reprojection error. An RGB-only solve has its own similarity gauge.

Set `SPIRULA_DEVICE` to a Vulkan device selector on a shared host. The default is `auto`. The launcher uses `SPIRULA_BIN`, then searches PATH, then installs the checksum-verified official Linux x86-64 binary in the user cache. The launcher probes the executable and reuses working system or Vulkan-only setups. On older hosts (including glibc 2.35), it downloads checksum-pinned Ubuntu 24.04 libraries and starts Spirula through their matching dynamic linker. This runtime lives in the user cache. The host supplies the Vulkan GPU driver. The Vulkan device listing in `sfm.log` shows which GPU the installed driver exposes. Rerun the same command to reuse native SfM caches after an interruption. Optional positional arguments are the source and output scene roots.

## Train vanilla Graphdeco 3DGS using the completed Spirula poses

```bash
conda activate 3dgs &&
cd ~/gaussian-splatting-orange4f &&
git fetch origin experiment/orange4f-joint-mesh &&
git switch --detach FETCH_HEAD &&
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.orange4f_spirula_3dgs
```

This uses the fork's native 3DGS trainer with SH degree 3, Adam, standard densification, default RGB loss and AA off. The camera adapter passes Spirula's focal lengths and principal points to the rasterizer. The run links the original photos and three new SfM binaries into its own scene directory. It initializes Gaussians from all Spirula XYZ/RGB points in this reconstruction's coordinates. Each of the 5,009 RGBs participated in SfM. Gaussian optimization uses the 1,670 sharp stride-3 keyframes, and the other 3,339 supply the held-out image scores. The previous split is reused when its registered name list matches. Otherwise the sharpest frame in each successive group of three is selected. Training, previews and scoring use 1,600-pixel width.

The output root is `/data/3dgs_outputs/orange4f/3dgs_finetuning/spirula_3dgs_stride3_w1600`. Open `model/debug_previews/index.html` for fixed train and held-out views at step 1, every 500 steps and step 30,000. Direct file viewing and HTTP viewing are supported. `protocol.json` records the split, source fingerprints and seed count. `model/point_cloud/iteration_7000` and `iteration_30000` contain the Gaussian PLYs. The final `evaluation/iteration_30000/` contains `psnr.json`, a gallery at `index.html`, per-frame GT/render comparisons, and `videos/recorded.mp4`, `videos/gt_render.mp4` and `videos/random/render.mp4`. PSNR is per-channel, averaged over channels and then images. The random path uses a centered virtual camera, seed 42 and smooth offsets from the recovered trajectory. It has no reference RGB.

To score the immutable step-7,000 PLY during training, use another available GPU once the PLY appears. For example:

```bash
conda activate 3dgs &&
cd ~/gaussian-splatting-orange4f &&
CUDA_VISIBLE_DEVICES=1 python -u -m scripts.orange4f_spirula_3dgs --evaluate-iteration 7000
```

## 1. Prepare the comparison and run SfM

In a Bash session on the GPU server, set the baseline directory and the Spirula executable. Use the full-scene RGB-only AA baseline with its 30k checkpoint and saved keyframe manifest.

~~~bash
cd ~/gaussian-splatting
git switch experiment/my-3dgs-changes
git pull --ff-only

BASE=/absolute/path/to/the/stride3-AA/3dgs
SPIRULA=/absolute/path/to/spirula
DEVICE=auto
RUN=/data/3dgs_outputs/orange4f/3dgs_finetuning/spirula_fresh_sfm_v1
HELPER=scripts/orange4f_spirula_compare.py

conda run --no-capture-output -n 3dgs python "$HELPER" prepare "$BASE" "$RUN"
conda run --no-capture-output -n 3dgs python "$HELPER" sfm "$RUN" \
  --spirula "$SPIRULA" --device "$DEVICE"
~~~

Set `DEVICE` to a free Spirula Vulkan device index on a shared server. Results go under `$RUN/sfm`. Inspect `$RUN/sfm/registration.json` for training and held-out coverage. The image names keep the original sort order within each capture folder, and the split labels control which registered images train the splat.

### Optional LiDAR alignment

Use a raw E57, LAS, or PLY cloud registered to the world frame of the original baseline cameras. The orange_4F floor audit defaults to the source dataset's `input.ply` as the raw cloud. Verify the frame and point to the actual scan. The baseline model's `input.ply` records its training seeds.

~~~bash
CLOUD=/absolute/path/to/the/aligned/raw_lidar.ply

conda run --no-capture-output -n 3dgs python "$HELPER" sfm "$RUN" \
  --spirula "$SPIRULA" --device "$DEVICE" --lidar-cloud "$CLOUD"
~~~

The second call reuses the completed SfM. It constructs alignment anchors from the existing **training** cameras in the scan frame, estimates a similarity transform for the fresh reconstruction, and lets Spirula refine the alignment with ICP. The LiDAR step places the fresh poses into the scan frame and makes seed points and per-view depth and normal maps. Check coverage in `$RUN/sfm_lidar/registration.json` and inspect scan overlays before using depth supervision. [Spirula LiDAR alignment](https://github.com/harry7557558/spirula-studio/blob/v2026.11.12/docs/notes/lidar-alignment.md)

## 2. Train the warp pair

Pick `sfm` for RGB only or `sfm_lidar` for the scan-assisted run. Both warp arms share the selected reconstruction, seed cloud, resolution, and splat cap. The LiDAR settings below enable its generated depth and normal maps for both arms.

~~~bash
VARIANT=sfm_lidar  # Set to sfm for the RGB-only reconstruction.
DATASET="$RUN/$VARIANT"
OUT="$RUN/training/$VARIANT"
CAP=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["cap_max"])' \
  "$RUN/split.json")
GEOMETRY=(--load-depths 0 --load-normals 0)
if [[ "$VARIANT" == sfm_lidar ]]; then
  GEOMETRY=(--load-depths 1 --load-normals 1 \
    --depth-supervision-weight 0.05 --normal-supervision-weight 0.01)
fi

(
  set -euo pipefail
  test -f "$DATASET/registration.json"
  mkdir -p "$OUT"
  for warp in off basic; do
    "$SPIRULA" train 3dgs \
      --device "$DEVICE" \
      --data "$DATASET" --data-format colmap --eval-mode filename \
      --train-resolution-divisor 1 --exif-orientation none \
      --seed-pointcloud sparse --random-init never \
      --cap-max "$CAP" --min-init-fraction 0 \
      --num-iterations 30000 --steps-per-save 30000 \
      "${GEOMETRY[@]}" \
      --save-eval-images 1 --eval-align 0 --eval-lpips 0 \
      --disable-viewer 1 \
      --calib-correction "$warp" \
      --output-dir-prefix "$OUT" --output-dir-name "warp_$warp" \
      2>&1 | tee "$OUT/warp_$warp.log"
  done
)
~~~

Allow disk space for native-resolution held-out PNGs from both runs. PNGs are used to compute a common per-channel PSNR metric.

## 3. Compare the same registered images

~~~bash
FLOOR=/data/3dgs_outputs/orange4f/3dgs_finetuning/stride3_noaa_v1/multiview_floor_distance_v3/protocol.json
CROPS=()
if [[ -f "$FLOOR" ]]; then
  CROPS=(--floor-protocol "$FLOOR")
fi

conda run --no-capture-output -n 3dgs python "$HELPER" \
  score "$RUN" --model-data "$DATASET" --output-dir "$OUT" "${CROPS[@]}"
~~~

Open `$OUT/comparison.html` for photos, renders, error maps, and per-view scores. Use `$OUT/raw_per_frame.csv` for the full shared held-out set. The baseline uses its original poses and Spirula uses its fresh poses. Compare `warp_basic` with `warp_off` to measure the warp with poses held fixed. Compare either with `3dgs_aa` to measure the end-to-end change.

## Replace the Orange4F RGB-only SfM model with learned, LiDAR-assisted SfM

```bash
conda activate 3dgs &&
cd ~/gaussian-splatting-orange4f &&
git fetch origin experiment/orange4f-joint-mesh &&
git switch --detach FETCH_HEAD &&
bash scripts/run_orange4f_superglue_lidar.sh
```

This reruns SfM only. The launcher picks one idle CUDA GPU for the neural networks, or honors a single `CUDA_VISIBLE_DEVICES`. Both mapping and scan-anchored bundle adjustment use CPU solvers. Inputs are every original registered image, the original pinhole calibration, and a metric scan registered to the original camera frame. The default scan is the original dataset's `input.ply`. Set `ORANGE4F_LIDAR_PLY=/absolute/path/to/registered_scan.ply` for another scan. Its coordinates must be in metres. The original exported camera poses place the fresh reconstruction in that frame by one similarity transform. They do not seed individual camera poses.

SuperPoint extracts up to 2,048 features at a maximum image side of 1,600 px. The pinned indoor SuperGlue network matches temporal neighbours and nonlocal appearance candidates. OpenCV MAGSAC verifies fundamental matrices and homographies. The resulting correspondences feed the Spirula v2026.11.12 mapper. It starts from the original pinhole intrinsics and can refine focal length. The LiDAR bundle adjustment holds those mapped intrinsics fixed. In the metric frame, RGB tracks with at least three observations and support from a local plane in the scan are projected onto that plane. These projected landmarks are fixed during robust COLMAP bundle adjustment. All camera poses and the other landmarks can move. The tangential coordinates of an anchor come from visual SfM. This is a custom scan-constrained SfM stage. Repeated walls or a misregistered input scan can produce incorrect associations. The report includes per-camera support and the plane residual on tracks excluded from anchoring.

The candidate is reopened and checked for registration of the full input set before exchanging `spirula_sfm_rgb_w1600/sparse/` with the complete new model. The old folder is retained at `spirula_sfm_rgb_w1600/sparse_sift_backup/`. Existing Gaussian training scene links are pinned to the old files before the switch. Interrupted runs reuse the feature and verified match caches in `spirula_sfm_rgb_w1600/superpoint_superglue_lidar/`. Look at `learned.log`, `mapper.log`, and `lidar-ba.log` there. The exact input fingerprints, final statistics, and solver report are in `sparse/superglue_lidar.json`. A new invocation of the vanilla 3DGS launcher picks a separate output directory for the learned/LiDAR model.

The launcher reuses the pinned SuperGlue repository when already cached, otherwise it downloads the models into the SfM workspace. When the required pycolmap 3.11.1 bindings are missing, it places their wheel in an isolated subdirectory of the active Conda environment.
