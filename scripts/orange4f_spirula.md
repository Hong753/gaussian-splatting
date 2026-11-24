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
