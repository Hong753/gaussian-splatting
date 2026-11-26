# Exposure normalization for visual inspection

This CPU-only tool estimates one exposure gain per registered image from the
brightness of shared COLMAP points. It writes a separate corrected image tree
and before/after galleries. It does not modify source images, camera poses,
training settings, or running experiments.

## Run on Orange4F

```bash
cd ~/gaussian-splatting-orange4f
conda activate 3dgs
python -u -m scripts.normalize_image_exposure \
  --source /data/colmap_scenes/orange4f \
  --output /data/3dgs_outputs/orange4f/exposure_check_v1 \
  --workers 4
```

No GPU is needed. Dependencies are NumPy, SciPy and Pillow. A new output
directory is required; choose another `--output` to repeat the experiment.

## Inspect the result

| Output | Contents |
| --- | --- |
| `corrected/` | All registered images, retaining source filenames and dimensions |
| `index.html` | Browse every frame, search filenames, compare originals and corrections, open full resolution |
| `preview.html` | Self-contained portable gallery with 24 representative before/after pairs |
| `contact_sheet.jpg` | Twelve paired examples spanning the sequence and strong/abrupt corrections |
| `exposure.csv` | Estimated EV, applied correction/gain, support, component, residual and clipping per image |
| `summary.json` | Settings, convergence, held-out consistency, disconnected components and warnings |
| `frames.json` | Image records used by the galleries |
| `status.json` | `complete` only after all exports and reports have succeeded |

Open `index.html` on the server desktop. `preview.html` can be copied to another
machine and opened alone, or uploaded for review. The full gallery uses an
`original/` symlink to the source images and must remain with its output tree.

An optional local HTTP server is:

```bash
python -m http.server 8088 --bind 127.0.0.1 \
  --directory /data/3dgs_outputs/orange4f/exposure_check_v1
```

## Interpretation

The fit uses log2 luminance in approximate linear sRGB:
`observed_log_luminance = frame_exposure_ev + shared_point_radiance`.
It robustly estimates frame offsets, then subtracts the mean exposure within
each connected group. The target is average **EV**, equivalent to geometric
mean exposure, not equal mean brightness across different scene contents.
COLMAP's fused point colors are not used as targets. Saturated and near-black
samples are excluded. Whole-image histogram matching is not used as a fallback.

Twenty percent of entire point tracks are held out from the fit. The report
compares their cross-view luminance disagreement before and after correction.
Prefer lower held-out disagreement and stable local appearance, not merely
similar-looking whole-image brightness. Held-out metrics describe the applied
gain before output clipping/quantization; image clipping is reported separately.

Corrections are capped at ±2 EV by default (`--max-correction-ev`), while the
uncapped estimates remain in the CSV. Unsupported frames receive zero correction.
Disconnected groups cannot be aligned photometrically from these observations;
they are normalized independently and explicitly reported.

JPEGs are re-encoded at quality 100 with 4:4:4 chroma sampling; PNGs remain PNGs.
This is global scalar exposure correction, not automatic-white-balance or local
tone-mapping correction. JPEG camera response is only approximately sRGB.
Clipped texture and motion blur cannot be recovered by adjusting brightness.
EXIF exposure metadata, when present, is recorded as supporting evidence, but
estimated gain variation alone does not prove the camera used automatic exposure.

The COLMAP model must contain actual 2D observations and shared point IDs.
Pose-only camera exports are insufficient. Input image dimensions must match
the model. `--allow-resized` is available only for a known full-frame resize;
do not use it to disguise a crop or a mismatched image set.

## Train with the completed export

[ORANGE4F_EXPOSURE_TRAINING.md](ORANGE4F_EXPOSURE_TRAINING.md) contains the four
GPU 0-3 launch commands. They use this export's `corrected/` directory while
retaining the original scene calibration and scale-control experiments, with
new outputs under `scale_control_exposure_v1`.

[ORANGE4F_SCAFFOLD_EXPOSURE.md](ORANGE4F_SCAFFOLD_EXPOSURE.md) provides a matched
Scaffold-GS suite on the same corrected images: anchor growth on/off crossed
with unconstrained/bounded child-center displacement, on GPUs 0-3.

## CPU validation

```bash
python -m unittest discover -s tests -p 'test_exposure*.py' -v
```

Tests use synthetic exposures, shared-point tracks, outliers, disconnected
components and image exports. They do not constitute an Orange4F result.
