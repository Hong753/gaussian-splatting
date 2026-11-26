# Orange4F training with exposure-corrected images

These commands repeat the four `scale_control_v1` variants using the completed
exposure export. They use the original approximately 3k-image scene at
`/data/colmap_scenes/orange4f` for camera calibration and point-cloud initialization,
and `/data/3dgs_outputs/orange4f/exposure_check_v1/corrected` for RGB targets.
All four start fresh and retain the previous training hyperparameters.

## Update the code

Run in each terminal before launching its training command:

```bash
cd ~/gaussian-splatting-orange4f
git fetch origin experiment/orange4f-joint-mesh
git switch --detach FETCH_HEAD
conda activate 3dgs
```

The exposure preprocessing command in [EXPOSURE_NORMALIZATION.md](EXPOSURE_NORMALIZATION.md)
must finish first. The launcher requires `status.json` to say `complete`, checks
that the export belongs to this scene, matches every registered filename, and
checks original/export dimensions using image headers. Preprocessing warnings
are retained in the printed and saved plan; they do not stop a complete export
from being used.

## One command per GPU

### GPU 0: baseline

```bash
python -u -m scripts.run_orange4f_scale_control \
  --variant baseline --gpu 0 \
  --exposure-dir /data/3dgs_outputs/orange4f/exposure_check_v1
```

### GPU 1: 3 cm split/clone boundary, no opacity resets

```bash
python -u -m scripts.run_orange4f_scale_control \
  --variant split3cm_noreset --gpu 1 \
  --exposure-dir /data/3dgs_outputs/orange4f/exposure_check_v1
```

### GPU 2: size-driven splitting

```bash
python -u -m scripts.run_orange4f_scale_control \
  --variant size_driven_split --gpu 2 \
  --exposure-dir /data/3dgs_outputs/orange4f/exposure_check_v1
```

### GPU 3: size-driven splitting plus scale regularization

```bash
python -u -m scripts.run_orange4f_scale_control \
  --variant size_driven_split_regularized --gpu 3 \
  --exposure-dir /data/3dgs_outputs/orange4f/exposure_check_v1
```

The launcher sets `CUDA_VISIBLE_DEVICES` for training, rendering, and auditing.
The default output root with `--exposure-dir` is:

```text
/data/3dgs_outputs/orange4f/scale_control_exposure_v1/
```

| GPU | Run subdirectory | Densification settings |
| --- | --- | --- |
| 0 | `baseline` | Existing gradient densification and opacity resets |
| 1 | `split3cm_noreset` | Gradient densification with a 3 cm split/clone boundary; no opacity resets |
| 2 | `size_driven_split` | GPU 1 settings plus budgeted subdivision above 10 cm maximum sigma |
| 3 | `size_driven_split_regularized` | GPU 2 settings plus the existing soft scale penalty above 5 cm |

The 3 cm boundary is a gradient-driven split-versus-clone threshold, not a hard
upper bound on Gaussian size. The size-driven budget limits extra subdivision,
not total Gaussian count. Scene units are assumed to be meters, as in the
original runs. Native deletion by screen or world size stays disabled in all
four variants; opacity pruning stays enabled.

## Schedule and outputs

- 120,000 training iterations, with the existing 1,600-pixel width setting.
- All registered images train; these commands do not create a held-out split.
- PLY snapshots and resumable checkpoints at 5,000, 6,000, 10,000, 15,000,
  20,000, 30,000, 60,000, 100,000, and 120,000 iterations.
- Fixed training previews at step 1, every 500 iterations, and the final step.
- Automatic final rendering of all available train/test views and a 12-view
  blur audit. With this training split there are no held-out test views.

`scale_control_plan.json` records the absolute corrected-image path and exposure
provenance. `exposure_preprocessing/` inside each run snapshots the preprocessing
status, summary, frame manifest, and CSV (when present), with SHA-256 metadata
hashes in the plan. RGB images are referenced in place, so keep the completed
export available during training and later rendering. Image headers are checked;
the launcher does not compute full pixel hashes for the image dataset.

Learned per-image exposure transforms remain disabled. Live previews, final
renders, and blur-audit targets all use the corrected images through the saved
`cfg_args`. The source calibration, initial point-cloud geometry/colors, and
optimization settings retain their original definitions.

PSNR values from the raw-image runs and corrected-image runs use different target
RGBs. Compare the four corrected runs to one another directly; a raw-versus-corrected
PSNR comparison needs a common-target evaluation protocol.

Append `--dry-run` to any command to validate inputs and print its complete plan
without creating a run or starting CUDA work. Existing run directories are never
overwritten. To repeat the experiments, pass a new parent such as
`--output-root /data/3dgs_outputs/orange4f/scale_control_exposure_v2` to all four
commands. Without `--exposure-dir`, the launcher retains the original image
directory and `scale_control_v1` output default.

## CPU checks

```bash
python -m unittest discover -s tests -p 'test_exposure*.py' -v
```

These tests use synthetic fixtures and do not represent an Orange4F training
result. Actual GPU training and rendering require the dataset and native CUDA
extensions on the training machine.
