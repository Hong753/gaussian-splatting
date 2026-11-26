# Scaffold-GS on the exposure-corrected Orange4F images

This suite compares anchor growth and child-center locality while holding the
completed exposure export, original source cameras, SfM initialization, voxel
spacing, architecture, optimization settings, and rendering protocol fixed.
It uses the repository's adapted Scaffold implementation, not a separate
unmodified checkout of the authors' training code.

| Input/output | Path |
| --- | --- |
| Original approximately 3k-image COLMAP scene | `/data/colmap_scenes/orange4f` |
| Completed exposure export | `/data/3dgs_outputs/orange4f/exposure_check_v1` |
| Corrected RGB | `/data/3dgs_outputs/orange4f/exposure_check_v1/corrected` |
| New run root | `/data/3dgs_outputs/orange4f/scaffold_exposure_v1` |

## Four matched variants

| GPU | Variant | Add new anchors | Prune anchors | Child-center displacement radius |
| --- | --- | --- | --- | --- |
| 0 | `baseline` | Yes | Yes | Unbounded |
| 1 | `fixed_anchors` | No | Yes | Unbounded |
| 2 | `bounded_offsets` | Yes | Yes | `0.05` scene units |
| 3 | `fixed_anchors_bounded_offsets` | No | Yes | `0.05` scene units |

**Surviving anchor coordinates are fixed in every run.** Standard Scaffold
already uses zero anchor-coordinate learning rates. Merely freezing those
coordinates would duplicate the baseline's geometry policy. Here `fixed_anchors`
means no new anchors can be added; the initial pool may still shrink by pruning.
It does not mean a fixed anchor count or fixed Gaussian centers.

The standard decoded child center is

```text
center = anchor + exp(log_scaling[:3]) * learned_offset
```

The bounded variants project the scaled displacement onto a radius-0.05 ball
after each optimizer step, before anchor growth and checkpoint saving. A bound
on the raw offset alone would not prevent the learned scaling from moving the
child farther away. Projected offsets are stored in the normal Scaffold PLY;
the existing decoder, renderer, and viewer therefore use the same saved centers.
Optimizer checkpoints record the radius policy and reject a different policy
on resume.

`0.05` is a heuristic, equal to 2.5 times the common voxel spacing `0.02`.
It means 5 cm only if the source coordinates have metric scale; the launcher
does not establish that scale. This constrains center displacement, not
Gaussian extent or surface distance. Growth-enabled runs can place new anchors
and are not confined to neighborhoods of the original anchor set.

The fixed-pool variants may struggle where sparse SfM points leave coverage
gaps. The radius cap may also suppress useful motion and new-anchor proposals.
The inherited multiresolution growth rule can skip finer levels if its coarse
level adds nothing; this rule is identical in all four runs. Use the logged
projection and growth/pruning counts to distinguish stable geometry from
insufficient capacity. No assumption is made that a constrained variant wins.

Pruning remains enabled everywhere. In the older observation experiments,
disabling pruning also selects a different optimizer/update path; these runs
all retain ordinary Adam and the same volume regularizer.

## Update once

```bash
cd ~/gaussian-splatting-orange4f
git fetch origin experiment/orange4f-joint-mesh
git switch --detach FETCH_HEAD
```

Use GPUs 0-3 when available. Each command launches a fresh run, checks the
completed exposure export, performs a small synthetic CUDA preflight, trains,
and renders the final image comparisons. It reuses the installed `simple_knn`
and rasterizer. It does not rebuild extensions, stop other processes, or
overwrite existing experiments.

## GPU 0

```bash
cd ~/gaussian-splatting-orange4f
conda activate 3dgs
python -u -m scripts.run_orange4f_scaffold_exposure \
  --variant baseline --gpu 0 \
  --exposure-dir /data/3dgs_outputs/orange4f/exposure_check_v1
```

## GPU 1

```bash
cd ~/gaussian-splatting-orange4f
conda activate 3dgs
python -u -m scripts.run_orange4f_scaffold_exposure \
  --variant fixed_anchors --gpu 1 \
  --exposure-dir /data/3dgs_outputs/orange4f/exposure_check_v1
```

## GPU 2

```bash
cd ~/gaussian-splatting-orange4f
conda activate 3dgs
python -u -m scripts.run_orange4f_scaffold_exposure \
  --variant bounded_offsets --gpu 2 \
  --exposure-dir /data/3dgs_outputs/orange4f/exposure_check_v1
```

## GPU 3

```bash
cd ~/gaussian-splatting-orange4f
conda activate 3dgs
python -u -m scripts.run_orange4f_scaffold_exposure \
  --variant fixed_anchors_bounded_offsets --gpu 3 \
  --exposure-dir /data/3dgs_outputs/orange4f/exposure_check_v1
```

The run directory is `<new run root>/<variant>`. Append `--dry-run` to validate
inputs and print commands without writing outputs or running CUDA. A failed
CUDA preflight happens before a run folder is created. For a repeat, supply
a new `--output-root` to all four commands. `--offset-radius` overrides the
bound in the two bounded variants; it should have the same value in both.

## Shared training settings

- 120,000 iterations, all registered source images, 1,600-pixel training width.
- Original source cameras and scene extent; no pose refinement.
- Identical source point coordinates voxelized at `0.02` scene-unit spacing.
  Scaffold initializes offsets/features to zero; it does not use the point
  cloud's colors as initial feature colors.
- 10 child offsets per anchor and 32-dimensional anchor features.
- No appearance embedding or learned per-image exposure. No antialiasing.
- Stock offset LR endpoints `0.01` to `0.0001`, with the usual scene-extent
  factor; common feature, scale, and MLP learning rates.
- All learning-rate horizons extended to 120k. The 30k population schedule is
  stretched by four: statistics start after 2k, updates every 400 steps after
  6k and before 60k. Actual growth/pruning events run from **6,400 to 59,600**.
  These timing choices are shared; they are not a claim of optimal tuning.

With this interval, the existing observation thresholds require more than
160 observations for a growth candidate and more than 320 for pruning
eligibility. Rarely seen anchors may not meet these thresholds. Counts and
the saved plans should be inspected alongside RGB quality.

## Saves, previews, and final comparisons

Snapshots and optimizer checkpoints are saved at:

```text
5000 6000 10000 15000 20000 30000 60000 100000 120000
```

Each snapshot includes the PLY, opacity/covariance/color networks, and a
`ready.json` publication marker. The model directory also has `chkpntN.pth`
for resuming. Keep the whole iteration directory: Scaffold's PLY alone is
insufficient to render the model.

Fixed-view GT/render/error previews run at step 1, every 500 iterations, and
the final step. All frames are training frames; there is no disjoint test
split. After training, `scripts/render.py --split both` produces all available
training-view comparisons, per-view PSNR, error images, and a browsable report,
and reports that the test split is empty. `--render-max-views` can limit this
final report, with `0` meaning all registered frames.

Each run saves `scaffold_exposure_plan.json`, an all-frame `keyframes.json`,
and a copy of the exposure metadata under `exposure_preprocessing/`. The
corrected images are referenced in place, so retain the export for training
and future image comparisons. `scaffold_training.jsonl` records training,
anchor counts, and the new offset/growth diagnostics.

Compare `offset_radius_before_projection_max` across free and bounded runs,
and `offset_projection_fraction` in the bounded runs. Projection counts refer
to the logged optimizer step, not a cumulative interval. The `anchor_update`
entry at population-update events gives actual added and pruned counts; a
constant total can otherwise hide simultaneous growth and pruning.

## Browser viewer

Start on the server, using a GPU with room for a separate model allocation:

```bash
cd ~/gaussian-splatting-orange4f
conda activate 3dgs
python -u scripts/view_scaffold.py \
  --suite /data/3dgs_outputs/orange4f/scaffold_exposure_v1 \
  --gpu 1 --host 127.0.0.1 --port 8091
```

On the local computer, replace `SERVER_IP` with the usual SSH host:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 8091:127.0.0.1:8091 sequor_vfm@SERVER_IP
```

Open `http://127.0.0.1:8091/`. Choose **Run**, **Checkpoint**, and **Open
checkpoint**. The list refreshes every ten seconds. Unlike native-3DGS
single-run mode, Scaffold suite mode requires selecting later checkpoints
explicitly. Set render width to 1600, enable Lossless PNG, and retain the
saved antialiasing setting when judging the floor details.

## Exposure interpretation

The existing shared-point scalar correction is a controlled baseline for
frame-dependent brightness variation. Visual improvement does not establish
that it is the best possible correction or that exposure was the only blur
source. One scalar does not correct white-balance changes, local tone mapping,
vignetting, or changing illumination; clipping and motion blur cannot be undone.
Keep this export fixed while comparing Scaffold variants. Richer photometric
models should be a separate experiment supported by remaining cross-view errors.

For comparison with 3DGS, use the same corrected export, image names, resolution,
and saved iteration. Raw-target and corrected-target PSNR measure different
objectives. These all-training-view reports measure fit, not held-out
generalization.

## CPU validation

```bash
python -m pytest -q \
  tests/test_orange4f_scaffold_exposure.py \
  tests/test_scaffold_offset_constraints.py \
  tests/test_orange4f_scaffold.py \
  tests/test_scaffold_offset_sh.py \
  tests/test_scaffold_observations.py
```

These checks exercise real CPU Torch/Adam, decoded Gaussian attributes,
serialization, command parsing, and tiny COLMAP/image fixtures. They replace
CUDA allocation/KNN where needed and do not constitute full-scene GPU results.
The launcher's synthetic CUDA preflight checks the installed rasterizer and
saved-render parity on the training machine before each real run.
