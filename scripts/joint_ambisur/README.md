# Orange4F joint Gaussian and mesh training

Run from the Orange4F checkout on the CUDA host. The default source is `/data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full`. The default output is `/data/3dgs_outputs/orange4f/3dgs_finetuning/ambisur_joint_mesh_w1600`.

```bash
conda activate 3dgs &&
cd ~/gaussian-splatting-orange4f &&
git fetch origin experiment/orange4f-joint-mesh &&
git switch --detach FETCH_HEAD &&
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.orange4f_joint_mesh --install-missing
```

The launcher verifies the two CUDA renderers on a tilted plane and checks that mesh gradients reach Gaussian means, scales, rotations and signed values. It reuses the active Torch and simple-knn installation. Missing DA3 dependencies and the two rasterizers can be installed with `--install-missing`. Training requires CUDA, the CUDA build toolchain for missing extensions, ffmpeg with libx264, and access to the source repositories and DA3 weights on first use. The actual CUDA check results go in `gpu_check.json`.

1. Read all registered pinhole cameras from the supplied COLMAP model. Keep these extrinsics and intrinsics fixed. Fit the sharpest image per natural-order group of three by default. An explicit `--keyframes manifest.json` can specify the fitting images. Render all cameras at up to 1,600 pixels wide. Images are read in place.
2. Run pose-conditioned DA3 on overlapping windows of fitting images. Check confidence and depth reprojection against two additional fitting views. Initialize up to 500,000 colored Gaussians from the accepted depth samples. The cache records which views and settings produced it. This is a pretrained-prior experiment. The supplied poses can incorporate all frames during the original bundle adjustment.
3. Optimize the pinned AmbiSuR trainer. It retains its RGB, flattening, ray-color, normal, multiview and ambiguity objectives. Training streams RGB and priors on demand. The adapter uses calibrated camera-z rays for plane depth, a multiview-validated prior mask, 8,192 sampled NCC patches, explicit masking of ambiguity gradients, and fixed-image previews.
4. At step 15,000 Gaussian densification has ended. Sample up to 32,000 Gaussian sites by opacity and generate a center and eight oriented pivots per site. Initialize the signed field using Gaussian and trusted DA3 depths from up to 256 fitting views. Tetrahedralize with SciPy and extract a connected surface with the pinned MILo marching-tetrahedra implementation. Site selection, spacing, field updates and losses are specific to this integration.
5. On every fourth step, rasterize the mesh with nvdiffrast. Compare positive first-hit camera-z and face normals with AmbiSuR. Fit trusted DA3 depths, signed samples, antialiased coverage and neighboring fitting-view RGB sampled at mesh reprojections. The reverse depth term updates the Gaussian rendering on rays where the mesh agrees with the independent prior. Gaussian RGB training continues between mesh updates. Every 1,000 steps refresh connectivity and admit up to 512 visible Gaussian sites for supported missing or misplaced surfaces, with 8,192 extra sites budgeted. The signed values and Gaussian geometry both receive mesh gradients. Unobserved regions and details smaller than the site spacing can remain missing.
6. Save the optimized mesh directly, with colors interpolated from view-independent Gaussian SH coefficients. Render Gaussian RGB along the original camera trajectory and a seeded, smoothly displaced 40-second path at 30 fps. A third video shows mesh normals on the same displaced path. The path uses the capture bounds and camera distances and may intersect objects.

## Watching and evaluating

The terminal prints the live URL, normally `http://127.0.0.1:8112/`. Forward that port through SSH when viewing from another machine. If the port is occupied, use the port in `preview_url.txt`. The self-contained HTML also works directly from a mounted output directory. The built-in HTTP server runs for the lifetime of the training command.

Live previews are saved at step 1, every 500 steps, checkpoints and the last step. The page links the history and publishes four fixed fitting and four fixed held-out views when available. During joint training, panels include GT RGB, Gaussian RGB, amplified error, Gaussian depth, mesh depth and mesh normals. Preview PSNR is a mean over the displayed views.

Run a full evaluation during training on a second GPU. Select a checkpoint already published in the terminal. The 7k checkpoint has Gaussian outputs. The default 15k, 20k and 30k checkpoints also contain the optimized mesh.

```bash
conda activate 3dgs &&
cd ~/gaussian-splatting-orange4f &&
CUDA_VISIBLE_DEVICES=1 python -u -m scripts.orange4f_joint_mesh --evaluate 20000
```

To continue after interruption, run the same launcher with `--resume`. It restores the latest atomically published checkpoint, both optimizers, signed values, pivot indices, tetrahedra, camera sampling stack and RNG state. If the final checkpoint already exists, it can regenerate the final evaluation. A failed attempt before checkpoint 1 can restart with the original command and reuse the completed fitting priors.

| Output relative to the run directory | Contents |
|---|---|
| `index.html`, `previews/STEP/` | Live page, historical images and sample PSNR |
| `cameras.json`, `config.json`, `da3/config.json` | Fitting and held-out assignments, fixed poses, random path, training configuration and cache settings |
| `gpu_check.json`, `commands.log`, `timings.jsonl`, `da3.log`, `train.log` | Native smoke check, commands, elapsed time and console logs |
| `mesh_training.jsonl` | Geometry sample counts, losses, mesh coverage and cumulative tetrahedralization time |
| `checkpoints/step_STEP.pt`, `checkpoints/mesh_STEP.ply` | Resumable training state and direct mesh at joint checkpoints |
| `model/gaussians.ply`, `final/mesh.ply` | Final Gaussian model and colored triangle mesh |
| `final/index.html`, `final/metrics.json` | Browsable per-camera comparisons, individual PSNR and mean per-image train/test/all PSNR at rendering resolution, before output encoding |
| `final/frames/`, `final/comparisons/` | One RGB PNG and one labeled GT/render/error panel per registered frame, in flat folders |
| `final/original.mp4`, `final/random.mp4`, `final/random_mesh.mp4` | Recorded trajectory, displaced trajectory and mesh normals on the displaced trajectory |
| `evaluation_STEP/` | Independently rendered checkpoint with the same split and cameras |

## Revisions and verification

AmbiSuR is pinned to `88d64054f53e0eba9ce49198282edf4a67fc8ca8`. MILo is pinned to `0c38aa65f8b768513964f13246ef1ae54f001c70`. The launcher snapshots the trainer, applies `ambisur.patch`, and loads MILo's standalone marching tetrahedra module from that revision. The Gaussian rasterizer and the mesh rasterizer keep separate visibility computations. The exported triangle surface is the one used by the joint objective.

For CPU regression tests, set `TEST_AMBISUR` and `TEST_MILO` to checkouts containing those commits, then run `python -m pytest tests/test_orange4f_joint_mesh.py` with Torch, SciPy, Pillow, plyfile and ffmpeg available. The tests cover the fit-only data boundary using a fixture prior, depth-sensitive photometric gradients, moving Gaussian pivots and signed values, supported topology growth and optimizer state, and the actual preview, PLY and video writers. For the native CUDA check alone, use `python -u -m scripts.orange4f_joint_mesh --stage check --install-missing`.
