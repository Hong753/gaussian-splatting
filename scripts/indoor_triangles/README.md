# Indoor triangle reconstruction reference experiment

This worktree adds a CPU reference implementation and a locked synthetic study. For an observed ScanNet++ comparison, run the stock references first and then the paired experiment with the same prepared input set. The CPU implementation targets small diagnostic reconstructions and uses an all-faces visibility pass.

## Reproduce the synthetic evidence

Run these commands from the fork root with Python 3.12, NumPy, SciPy and Pillow. LPIPS separately requires PyTorch, torchvision and `lpips==0.1.4`, plus the standard pretrained AlexNet weights. That environment is used only for evaluation.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
OUT="$(realpath -m ../triangle_evidence)"
mkdir -p "$OUT"
python -m scripts.indoor_triangles make-synthetic --out "$OUT/data"
python -m scripts.indoor_triangles check --out "$OUT/sanity.json"

python -u -m scripts.indoor_triangles fit --data "$OUT/data" --out "$OUT/sparse_seed17" --seed 17
python -u -m scripts.indoor_triangles fit --data "$OUT/data" --out "$OUT/sparse_seed29" --seed 29 --minimal
python -u -m scripts.indoor_triangles fit --data "$OUT/data" --out "$OUT/dense_seed17" --seed 17 --dense

for run in sparse_seed17 sparse_seed29 dense_seed17; do
  python -u -m scripts.indoor_triangles evaluate --data "$OUT/data" --run "$OUT/$run" --out "$OUT/${run}_eval"
  python -m scripts.indoor_triangles lpips --eval "$OUT/${run}_eval"
done
python -m scripts.indoor_triangles fit-diagnostics --checkpoint "$OUT/sparse_seed17/shared_start.npz" \
  --fit "$OUT/data/fit.npz" --out "$OUT/fit_image_coverage.json"
python -m scripts.indoor_triangles render --checkpoint "$OUT/sparse_seed17/connected_hard.npz" \
  --views "$OUT/data/final.npz" --out "$OUT/hard_render"
python -m scripts.indoor_triangles export --checkpoint "$OUT/sparse_seed17/connected_hard.npz" \
  --threshold 0.90 --out "$OUT/hard_export"
```

The fitter reads only fitting and edit-validation RGB and camera arrays. The room generator keeps the procedural triangles and material IDs in a separate evaluator file. The synthetic rig uses fixed identity world-to-camera rotations with x right, y down, z forward, and normalized rays with metric range. Final views are 0.27 m and 0.28 m from the closest fitting center. Atlas UVs are assigned from a fitting-camera projection. Each trainable texel stores three FP64 channels in this reference implementation.

The sparse initialization uses rectified, multiview patch correspondences at corner tracks. The matched denser initialization also selects reliable stereo seeds at non-corner pixels and uses closer spatial sampling. The two initialization cohorts have different geometry sizes. The full candidate uses image-space Delaunay with gates on screen edge length, world edge length, and triangle aspect, then welds initially private faces through known fitting-track indices. This differs from restricted Delaunay in 3D and reconstructs only one reference chart. Both controlled arms start from an identical checkpoint and reset Adam identically. They use the same fixed fitting-track support policy, texture, hyperparameters, randomized pixels and trial budget of zero. The global arm hardens all nontransmitting faces at that same branch point.

Range comes from per-sample ray–plane intersections. Welded edges get deterministic exact ownership, and committed opacity is always one on the interior. Each sample composites positive events in range order. The CPU implementation evaluates the surrogate geometry derivative by paired finite-displacement probes with shared subpixel jitter. It is sensitive to sample count and does not implement an exact visibility-boundary gradient, GPU rasterization, mipmapping, or scan-scale topology. The appearance gradient is analytic. Warps gate low variance, saturation, view translation, support at five reference and target patch locations and same-face visibility before computing robust ZNCC. Track free-space uses actual transmittance weights, and missing anchor assignments pay the capped penalty. Bending uses a frozen initial per-phase dihedral gate, with zero coupling beyond 28 degrees. Stage transitions reset Adam. The Gaussian color control has fixed chart-lattice RBF locations and widths, equal trainable color bytes and fixed hard-hit geometry. The analytic pane is supported by the compositor. The fitted arms lack a translucency classifier.

## Room preparation and candidate commands

Prefer the first available suitable room among `8b5caf3398`, `b20a261fdf`, `66c98f4a9b`, and `88cf747085`, in this order, after inspecting fitting RGB coverage and checking that a co-registered scan exists. Keep the reference scan out of fitting and choice of hyperparameters. Record the exact release and pose files. Use the official [ScanNet++ DSLR undistortion](https://github.com/scannetpp/scannetpp) if the supplied cameras are fisheye, and export the calibration as an undistorted PINHOLE COLMAP model. Original transforms can have been estimated from all views. The procedure preserves those calibration parameters and rebuilds the actual feature tracks from fitting images only. The measured integration preflight used synthetic fixed calibration.

Create a JSON manifest with keys `fit`, `edit`, and `final`, each listing original relative image filenames in the desired order. Use translated camera blocks and select a fitting reference view from RGB and camera coverage. The adapter enforces at least 0.15 m final-to-fit and 0.08 m edit-to-fit translation. A full room needs a registered scan in the supplied camera frame, with distance units in metres and any registration fixed before scoring. Record the alignment provenance beside the results.

```bash
python -m pip install pycolmap==3.11.1 open3d
python -m scripts.indoor_triangles prepare-room \
  --model "$PINHOLE_COLMAP" --images "$UNDISTORTED_RGB" --split "$SPLIT_JSON" \
  --reference-view "$FIT_REFERENCE_IMAGE" --pose-provenance "$POSE_PROVENANCE" \
  --width 240 --cell 18 --out "$ROOM_INPUT"

CANDIDATE_CONFIG=scripts/indoor_triangles/room_config.json
python -u -m scripts.indoor_triangles fit --data "$ROOM_INPUT" --out "$ROOM_RUN" \
  --tracks "$ROOM_INPUT/fitting_tracks.npz" --config "$CANDIDATE_CONFIG" --seed 17
python -u -m scripts.indoor_triangles evaluate --data "$ROOM_INPUT" --run "$ROOM_RUN" \
  --out "$ROOM_EVAL" --reference-mesh "$REGISTERED_SCAN" --reference-masks "$EVALUATOR_MASKS"
python -m scripts.indoor_triangles lpips --eval "$ROOM_EVAL"
```

The optional evaluator mask NPZ uses `kinds` of shape (final views, height, width), with 0 for floor, 1 for wall, 2 for doorway rear targets and −1 for unspecified. Without it only whole-scene metrics and RGB scores are defined. Set mask labels from evaluator geometry and view definitions independent of any candidate. Use repeating locked scene manifests. Classical stereo has a measured synthetic arm only. Orange4F has no run in this branch.

## Stock CUDA references

Pinned Git revisions, inspected here, have separate original renderers and shader capacities.

| Reference | Revision | Surface path |
|---|---|---|
| [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) | `54c035f7834b564019656c3e3fcc3646292f727d` | Fit-view depth TSDF adapter below |
| [Triangle Splatting](https://github.com/trianglesplatting/triangle-splatting) | `685ff270a8273afc3a2b094ae3148dd0b98e31e9` | Official fit-only bounded TSDF and its separate cleaned export |
| [MeshSplatting](https://github.com/meshsplatting/mesh-splatting) | `2a810a6c353215685307da3d4cc6ebd73b1c387b` | Direct PLY from final checkpoint |

Use the above revisions, their submodules and separate CUDA environments. Invoke from each reference repository root. Paths for both stock inputs and all models should be absolute. Verify that `stock_centered_camera_compatible` is true in the room manifest before using these stock loaders. With off-center principal points, a common fixed crop/reprojection or an explicit calibrated adapter is needed. The stock train source contains only fitting RGB and points. Render the separate final-view source at evaluation. Keep final-view renders in the original model's ordered camera manifest, with filename order checked against the staged undistorted images.

```bash
FIT="$ROOM_INPUT/stock_fit"
FINAL="$ROOM_INPUT/stock_final"
ITER=30000

# From the stock 3DGS checkout.
time python train.py -s "$FIT" -m "$GS_RUN" -r 1 --iterations "$ITER"
python render.py -m "$GS_RUN" -s "$FINAL" -r 1 --iteration "$ITER" --skip_test
# From the fork root, using the stock 3DGS CUDA environment with Open3D.
python -m scripts.indoor_triangles.stock_3dgs_tsdf --repo "$GS_REPO" --source "$FIT" \
  --model "$GS_RUN" --iteration "$ITER" --out "$GS_RUN/fitting_tsdf.ply"

# From stock Triangle Splatting.
time python train.py -s "$FIT" -m "$TRI_RUN" -r 1 --iterations "$ITER"
python render.py -m "$TRI_RUN" -s "$FINAL" -r 1 --iteration "$ITER" --skip_test
time python mesh.py -m "$TRI_RUN" -s "$FIT" -r 1 --iteration "$ITER" \
  --skip_train --skip_test --voxel_size 0.025 --sdf_trunc 0.075 --depth_trunc 12

# From stock MeshSplatting. Keep optional depth and normal prior assets absent.
time python train.py -s "$FIT" -m "$MESH_RUN" -r 1 --iterations "$ITER"
python render.py -m "$MESH_RUN" -s "$FINAL" -r 1 --iteration "$ITER" --skip_test
time python create_ply.py "$MESH_RUN/point_cloud/iteration_$ITER" \
  --out "$MESH_RUN/direct.ply" --cpu
```

MeshSplatting's `--indoor` preset also enables an optional supervised normal weight; verify the exact preset and input prior flags before using it for another labeled arm. OMeGa's published geometry results use different input guidance and are a prior-assisted reference. [OMeGa paper](https://openaccess.thecvf.com/content/WACV2026/papers/Cao_OMeGa_Joint_Optimization_of_Explicit_Meshes_and_Gaussian_Splats_for_WACV_2026_paper.pdf)

Score the stock direct/TSDF and postprocessed meshes separately:

```bash
python -m scripts.indoor_triangles score-external --data "$ROOM_INPUT" \
  --reference-mesh "$REGISTERED_SCAN" --reference-masks "$EVALUATOR_MASKS" \
  --mesh "$GS_RUN/fitting_tsdf.ply" --renders "$GS_RUN/train/ours_$ITER/renders" \
  --config scripts/indoor_triangles/room_config.json --out "$GS_RUN/paired_evaluation.json"
```

The experimental 3DGS adapter conditions the rasterizer's accumulated inverse depth by alpha before TSDF at 2.5 cm voxels, 7.5 cm truncation, 12 m maximum z and alpha at least 0.55. Its time and memory are tracked separately. These stock and real-room commands were inspected but not executed on CUDA or ScanNet++ in this workspace.
