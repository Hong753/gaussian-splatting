"""CPU tests: selection, occlusion, calibrated I/O and independent BA validation."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from plyfile import PlyData, PlyElement
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.keyframe_pipeline.common import (Image, image_path, project,
                                              read_gray, read_model, visible_points)
from scripts.keyframe_pipeline.selection import choose_frames, main as select_main
from scripts.keyframe_pipeline.features import build_tracks, surface_normals
from scripts.keyframe_pipeline.bundle import solve_bundle
from scripts.keyframe_pipeline.refinement import export_dataset, main as refine_main, parser as refinement_parser


def save_cloud(path, xyz):
    vertices = np.zeros(len(xyz), dtype=[(axis, "f4") for axis in ("x", "y", "z", "nx", "ny", "nz")] +
                        [(color, "u1") for color in ("red", "green", "blue")])
    for j, axis in enumerate("xyz"):
        vertices[axis] = xyz[:, j]
    for color in ("red", "green", "blue"):
        vertices[color] = 128
    PlyData([PlyElement.describe(vertices, "vertex")]).write(path)


@pytest.fixture
def scene(tmp_path):
    source = tmp_path / "source"
    model = source / "sparse/0"
    model.mkdir(parents=True)
    (source / "images").mkdir()
    (model / "cameras.txt").write_text("1 SIMPLE_PINHOLE 160 120 120 80 60\n")
    rows = []
    rng = np.random.default_rng(10)
    for i in range(12):
        name = f"frame {i:03d}.png"
        cv2.imwrite(str(source / "images" / name), rng.integers(0, 256, (120, 160), dtype=np.uint8))
        rows.append(f"{i + 1} 1 0 0 0 {-i * .05} 0 0 1 {name}\n\n")
    (model / "images.txt").write_text("".join(rows))
    x, y = np.meshgrid(np.linspace(-2, 2, 50), np.linspace(-1.5, 1.5, 40))
    save_cloud(model / "points3D.ply", np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 3.))))
    return source


def test_greedy_prefers_new_surface_and_direction_over_duplicates():
    def obs(patches, bins):
        return np.array(patches), np.array(bins), np.ones(len(patches))
    observations = [obs([0, 1], [0, 0]), obs([0, 1], [0, 0]),
                    obs([2, 3], [0, 0]), obs([0, 1], [1, 1])]
    selected, coverage = choose_frames(observations, np.ones(4), 3, [0], 4)
    assert selected == [0, 2, 3]
    np.testing.assert_array_equal(coverage, [2, 2, 1, 1])
    selected, _ = choose_frames(observations[:2], np.ones(2), 2, [0], 4)
    assert selected == [0]  # Never pad a budget with exact duplicates.


def test_visibility_excludes_back_surface_and_uses_principal_point():
    image = Image(1, 1, "a.png", np.eye(3), np.zeros(3))
    points = np.array([[0, 0, 2], [0, 0, 4], [.4, 0, 2], [0, 0, -1]])
    ids, uv, depth = visible_points(points, image, np.array([100., 100., 17., 23.]), (80, 100))
    np.testing.assert_array_equal(ids, [0, 2])
    np.testing.assert_allclose(uv[0], [17, 23])
    np.testing.assert_allclose(depth, 2)


def test_selection_manifest_eval_reuse_and_source_unchanged(scene, tmp_path):
    original = {p: p.read_bytes() for p in scene.rglob("*") if p.is_file()}
    output = tmp_path / "selected"
    argv = ["--source", str(scene), "--output", str(output), "--target-frames", "5", "--max-gap", "4", "--eval"]
    path = select_main(argv)
    manifest = json.loads(path.read_text())
    assert manifest["selected_count"] == 5
    assert manifest["observed_patch_count"] > 0
    assert not set(manifest["selected_images"]) & {"frame 000.png", "frame 008.png"}
    assert (output / "image_list.txt").read_text().splitlines() == manifest["selected_images"]
    assert select_main(argv) == path
    for p, content in original.items():
        assert p.read_bytes() == content
    with pytest.raises(SystemExit):
        select_main([*argv, "--target-frames", "6"])
    # Compatibility with the existing training launcher, without CUDA imports.
    result = subprocess.run([sys.executable, str(ROOT / "scripts/train.py"), "--data-root", str(scene),
                             "--scenes", ".", "--output-root", str(tmp_path / "training"),
                             "--keyframe-manifest", str(path), "--no-exposure-compensation", "--eval", "--dry-run"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--keyframe_manifest" in result.stdout


def test_bad_image_size_and_unsafe_names(scene):
    _, cameras, images, _ = read_model(scene)
    cameras[1].width += 1
    with pytest.raises(ValueError, match="image size"):
        read_gray(scene, "images", images[0], cameras[1], 960)
    with pytest.raises(ValueError, match="Invalid COLMAP"):
        image_path(scene, "images", "../escape.png")


def test_surface_normals_distinguish_plane_and_line():
    x, y = np.meshgrid(np.arange(15) * .04, np.arange(15) * .04)
    plane = np.column_stack((x.ravel(), y.ravel(), np.ones(x.size)))
    normals, valid = surface_normals(plane, .3)
    assert valid.all()
    np.testing.assert_allclose(np.abs(normals[:, 2]), 1)
    line = np.column_stack((np.arange(50) * .01, np.zeros(50), np.ones(50)))
    _, valid_line = surface_normals(line, .3)
    assert not valid_line.any()


def test_track_merge_rejects_two_features_from_same_image():
    feature = dict(xyz=np.array([[0., 0, 3], [.1, 0, 3]]), normals=np.array([[0., 0, 1]] * 2))
    pairs = [(0, 1, np.array([[0, 0]])), (1, 2, np.array([[0, 0]])),
             (2, 0, np.array([[0, 1]]))]
    tracks = build_tracks([feature] * 3, pairs)
    assert len(tracks) == 1
    assert tracks[0]["obs"] == [(0, 0), (1, 0), (2, 0)]


def synthetic_bundle(perturb=True):
    rng = np.random.default_rng(42)
    K = np.array([500., 500., 320., 240.])
    truth = np.column_stack((rng.uniform(-.8, .8, 70), rng.uniform(-.6, .6, 70), np.full(70, 3.)))
    images, features, true_centers = [], [], []
    for i in range(5):
        Ctrue = np.array([i * .18 - .36, .03 * np.sin(i), 0.])
        Rtrue = Rotation.from_rotvec([0., -.02 * (i - 2), 0.]).as_matrix()
        R = Rotation.from_rotvec(rng.normal(0, .001, 3)).as_matrix() @ Rtrue if perturb else Rtrue
        C = Ctrue + rng.normal(0, .007, 3) if perturb else Ctrue
        uv, _ = project(truth, Rtrue, Ctrue, K)
        rays = np.column_stack(((uv - K[2:]) / K[:2], np.ones(len(uv)))) @ R
        anchor = C + ((3. - C[2]) / rays[:, 2])[:, None] * rays
        features.append(dict(uv=uv, xyz=anchor, normals=np.tile([0., 0, 1], (len(truth), 1)), K=K, scale=np.ones(2)))
        images.append(Image(i + 1, 1, f"view{i}.png", R, C))
        true_centers.append(Ctrue)
    # No observations in this camera: must remain unchanged.
    images.append(Image(6, 1, "unobserved.png", np.eye(3), np.array([9., 9., 9.])))
    features.append(dict(K=K, scale=np.ones(2), uv=np.empty((0, 2))))
    tracks = [dict(obs=[(i, k) for i in range(5)]) for k in range(len(truth))]
    args = refinement_parser().parse_args(["--source", ".", "--output", ".", "--keyframes", "fake.json",
                                           "--min-observations", "15", "--max-steps", "60"])
    return images, features, tracks, args, np.array(true_centers)


def test_bundle_improves_unseen_observations_and_freezes_unsupported_camera():
    images, features, tracks, args, _ = synthetic_bundle()
    R, C, X, report = solve_bundle(images, features, tracks, args)
    assert report["accepted"], report
    assert report["validation_after"]["median"] < report["validation_before"]["median"] * .2
    np.testing.assert_allclose(X[:, 2], 3., atol=.002)
    np.testing.assert_array_equal(R[-1], images[-1].R)
    np.testing.assert_array_equal(C[-1], images[-1].C)
    assert report["max_translation"] < args.max_translation


def test_bundle_declines_when_poses_are_already_consistent():
    images, features, tracks, args, _ = synthetic_bundle(perturb=False)
    _, _, _, report = solve_bundle(images, features, tracks, args)
    assert not report["accepted"]
    assert "did not improve" in report["rejection_reasons"][0]


def test_export_binary_tracks_and_all_original_cameras(scene, tmp_path):
    model, cameras, original, _ = read_model(scene)
    selected = original[:3]
    R = np.array([im.R for im in selected])
    C = np.array([im.C for im in selected]) + [.001, 0, 0]
    X = np.array([[.1, .1, 3.]])
    features = [dict(uv=project(X, im.R, im.C, cameras[1].params)[0], scale=np.ones(2)) for im in selected]
    tracks = [dict(obs=[(i, 0) for i in range(3)])]
    output = tmp_path / "refinement"
    output.mkdir()
    manifest = dict(selected_images=[im.name for im in selected])
    dataset = export_dataset(output, scene, "images", model / "points3D.ply", cameras, original,
                             selected, features, tracks, R, C, X, manifest)
    _, exported_cameras, exported, _ = read_model(dataset)
    assert len(exported) == len(original)
    np.testing.assert_allclose(exported[0].C, C[0])
    np.testing.assert_allclose(exported[-1].C, original[-1].C)
    np.testing.assert_allclose(exported_cameras[1].params, cameras[1].params)
    assert (dataset / "images").is_symlink()
    assert (dataset / "sparse/0/points3D.ply").resolve() == model / "points3D.ply"
    # Independently read with the actual training-side COLMAP parser.
    spec = importlib.util.spec_from_file_location("training_colmap_loader", ROOT / "scene/colmap_loader.py")
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    records = loader.read_extrinsics_binary(dataset / "sparse/0/images.bin")
    assert records[selected[0].id].name == selected[0].name
    assert records[selected[0].id].point3D_ids.tolist() == [1]
    xyz, _, _ = loader.read_points3D_binary(dataset / "sparse/0/points3D.bin")
    np.testing.assert_allclose(xyz, X)
    with pytest.raises(ValueError, match="overwrite"):
        export_dataset(output, scene, "images", model / "points3D.ply", cameras, original,
                       selected, features, tracks, R, C, X, manifest)


def test_rendered_images_selection_matching_refinement_and_export(tmp_path):
    source = tmp_path / "rendered"
    model = source / "sparse/0"
    model.mkdir(parents=True)
    (source / "images").mkdir()
    (model / "cameras.txt").write_text("1 PINHOLE 400 300 350 350 200 150\n")
    rng = np.random.default_rng(151)
    texture = rng.integers(0, 256, (800, 1200), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), .65)
    # Texture pixels -> metric plane at z=3. COLMAP K uses half-pixel centers.
    texture_to_plane = np.array([[6 / 1200, 0, -3], [0, 4 / 800, -2], [0, 0, 1.]])
    Kcv = np.array([[350., 0, 199.5], [0, 350., 149.5], [0, 0, 1.]])
    rows = []
    for i in range(8):
        Ctrue = np.array([i * .09 - .3, .02 * np.sin(i), 0])
        Rtrue = Rotation.from_rotvec([0, .015 * (i - 3), 0]).as_matrix()
        H = Kcv @ np.column_stack((Rtrue[:, 0], Rtrue[:, 1], Rtrue @ (np.array([0., 0, 3]) - Ctrue)))
        gray = cv2.warpPerspective(texture, H @ texture_to_plane, (400, 300))
        name = f"frame{i:03d}.png"
        cv2.imwrite(str(source / "images" / name), gray)
        R = Rotation.from_rotvec(rng.normal(0, .001, 3)).as_matrix() @ Rtrue
        C = Ctrue + rng.normal(0, .005, 3)
        q = Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]
        values = " ".join(str(x) for x in [*q, *(-R @ C)])
        rows.append(f"{i + 1} {values} 1 {name}\n\n")
    (model / "images.txt").write_text("".join(rows))
    x, y = np.meshgrid(np.arange(-2.5, 2.5, .025), np.arange(-1.8, 1.8, .025))
    save_cloud(model / "points3D.ply", np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 3.))))
    manifest = select_main(["--source", str(source), "--output", str(tmp_path / "selection"),
                            "--target-frames", "6", "--max-gap", "4"])
    output = tmp_path / "refined"
    argv = ["--source", str(source), "--keyframes", str(manifest), "--output", str(output),
            "--voxel-size", ".025", "--max-features", "1000", "--max-tracks", "200",
            "--min-pair-matches", "10", "--min-observations", "15"]
    dataset = refine_main(argv)
    report = json.loads((output / "refinement_report.json").read_text())
    assert report["accepted"], report
    assert report["validation_after"]["median"] < report["validation_before"]["median"]
    assert dataset.is_dir()
    assert len(read_model(dataset)[2]) == 8
    assert refine_main(argv) == dataset
