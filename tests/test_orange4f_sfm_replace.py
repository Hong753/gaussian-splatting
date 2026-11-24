import numpy as np
import pytest

from scripts.orange4f_learned_sfm import BINARIES, MARKER, publish
from scripts.orange4f_learned_matches import pairs, verified


def test_publish_keeps_old_training_inputs_and_whole_backup(tmp_path):
    scene = tmp_path / "sfm"
    old = scene / "sparse/0"
    stage = tmp_path / "gs/scene/sparse/0"
    candidate = scene / "job/candidate"
    for p in (old, stage, candidate / "0"):
        p.mkdir(parents=True)
    for f in BINARIES:
        (old / f).write_bytes(b"old")
        (stage / f).symlink_to(old / f)
        (candidate / "0" / f).write_bytes(b"new")
    (candidate / MARKER).write_text("{}")
    publish(scene, candidate)
    for f in BINARIES:
        assert (old / f).read_bytes() == b"new"
        assert (stage / f).read_bytes() == b"old"
        assert (scene / "sparse_sift_backup/0" / f).samefile(stage / f)


def test_nonlocal_pairs_and_planar_verification():
    v = np.eye(75)
    v[55] = v[0]
    result = pairs(v)
    assert (0, 55) in result and (0, 1) in result and len(set(result)) == len(result)
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 1000, (60, 2)).astype("f")
    a = dict(keypoints_colmap=points, original_size=[1600, 1200])
    b = dict(a, keypoints_colmap=points + [20, -5])
    indices = np.arange(len(points), dtype=np.uint32)
    kind, kept = verified(a, b, np.column_stack([indices, indices]))
    assert kind == 3 and len(kept) == 60


def test_scan_bundle_recovers_perturbed_extrinsics(tmp_path):
    c = pytest.importorskip("pycolmap")
    if c.__version__ != "3.11.1":
        pytest.skip("Spirula adapter targets pycolmap 3.11.1")
    from plyfile import PlyData, PlyElement
    from scripts.orange4f_scan_bundle import refine
    from scripts.view_comparisons import read_colmap_records

    opts = c.SyntheticDatasetOptions()
    opts.camera_model_id = c.CameraModelId.PINHOLE
    opts.camera_params = [1024, 1024, 512, 384]
    opts.num_cameras, opts.num_images, opts.num_points3D = 1, 10, 300
    c.set_random_seed(23)
    rec = c.synthesize_dataset(opts)
    truth = tmp_path / "original/sparse/0"
    truth.mkdir(parents=True)
    rec.write_binary(str(truth))
    records = read_colmap_records(tmp_path / "original")
    rng = np.random.default_rng(41)
    xyz = np.stack([p.xyz for p in rec.points3D.values()])
    jitter = rng.uniform(-0.015, 0.015, (len(xyz), 25, 3))
    jitter[np.arange(len(xyz)), :, np.arange(len(xyz)) % 3] = 0
    scan = (xyz[:, None, :] + jitter).reshape(-1, 3)
    vertices = np.empty(len(scan), dtype=[(s, "f4") for s in "xyz"])
    for j, s in enumerate("xyz"):
        vertices[s] = scan[:, j]
    ply = tmp_path / "scan.ply"
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(ply))
    reference = {i: rec.image(i).projection_center().copy() for i in rec.reg_image_ids()}
    for i in rec.reg_image_ids():
        image = rec.image(i)
        image.cam_from_world = c.Rigid3d(
            image.cam_from_world.rotation, image.cam_from_world.translation + rng.normal(0, 0.008, 3)
        )
    perturbed = tmp_path / "perturbed"
    perturbed.mkdir()
    rec.write_binary(str(perturbed))
    out = tmp_path / "refined"
    result = refine(perturbed, ply, records, out)
    refined = c.Reconstruction(str(out))
    before = np.mean([np.linalg.norm(rec.image(i).projection_center() - q) for i, q in reference.items()])
    after = np.mean([np.linalg.norm(refined.image(i).projection_center() - q) for i, q in reference.items()])
    print(f"Injected pose drift: mean center error {before:.6f} -> {after:.6f} m")
    assert result["scan_anchors"] > 30 and result["median_camera_update_m"] > 0
    assert after < before
