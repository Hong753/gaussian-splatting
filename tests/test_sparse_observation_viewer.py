import json
from pathlib import Path
import struct
from threading import Thread
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
import pytest

from scripts.view_sparse_observations import Inspector, ThreadingHTTPServer, handler


def fixture(source, populated=True):
    model = source / "sparse/0"
    model.mkdir(parents=True)
    (source / "images").mkdir()
    (model / "cameras.bin").write_bytes(struct.pack("<QiiQQ4d", 1, 1, 1, 64, 48, 32, 32, 31, 23))
    # IDs are deliberately unrelated to row order. The valid large ID must survive JS JSON decoding.
    large = 2**53 + 7
    frames = [
        (11, "empty.jpg", []),
        (31, "09882.jpg", [(31.0, 23.0, large), (47.0, 23.0, 7), (12.0, 12.0, 999), (0.0, 0.0, -1)]),
    ]
    with (model / "images.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(frames)))
        for iid, name, obs in frames:
            im = Image.new("RGB", (64, 48), (50, 55, 65))
            im.save(source / "images" / name)
            f.write(struct.pack("<i7di", iid, 1, 0, 0, 0, 0, 0, 0, 1) + name.encode() + b"\0")
            obs = obs if populated else []
            f.write(struct.pack("<Q", len(obs)))
            for row in obs:
                f.write(struct.pack("<ddq", *row))
    pts = [(7, (1, 0, 2), []), (large, (0, 0, 2), [(31, 0)]), (2, (0, 0.5, 2), [(31, 3)])]
    with (model / "points3D.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(pts)))
        for pid, xyz, track in pts:
            track = track if populated else []
            f.write(struct.pack("<Q3d3BdQ", pid, *xyz, 0, 10, 100, 0, len(track)))
            for image, index in track:
                f.write(struct.pack("<ii", image, index))
    # Reordered raw PLY, duplicate matching position, plus unrelated point.
    vertices = np.array(
        [(0, 0.5, 2), (5, 3, 2), (0, 0, 2), (1, 0, 2), (1, 0, 2)], dtype=[(d, "<f4") for d in "xyz"]
    )
    PlyData([PlyElement.describe(vertices, "vertex")]).write(model / "points3D.ply")
    return model


def test_bidirectional_links_reprojection_and_exact_ply_identity(tmp_path):
    fixture(tmp_path)
    viewer = Inspector(tmp_path)
    stats = viewer.frame(31)
    assert (
        viewer.summary["initial"] == 31
        and viewer.summary["with_forward"] == 1
        and viewer.summary["with_reverse"] == 1
    )
    assert (stats["features"], stats["forward"], stats["reverse"], stats["reciprocal"]) == (4, 3, 2, 1)
    assert (stats["forward_only"], stats["reverse_only"], stats["resolved"], stats["missing_xyz"]) == (
        2,
        1,
        3,
        1,
    )
    assert (stats["matched"], stats["ambiguous"], stats["distinct"]) == (2, 1, 4)
    assert stats["reciprocal_ply"] == 1
    assert stats["median_error_px"] == 0
    markers = {s[0]: s for s in stats["samples"]}
    assert markers[str(2**53 + 7)][1:6] == [0, 3, [31, 23], [31, 23], 2]
    assert markers["7"][5] == -2 and markers["999"][4:] == [None, -1, None]
    assert markers["2"][2:6] == [2, None, [31, 31], 0]
    assert viewer.frame(11)["pairs"] == 0
    assert len(viewer.context(11)) > 0  # Context alone cannot supply a stored measurement.
    assert len(Inspector(tmp_path, limit=1).frame(31)["samples"]) == 1


def test_empty_point_tracks_and_missing_binary_and_ply(tmp_path):
    model = fixture(tmp_path, False)
    v = Inspector(tmp_path)
    assert (
        v.summary["points"] == 3
        and v.summary["forward"] == v.summary["reverse"] == 0
        and v.frame(31)["pairs"] == 0
    )
    (model / "points3D.bin").unlink()
    (model / "points3D.ply").unlink()
    v = Inspector(tmp_path)
    assert not v.summary["point_file"] and v.summary["ply"] is None and v.context(11) == []
    assert v.frame(31)["samples"] == []


def test_http_serves_registered_frame_and_overlay(tmp_path):
    fixture(tmp_path)
    inspector = Inspector(tmp_path)
    with ThreadingHTTPServer(("127.0.0.1", 0), handler(inspector)) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            assert b"Stored image pixels" in urlopen(url + "/").read()
            assert json.load(urlopen(url + "/summary"))["initial"] == 31
            assert json.load(urlopen(url + "/frame/31"))["matched"] == 2
            assert b"drawElements" in urlopen(url + "/sparse_scene.js").read()
            assert len(urlopen(url + "/cloud/0").read()) == 5 * 12
            assert urlopen(url + "/indexed/31").read() == inspector.frame_data(31)[1]
            assert urlopen(url + "/indexed/11").read() == bytes(8)
            assert urlopen(url + "/photo/31").read().startswith(b"\xff\xd8")
            assert len(json.load(urlopen(url + "/context/31"))) > 0
            with pytest.raises(HTTPError):
                urlopen(url + "/photo/999999")
        finally:
            server.shutdown()
            thread.join()


def test_distorted_camera_still_displays_saved_feature_pixels(tmp_path):
    model = fixture(tmp_path)
    (model / "cameras.bin").write_bytes(struct.pack("<QiiQQ4d", 1, 1, 2, 64, 48, 32, 31, 23, 0.02))
    viewer = Inspector(tmp_path)
    f = viewer.frame(31)
    assert not f["can_project"] and f["camera_model"] == "SIMPLE_RADIAL"
    assert f["samples"][0][3:5] == [[31, 23], None] and f["reciprocal_ply"] == 1
    assert viewer.context(31) == []


def test_3d_scene_and_indices_use_every_original_row(tmp_path, monkeypatch):
    from scripts import view_sparse_observations as viewer

    monkeypatch.setattr(viewer, "CLOUD_CHUNK", 2)
    fixture(tmp_path)
    v = Inspector(tmp_path, limit=1)
    # Last partial chunk, duplicate vertices, and stored PLY order are preserved.
    cloud = np.frombuffer(b"".join(v.cloud_chunk(i) for i in range(3)), "<f4").reshape(-1, 3)
    np.testing.assert_allclose(cloud + v.origin, v.ply.coords(slice(None)))
    with pytest.raises(KeyError):
        v.cloud_chunk(3)
    frame, data = v.frame_data(31)
    assert struct.unpack_from("<II", data) == (2, 1)
    np.testing.assert_array_equal(np.frombuffer(data, "<u4", count=2, offset=8), [0, 2])
    np.testing.assert_allclose(np.frombuffer(data, "<f4", offset=16) + v.origin, [1, 0, 2])
    assert len(frame["samples"]) == 1 and frame["indexed_rows"] == 2
    assert v.frame_data(11)[1] == bytes(8)  # plenty of points in the frustum, none indexed
    # Off-center principal point and nonidentity COLMAP world-to-camera pose.
    pose = v.frames[11]
    pose.R = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    pose.t = np.array([2.0, -3.0, 5.0])
    g = viewer.camera_geometry(pose, v.cameras[1], v.origin)
    center = np.asarray(g["center"]) + v.origin
    np.testing.assert_allclose(pose.R @ center + pose.t, 0)
    rays = np.asarray(g["rays"])
    uv, _ = viewer.project(center + rays, pose, v.cameras[1])
    np.testing.assert_allclose(uv, [[0, 0], [64, 0], [64, 48], [0, 48]], atol=1e-10)


def test_stored_lidar_rows_override_track_matches(tmp_path):
    from utils.checkpoint_snapshots import digest

    model = fixture(tmp_path)
    path = model / "lidar_observations.json"
    source = {
        "format": "lidar_rows_v1",
        "point_cloud_sha256": digest(model / "points3D.ply"),
        "frames": {"09882.jpg": "rows.npy", "empty.jpg": "empty.npy"},
    }
    path.write_text(json.dumps(source))
    np.save(model / "rows.npy", np.array([4, 1, 4], dtype=np.uint32))
    np.save(model / "empty.npy", np.array([], dtype=np.uint32))
    v = Inspector(tmp_path, limit=1)
    f, data = v.frame_data(31)
    assert f["indexed_source"] == "per-image LiDAR row indices" and f["indexed_rows"] == 2
    assert len(f["manifest_xy"]) == 1
    np.testing.assert_array_equal(np.frombuffer(data, "<u4", count=2, offset=8), [1, 4])
    assert v.frame_data(11)[1] == bytes(8)
    np.save(model / "rows.npy", np.array([5]))
    with pytest.raises(ValueError, match="Invalid LiDAR row indices"):
        Inspector(tmp_path).frame(31)
    source["point_cloud_sha256"] = "wrong"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="different PLY"):
        Inspector(tmp_path)


def test_sfm_uses_saved_scan_pixels_across_reordered_ply(tmp_path):
    from scripts.orange4f_lidar_tracks import collect, scan_inputs
    from scripts.view_comparisons import read_colmap_records

    model = fixture(tmp_path)
    scan, points = scan_inputs(tmp_path)
    assert scan == model / "points3D.ply"
    assert points == model / "points3D.bin"
    reference = read_colmap_records(tmp_path)
    data, stats = collect(tmp_path, scan, points, reference)
    assert stats["scan_points"] == 5
    assert stats["usable_pixel_links"] == 2
    assert stats["images_with_links"] == 1
    links = dict(zip(data["pid"], data["xy"].tolist()))
    # ID 2 has only a reverse link. Its saved pixel differs from its projection.
    # The forward-only ID 7 has ambiguous duplicate coordinates in the PLY.
    assert links == {2: [0., 0.], 2**53 + 7: [31., 23.]}
    scan.unlink()
    scan, points = scan_inputs(tmp_path)
    data, stats = collect(tmp_path, scan, points, reference)
    assert set(data["pid"].tolist()) == {2, 7, 2**53 + 7}
    assert stats["usable_pixel_links"] == 3
