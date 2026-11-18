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
