import json

import cv2
import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from scripts import orange4f_floor_neighbors as audit
from scripts.keyframe_pipeline.common import Camera, Image


@pytest.mark.parametrize("offset,expected", [(0, "aligned"), (3, "coherent shift")])
def test_nearby_training_photos_have_known_native_pixel_shifts(tmp_path, offset, expected):
    suite = tmp_path/"stride3_noaa_v1"
    suite.mkdir()
    sparse = tmp_path/"sparse/0"
    sparse.mkdir(parents=True)
    folder = tmp_path/"images"
    folder.mkdir()
    width, height, focal, depth = 960, 720, 600, 3.
    (sparse/"cameras.txt").write_text(f"1 PINHOLE {width} {height} {focal} {focal} 480 360\n")
    rng = np.random.default_rng(463)
    texture = cv2.GaussianBlur(rng.integers(15, 235, (height,width), np.uint8), (3,3), .65)
    cameras, train = [], []
    for i in range(6):
        name = f"{9882+i:05}.jpg"
        cameras.append(f"{i+1} 1 0 0 0 {i*.03} 0 0 1 {name}\n\n")
        if i == 5:  # Outside the training split. Pixel data deliberately unavailable.
            continue
        train.append(name)
        warp = np.float32([[1,0,6*i],[0,1,offset if i else 0]])
        image = cv2.warpAffine(texture,warp,(width,height))
        if i == 1:
            image = cv2.GaussianBlur(image, (11,11), 3.)
        assert cv2.imwrite(str(folder/name), image, [cv2.IMWRITE_JPEG_QUALITY,100])
    (sparse/"images.txt").write_text("".join(cameras))
    yy, xx = np.mgrid[0:height:4, 0:width:4]
    xyz = np.stack(((xx+.5-480)*depth/focal, (yy+.5-360)*depth/focal, np.full_like(xx,depth)), axis=-1)
    vertices = np.array(list(map(tuple,xyz.reshape(-1,3))), dtype=[(k,"f4") for k in "xyz"])
    cloud = tmp_path/"input.ply"
    PlyData([PlyElement.describe(vertices, "vertex")]).write(cloud)
    manifest = suite/"keyframes.json"
    manifest.write_text(json.dumps({"selected_images": train}))
    (suite/"suite.json").write_text(json.dumps(dict(source=str(tmp_path), cloud={"cached":str(cloud)}, selection=str(manifest), runs=[])))
    audit.main(["--suite", str(suite)])
    page = suite/"floor_neighbors_09882_v1"
    report = json.loads((page/"report.json").read_text())
    assert (page/"index.html").is_file()
    assert 'href="floor_neighbors_09882_v1/index.html"' in (suite/"index.html").read_text()
    assert report["views"][0]["status"] == "soft warped texture"
    assert len(report["views"]) == 4
    assert "Inconclusive" not in report["verdict"]
    for row in report["views"][1:]:
        assert row["status"] == expected
        assert row["shift_xy_px"] == pytest.approx([0,offset], abs=.2)
        assert row["matches"] >= 12
        assert (page/row["overlay"]).is_file()


def test_visibility_requires_floor_support_without_nearer_return_in_each_camera():
    camera = Camera(1, 64, 64, np.array([50.,50.,32.,32.]))
    ref = Image(1, 1, "ref", np.eye(3), np.zeros(3))
    source = Image(2, 1, "other", np.eye(3), np.array([1.,0.,0.]))
    # The blocker projects onto the floor patch only in the reference image.
    xyz = np.array([[0.,0.,3.], [0.,0.,1.]])
    normal = np.array([0.,0.,1.])
    for view, visible in [(ref, False), (source, True)]:
        uv, _ = audit.project(xyz[:1], view.R, view.C, camera.params)
        xy = (uv-.5).reshape(1,1,2)
        mask = audit.visible_floor(xyz, xyz[0], normal, view, camera, xy, np.ones((1,1), bool))
        assert mask.item() == visible
        empty = audit.visible_floor(xyz[:0], xyz[0], normal, view, camera, xy, np.ones((1,1), bool))
        assert not empty.item()
    assert "Inconclusive" in audit.verdict([dict(status="weak or ambiguous")]*4)


def test_geometry_screen_rejects_distant_low_resolution_sample():
    cam = Camera(1, 640, 480, np.array([500., 500., 320., 240.]))
    ref = Image(1, 1, "ref", np.eye(3), np.zeros(3))
    far = Image(2, 1, "far", np.eye(3), np.array([0., 0., -3.]))
    yy, xx = np.mgrid[0:70, 0:120]
    world = np.stack((xx/500, yy/500, np.ones_like(xx)), axis=-1)
    selected, rejected = audit.candidates([ref,far], {1:cam}, 0, {"ref","far"}, world)
    assert selected == []
    assert rejected["different pixel scale"] == 1
