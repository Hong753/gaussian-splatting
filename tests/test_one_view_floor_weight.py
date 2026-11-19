import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from scripts import orange4f_one_view_weight as probe
from scripts import render as gallery
from scripts.keyframe_pipeline.common import Camera, Image as View
from utils.region_loss import load_rgb_weight, mask_path, weighted_l1


def test_single_photo_gradient_and_loss_scale(tmp_path):
    cam = SimpleNamespace(image_name=probe.REFERENCE, image_width=2, image_height=2)
    path = mask_path(tmp_path, cam.image_name)
    Image.fromarray(np.array([[0, 255], [0, 0]], np.uint8)).save(path)
    weight = load_rgb_weight(tmp_path, cam)
    img = torch.ones(3, 2, 2, requires_grad=True)
    loss = weighted_l1(img, torch.zeros_like(img), weight)
    assert loss.item() == pytest.approx(1)
    loss.backward()
    assert (img.grad[:, 0, 1] / img.grad[:, 0, 0]).tolist() == [2.0] * 3
    cam.image_width = 3
    with pytest.raises(ValueError, match="native-resolution"):
        load_rgb_weight(tmp_path, cam)


def test_weights_use_surface_sampling_and_keep_rest_of_photo(tmp_path):
    cam = Camera(1, 100, 80, np.array([60.0, 60.0, 50.0, 40.0]))
    view = View(1, 1, probe.REFERENCE, np.eye(3), np.zeros(3))
    mask = np.ones((50, 80), bool)
    mask[5:8, :] = False
    patch = dict(view=view, cam=cam, rect=(5, 10, 80, 50), mask=mask)
    plane = dict(center=[0, 0, 2], normal=[0, -0.5, 1])
    weights, stats = probe.surface_weights(patch, plane)
    assert weights.shape == (80, 100)
    assert weights[:10].sum() == 0 and weights[15:18].sum() == 0
    assert weights[10, 20] == 255 and 0 < weights[59, 20] < 255
    assert stats["floor_weight_min_median_max"][2] == 2


def test_one_view_launcher_and_named_evaluation(tmp_path, monkeypatch):
    suite = tmp_path / "suite"
    base = suite / "3dgs"
    base.mkdir(parents=True)
    src = tmp_path / "scene"
    sparse = src / "sparse/0"
    sparse.mkdir(parents=True)
    (src / "images").mkdir()
    w, h = 120, 100
    (sparse / "cameras.txt").write_text(f"1 PINHOLE {w} {h} 80 80 60 50\n")
    names = [probe.REFERENCE, probe.NEIGHBOR, "09900.jpg"]
    (sparse / "images.txt").write_text(
        "".join(f"{i + 1} 1 0 0 0 {i * 0.1} 0 0 1 {n}\n\n" for i, n in enumerate(names))
    )
    cam = Camera(1, w, h, np.array([80.0, 80.0, 60.0, 50.0]))
    rgb = np.full((h, w, 3), 48, np.uint8)
    rgb[2::6, 3::6] = 200
    for name in names:
        Image.fromarray(rgb).save(src / "images" / name, quality=100)
    (base / "keyframes.json").write_text(json.dumps(dict(selected_images=names[:2])))
    cfg = argparse.Namespace(
        source_path=str(src),
        resolution=1,
        keyframe_manifest=str(base / "keyframes.json"),
        optimization_settings=dict(iterations=30000, percent_dense=0.01),
    )
    (base / "cfg_args").write_text(str(cfg))
    (base / "input.ply").write_bytes(b"input is read only")
    (suite / "suite.json").write_text('{"runs":[]}')
    rect = (10, 10, 80, 50)
    data = {}
    for i, name in enumerate(names[:2]):
        mask = np.ones((50, 80), bool)
        patch = np.asarray(Image.open(src / "images" / name).convert("RGB"))[10:60, 10:90]
        data[name] = dict(
            cam=cam,
            view=View(i + 1, 1, name, np.eye(3), np.array([-i * 0.1, 0.0, 0.0])),
            rect=rect,
            rgb=patch,
            mask=mask,
            cores=patch[..., 0] > 150,
            detail={"median_diameter_px": 3},
            path=(src / "images" / name).resolve(),
        )

    def patches(suite, root):
        for n, datum in data.items():
            Image.fromarray(datum["rgb"]).save(root / f"{Path(n).stem}_gt.png")
        return data, dict(center=[0, 0, 2], normal=[0, -0.5, 1])

    seen = []
    monkeypatch.setattr(probe, "patches", patches)
    monkeypatch.setattr(probe, "read_snapshot", lambda p, step, **kw: (p / f"chkpnt{step}.pth").is_file())

    def execute(cmd, **kwargs):
        seen.append(cmd)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "6"
        if "train.py" in cmd:
            out = Path(cmd[cmd.index("--model_path") + 1])
            assert json.loads((out / "keyframes.json").read_text()) == dict(selected_images=[probe.REFERENCE])
            assert float(cmd[cmd.index("--training_scene_extent") + 1]) == pytest.approx(0.055)
            assert ("--rgb_loss_weight_dir" in cmd) == (out.name == "weighted")
            (out / "chkpnt2000.pth").touch()
            ply = out / "point_cloud/iteration_2000/point_cloud.ply"
            ply.parent.mkdir(parents=True)
            ply.touch()
            cfg.keyframe_manifest = str(out / "keyframes.json")
            (out / "cfg_args").write_text(str(cfg))
        else:
            # Exercise the real render CLI's split and frame selection before simulating GPU pixels.
            args = cmd[cmd.index("scripts/render.py") + 1 :]
            gallery.main([*args, "--dry-run"])
            dst = Path(cmd[cmd.index("--output") + 1])
            for split, name in [("train", probe.REFERENCE), ("eval", probe.NEIGHBOR)]:
                for index in range(2):
                    d = dst / split / f"model_{index}" / "renders"
                    d.mkdir(exist_ok=True, parents=True)
                    Image.fromarray(rgb).save(d / f"{probe.image_id(name)}.png")

    monkeypatch.setattr(probe.subprocess, "run", execute)
    probe.main(["--suite", str(suite), "--gpu", "6"])
    assert len(seen) == 3
    report = suite / probe.REPORT
    result = json.loads((report / "results.json").read_text())
    assert result["status"] == "Complete" and len(result["measurements"]) == 4
    for r in result["measurements"]:
        assert (report / r["render"]).is_file() and r["spot_contrast"] > 0.9
    assert f"{probe.REPORT}/index.html" in (suite / "index.html").read_text()
    assert (base / "input.ply").read_bytes() == b"input is read only"
    seen.clear()
    probe.main(["--suite", str(suite), "--gpu", "6"])
    assert all("train.py" not in c for c in seen)
