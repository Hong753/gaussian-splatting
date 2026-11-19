import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
import pytest

from scripts import orange4f_multiview_floor_distance as probe
from scripts import render as gallery
from utils.region_loss import load_view_weights, mask_path
from utils.training_previews import create_previewer


def make_suite(tmp_path):
    suite = tmp_path / "suite"
    model = suite / "3dgs"
    model.mkdir(parents=True)
    source = tmp_path / "scene"
    sparse = source / "sparse/0"
    sparse.mkdir(parents=True)
    (source / "images").mkdir()
    size = 420
    poses = {
        "09879.jpg": (0.26, 0.45, 0.45),
        probe.REFERENCE: (0, 0, 0),
        probe.NEIGHBOR: (0.08, 0.05, 0.0),
        "09885.jpg": (0.2, 0.15, -0.25),
        "09888.jpg": (-0.15, -0.3, -2.0),
        "09891.jpg": (10.0, 0.0, -2.0),
    }
    (sparse / "cameras.txt").write_text(f"1 PINHOLE {size} {size} 200 200 210 210\n")
    (sparse / "images.txt").write_text(
        "".join(
            f"{i + 1} 1 0 0 0 {-pos[0]} {-pos[1]} {-pos[2]} 1 {name}\n\n"
            for i, (name, pos) in enumerate(poses.items())
        )
    )
    rgb = np.full((size, size, 3), 65, np.uint8)
    rgb[2::6, 2::6] = 185
    rgb[3::6, 2::6] = 185
    for name in poses:
        Image.fromarray(rgb).save(source / "images" / name, quality=100)
    x, y = np.meshgrid(np.linspace(-3.2, 3.2, 220), np.linspace(-3.2, 3.2, 220))
    xyz = np.zeros(x.size, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    xyz["x"], xyz["y"], xyz["z"] = x.ravel(), y.ravel(), 3.0
    cloud = model / "input.ply"
    PlyData([PlyElement.describe(xyz, "vertex")]).write(cloud)
    manifest = model / "keyframes.json"
    manifest.write_text(json.dumps(dict(selected_images=list(poses), image_directory="images")))
    cfg = argparse.Namespace(
        source_path=str(source),
        resolution=1,
        keyframe_manifest=str(manifest),
        optimization_settings=dict(iterations=30000, percent_dense=0.01),
    )
    (model / "cfg_args").write_text(str(cfg))
    (suite / "suite.json").write_text(
        json.dumps(
            dict(
                source=str(source),
                cloud=dict(cached=str(cloud)),
                selection=str(manifest),
                runs=[],
            )
        )
    )
    return suite, cfg, poses


def test_multiview_probe_prepares_real_colmap_and_lidar_then_uses_shared_split(tmp_path, monkeypatch):
    suite, cfg, poses = make_suite(tmp_path)
    data_path = suite / probe.REPORT
    probe.main(["--suite", str(suite), "--steps-per-view", "5", "--prepare-only"])
    protocol = json.loads((data_path / "protocol.json").read_text())
    train = protocol["train"]
    holdout = protocol["held_out"]
    assert train[0] == probe.REFERENCE and len(train) == 4
    assert holdout == [probe.NEIGHBOR]
    assert set(train) == set(poses) - {probe.NEIGHBOR, "09891.jpg"}
    assert protocol["iteration"] == 20
    distance_weights = protocol["training_weights"]["distance_weighted"]
    d = {name: protocol["views"][name]["distance_m"] for name in train}
    closest, farthest = min(d, key=d.get), max(d, key=d.get)
    assert distance_weights[closest] / distance_weights[farthest] == pytest.approx((d[farthest] / d[closest]) ** 2)
    assert np.mean(list(distance_weights.values())) == pytest.approx(1.0)
    assert protocol["training_weights"]["uniform"] == dict.fromkeys(train, 1.0)
    for name, view in protocol["views"].items():
        mask = np.asarray(Image.open(mask_path(data_path / "masks", name)))
        assert np.count_nonzero(mask) == view["supported_pixels"] < mask.size
    # Run the real preview selector with test-only cameras loaded through a stub.
    native_scene = SimpleNamespace(model_path=str(tmp_path / "preview"))
    native_scene.getTrainCameras = lambda: [SimpleNamespace(image_name=n) for n in train]
    native_scene.getTestCameras = list
    manifest = tmp_path / "preview.json"
    manifest.write_text(json.dumps(dict(selected_images=train, preview_heldout_images=holdout)))
    dataset = SimpleNamespace(keyframe_manifest=str(manifest), source_path=cfg.source_path, images="images")
    monkeypatch.setattr(
        "utils.training_previews.load_heldout_camera", lambda r, dataset: SimpleNamespace(image_name=r["name"])
    )
    preview = create_previewer(native_scene, dataset, dict(every=500, train=2, test=1, seed=0))
    assert [(label, c.image_name) for label, c in preview.frames if label == "held-out"] == [
        ("held-out", probe.NEIGHBOR)
    ]
    # Substitute only GPU subprocesses, checking the commands against the real native CLI schema and gallery.
    monkeypatch.setattr(probe, "read_snapshot", lambda out, step, **kw: (out / f"chkpnt{step}.pth").exists())
    commands = []

    def execute(cmd, **kwargs):
        commands.append(cmd)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "6"
        if "train.py" in cmd:
            from arguments import ModelParams, OptimizationParams, PipelineParams

            parser = argparse.ArgumentParser()
            for group in (ModelParams, OptimizationParams, PipelineParams):
                group(parser)
            options, _ = parser.parse_known_args(cmd[cmd.index("train.py") + 1 :])
            model = Path(options.model_path)
            assert options.loss_mask_dir == str(data_path / "masks")
            assert options.training_scene_extent > 0
            assert json.loads(Path(options.keyframe_manifest).read_text()) == dict(
                selected_images=train,
                preview_heldout_images=holdout,
            )
            if model.name == "distance_weighted":
                assert load_view_weights(options.view_loss_weights, train) == distance_weights
            else:
                assert options.view_loss_weights == ""
            (model / "chkpnt20.pth").touch()
            ply = model / "point_cloud/iteration_20/point_cloud.ply"
            ply.parent.mkdir(parents=True)
            ply.touch()
            args = argparse.Namespace(
                **{
                    **vars(cfg),
                    "keyframe_manifest": options.keyframe_manifest,
                    "loss_mask_dir": options.loss_mask_dir,
                }
            )
            (model / "cfg_args").write_text(str(args))
            return
        args = cmd[cmd.index("scripts/render.py") + 1 :]
        gallery.main([*args, "--dry-run"])
        dst = data_path / "comparison"
        for split, names in [("train", train), ("eval", holdout)]:
            for i in range(2):
                directory = dst / split / f"model_{i}" / "renders"
                directory.mkdir(parents=True, exist_ok=True)
                for name in names:
                    with Image.open(Path(cfg.source_path) / "images" / name) as image:
                        image.save(directory / f"{probe.image_id(name)}.png")

    monkeypatch.setattr(probe.subprocess, "run", execute)
    probe.main(["--suite", str(suite), "--steps-per-view", "5", "--gpu", "6"])
    assert len(commands) == 3
    result = json.loads((data_path / "results.json").read_text())
    assert result["metrics"]["uniform"][probe.NEIGHBOR]["psnr_db"] == "inf"
    assert "Complete" in (data_path / "index.html").read_text()
    assert probe.REPORT in (suite / "index.html").read_text()
    commands.clear()
    probe.main(["--suite", str(suite), "--steps-per-view", "5", "--gpu", "6"])
    assert len(commands) == 1 and "scripts/render.py" in commands[0]


def test_view_weight_loader_rejects_eval_frames_and_nonpositive_weights(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"train.jpg": 1, "eval.jpg": 2}')
    with pytest.raises(ValueError, match="held-out"):
        load_view_weights(path, ["train.jpg"])
    path.write_text('{"train.jpg": 0}')
    with pytest.raises(ValueError, match="positive"):
        load_view_weights(path, ["train.jpg"])
