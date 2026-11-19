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
from scripts.orange4f_splat_footprints import NEIGHBOR
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
        NEIGHBOR: (0.08, 0.05, 0.0),
        "09885.jpg": (0.2, 0.15, -0.25),
        "09888.jpg": (-0.15, -0.3, -2.0),
        "09891.jpg": (10.0, 0.0, -2.0),
        "10900.jpg": (0.1, -0.1, -6.2),
        "10901.jpg": (-0.3, 0.0, -6.7),
        "10903.jpg": (0.4, 0.3, -7.0),
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
    # An intervening surface hides the original crop from distant cameras.
    ox, oy = np.meshgrid(np.linspace(0.15, 2.8, 140), np.linspace(0.85, 2.9, 110))
    xyz = np.zeros(x.size + ox.size, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    xyz["x"], xyz["y"] = np.r_[x.ravel(), ox.ravel()], np.r_[y.ravel(), oy.ravel()]
    xyz["z"] = np.r_[np.full(x.size, 3.0), np.full(ox.size, -1.0)]
    cloud = model / "input.ply"
    PlyData([PlyElement.describe(xyz, "vertex")]).write(cloud)
    manifest = model / "keyframes.json"
    selected = [name for name in poses if name != "10903.jpg"]
    manifest.write_text(json.dumps(dict(selected_images=selected, image_directory="images")))
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
    suite, cfg, _ = make_suite(tmp_path)
    data_path = suite / probe.REPORT
    source = Path(cfg.source_path)
    _, cameras, images, stamps = probe.read_model(source)
    by_name = {view.name: view for view in images}
    ref = by_name[probe.REFERENCE]
    cam = cameras[ref.camera_id]
    left, top, right, bottom = [
        round(frac * size)
        for frac, size in zip(
            probe.ROI_FRACTIONS,
            (cam.width, cam.height) * 2,
        )
    ]
    rect = left, top, right - left, bottom - top
    center, normal = np.array([0.0, 0.0, 3.0]), np.array([0.0, 0.0, 1.0])
    world = probe.plane_grid(ref, cam, rect, center, normal)
    original_range = float(np.linalg.norm(ref.C - world.mean(axis=(0, 1))))
    allowed = set(json.loads((suite / "3dgs/keyframes.json").read_text())["selected_images"])
    with pytest.raises(ValueError, match="No distant train/eval pair"):
        probe.match_floor_views(
            source,
            suite / "3dgs/input.ply",
            "images",
            images,
            cameras,
            stamps,
            allowed,
            4,
            None,
            2.0,
            original_range,
            ref,
            rect,
            [by_name[NEIGHBOR]],
            (center, normal, world),
            {},
        )
    args = ["--suite", str(suite), "--steps-per-view", "5", "--search-window", "2"]
    probe.main([*args, "--prepare-only"])
    protocol = json.loads((data_path / "protocol.json").read_text())
    train = protocol["train"]
    holdout = protocol["held_out"]
    assert len(train) == 4 and len(holdout) == 2
    assert not set(train).intersection(holdout)
    assert {protocol["views"][name]["distance_group"] for name in holdout} == {"near", "far"}
    assert [protocol["views"][name]["distance_group"] for name in train].count("far") == 2
    geometry = protocol["geometry"]
    assert geometry["reference"] == train[0]
    assert np.linalg.norm(np.array(geometry["patch_center_m"]) - geometry["original_patch_center_m"]) >= 1
    far_min = geometry["far_min_distance_m"]
    assert far_min == pytest.approx(
        2 * max(geometry["original_reference_range_m"], protocol["views"][train[0]]["distance_m"])
    )
    for name in protocol["views"]:
        if protocol["views"][name]["distance_group"] == "far":
            assert name.startswith("109") and protocol["views"][name]["distance_m"] >= far_min
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
    preview = create_previewer(native_scene, dataset, dict(every=500, train=2, test=2, seed=0))
    assert [c.image_name for label, c in preview.frames if label == "held-out"] == sorted(holdout)
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
    probe.main([*args, "--gpu", "6"])
    assert len(commands) == 3
    result = json.loads((data_path / "results.json").read_text())
    assert result["held_out"] == holdout
    for name in holdout:
        assert result["metrics"]["uniform"][name]["psnr_db"] == "inf"
    assert "Complete" in (data_path / "index.html").read_text()
    assert probe.REPORT in (suite / "index.html").read_text()
    commands.clear()
    probe.main([*args, "--gpu", "6"])
    assert len(commands) == 1 and "scripts/render.py" in commands[0]


def test_view_weight_loader_rejects_eval_frames_and_nonpositive_weights(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"train.jpg": 1, "eval.jpg": 2}')
    with pytest.raises(ValueError, match="held-out"):
        load_view_weights(path, ["train.jpg"])
    path.write_text('{"train.jpg": 0}')
    with pytest.raises(ValueError, match="positive"):
        load_view_weights(path, ["train.jpg"])


def test_distant_run_refuses_to_prepare_close_only(tmp_path):
    suite, _, _ = make_suite(tmp_path)
    with pytest.raises(ValueError, match="No alternative floor crop"):
        probe.main(["--suite", str(suite), "--far-min-ratio", "4", "--prepare-only"])
    assert not (suite / probe.REPORT / "protocol.json").is_file()


def test_distant_frames_can_come_from_outside_stride_split(tmp_path):
    suite, _, _ = make_suite(tmp_path)
    manifest = suite / "3dgs/keyframes.json"
    selection = json.loads(manifest.read_text())
    selection["selected_images"] = [name for name in selection["selected_images"] if not name.startswith("109")]
    manifest.write_text(json.dumps(selection))
    probe.main(["--suite", str(suite), "--prepare-only", "--train-views", "2"])
    plan = json.loads((suite / probe.REPORT / "protocol.json").read_text())
    assert plan["train"][1] not in selection["selected_images"]
    assert plan["views"][plan["train"][1]]["distance_group"] == "far"
    assert plan["views"][plan["held_out"][1]]["distance_group"] == "far"
    assert not set(plan["train"]).intersection(plan["held_out"])
