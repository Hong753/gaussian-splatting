import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from shutil import copytree
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from plyfile import PlyData, PlyElement

from arguments import ModelParams, OptimizationParams, PipelineParams
from scripts import orange4f_detail as audit


def fixture(root):
    source = root / "capture"
    sparse = source / "sparse/0"
    sparse.mkdir(parents=True)
    (sparse/"cameras.txt").write_text("1 PINHOLE 2048 1536 600 600 735 1393\n")
    names = ["12560.jpg", "12564.jpg", audit.FRAME, "13000.jpg"]
    images = []
    for i, (name, offset) in enumerate(zip(names, (0., .1, 0., 10.)), 1):
        center = audit.CENTER + [offset, 0., 2.]
        tx, ty, tz = center * [-1, 1, 1]
        images.append(f"{i} 0 1 0 0 {tx} {ty} {tz} 1 {name}\n\n")
    (sparse/"images.txt").write_text("".join(images))
    paths = []
    for label, selected, frozen in (("base", [0, 1, 3], False), ("region", [0, 1], False),
                                    ("fixed", [0, 1, 3], True)):
        model = root / label
        ply = model / "point_cloud/iteration_30000/point_cloud.ply"
        ply.parent.mkdir(parents=True)
        ply.touch()
        cfg = argparse.Namespace(source_path=str(source), resolution=1, antialiasing=False,
                                 keyframe_manifest=str(model/"keyframes.json"),
                                 optimization_settings=dict(iterations=30000, percent_dense=.01,
                                     freeze_lidar_positions=frozen, position_lr_max_steps=30000,
                                     scaling_lr=.005, opacity_lr=.05, densify_until_iter=15000))
        (model/"cfg_args").write_text(str(cfg))
        (model/"keyframes.json").write_text(json.dumps(dict(selected_images=[names[i] for i in selected])))
        (model/"input.ply").touch()
        paths.append(model)
    return paths


def ply(path, xyz, log_sigma=None):
    fields = {"x": xyz[:,0], "y": xyz[:,1], "z": xyz[:,2]}
    if log_sigma is not None:
        fields.update({f"scale_{a}": log_sigma for a in range(3)})
    v = np.zeros(len(xyz), dtype=[(k, "f4") for k in fields])
    for k, a in fields.items():
        v[k] = a
    PlyData([PlyElement.describe(v, "vertex")]).write(path)


class Orange4FDetailTest(unittest.TestCase):
    def test_estimate_fixed_scale_needs_only_original_cloud(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            xyz = np.float32([[0,0,0], [1,0,0], [2,0,0], [3,0,0]])
            ply(root/"input.ply", xyz)
            buf = io.StringIO()
            with redirect_stdout(buf):
                audit.main(["--fixed-model", str(root), "--estimate-fixed-scale"])
            result = json.loads(buf.getvalue().split("\n", 1)[1])
            expected = np.median(np.sqrt(np.array([14,6,6,14])/3))
            self.assertAlmostEqual(result["initial_3nn_sigma"]["median"], expected, places=6)
            self.assertAlmostEqual(result["proposed_per_anchor_axis_ceiling"]["median"], 2*expected, places=6)

    def test_fixed_scale_training_reaches_trainer_without_floor_region_or_old_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, fixed = fixture(root)
            images = root/"capture/sparse/0/images.txt"
            images.write_text(images.read_text().replace(audit.FRAME, "other.jpg"))
            (fixed/"point_cloud/iteration_30000/point_cloud.ply").unlink()
            output = root/"new_fixed"
            with patch.object(audit, "regional_model", side_effect=AssertionError), \
                    patch.object(audit.subprocess, "run") as train, patch.object(audit, "evaluate") as evaluate, \
                    patch.dict("os.environ"):
                audit.main(["--train-fixed-scale", "--fixed-model", str(fixed), "--output", str(output), "--gpu", "0"])
            train.assert_called_once()
            evaluate.assert_called_once()
            command = train.call_args.args[0]
            self.assertEqual(command[2], "train.py")
            parser = argparse.ArgumentParser()
            ModelParams(parser)
            OptimizationParams(parser)
            PipelineParams(parser)
            opt, _ = parser.parse_known_args(command[3:])
            self.assertTrue(opt.fixed_lidar)
            self.assertFalse(opt.freeze_lidar_positions)
            self.assertEqual((opt.scale_control, opt.max_scale_ratio, opt.densify_until_iter), ("clamp", 2., 0))
            self.assertEqual(opt.iterations, 50000)
            self.assertEqual(audit.read_json(output/"keyframes.json"), audit.read_json(fixed/"keyframes.json"))

    def test_adaptive_control_accepts_training_or_absent_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, region, _ = fixture(root)
            manifest = audit.read_json(base/"keyframes.json")
            manifest["selected_images"][-1] = audit.FRAME
            (base/"keyframes.json").write_text(json.dumps(manifest))
            for available in (True, False):
                if not available:
                    images = root/"capture/sparse/0/images.txt"
                    images.write_text(images.read_text().replace(audit.FRAME, "other.jpg"))
                    manifest["selected_images"][-1] = "other.jpg"
                    (base/"keyframes.json").write_text(json.dumps(manifest))
                with self.subTest(available=available):
                    buf = io.StringIO()
                    with redirect_stdout(buf):
                        audit.main(["--baseline", str(base), "--regional", str(region), "--plan-only"])
                    self.assertIn("training" if available else "0 matching registered images", buf.getvalue())
                    plan = json.loads(buf.getvalue()[buf.getvalue().index("{"):])
                    self.assertIn("--percent_dense", plan["command"])
                    spec = audit.model_info(base)
                    with patch.object(audit, "measured_frames", return_value=dict(unselected=22)), \
                            patch.object(audit, "floor_render", return_value=("gt", dict(crop_psnr_db=25, gradient_agreement=.8))) as render, \
                            patch.object(audit, "run"):
                        audit.evaluate(spec, region, "0", dict(method="shared-camera similarity", applied_scale=1.))
                    report = audit.read_json(region/"comparison/results.json")
                    self.assertIs(report["held_out"], False if available else None)
                    self.assertEqual(render.call_count, 3 if available else 0)
                    self.assertEqual("crop_psnr_db" in report["rows"][0], available)

    def test_two_checkpoints_keep_original_schedules_and_use_measured_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, region, _ = fixture(root)
            (region / "densification.jsonl").write_text(json.dumps(dict(scene_extent=.055, split_sigma_threshold=.00055))+"\n")
            buf = io.StringIO()
            with redirect_stdout(buf):
                audit.main(["--baseline", str(base), "--regional", str(region), "--output", str(root/"exp"), "--plan-only"])
            output = buf.getvalue()
            data = json.loads(output[output.index("{"):])
            self.assertAlmostEqual(data["full_extent"] * data["percent_dense"], .00055)
            parser = argparse.ArgumentParser()
            ModelParams(parser)
            OptimizationParams(parser)
            PipelineParams(parser)
            settings, _ = parser.parse_known_args(data["command"][3:])
            self.assertEqual(settings.iterations, 50000)
            self.assertEqual(settings.position_lr_max_steps, 30000)
            self.assertEqual(settings.densify_until_iter, 15000)
            self.assertEqual(settings.opacity_lr, .05)
            self.assertFalse(settings.antialiasing)
            self.assertIn("30000", data["command"][data["command"].index("--save_iterations")+1:])
            self.assertFalse((root/"exp").exists())

    def test_export_only_threshold_is_explicitly_proposed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, region, _ = fixture(root)
            _, _, rec = audit.split_settings(audit.model_info(region))
            self.assertIn("saved percent_dense", rec)
            export = root / "capture"
            manifest = json.loads((region/"keyframes.json").read_text())
            (export/"keyframes.json").write_text(json.dumps(manifest))
            (export/"selection.json").write_text(json.dumps(dict(selection=[
                dict(name=n, split="train") for n in manifest["selected_images"]])))
            spec = audit.regional_reference(export)
            extent, target, source = audit.split_settings(spec)
            transferred, info = audit.regional_cutoff(audit.model_info(base), spec, extent, target)
            self.assertEqual(transferred, target)
            self.assertEqual(info["applied_scale"], 1.)
            self.assertAlmostEqual(target, extent*.01)
            self.assertIn("proposed", source)

    def test_plan_transfers_cutoff_between_coordinate_systems(self):
        from scipy.spatial.transform import Rotation
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, region, _ = fixture(root)
            original = root / "capture"
            local = root / "other-capture"
            copytree(original, local)
            cfg_file = region / "cfg_args"
            cfg_file.write_text(cfg_file.read_text().replace(str(original), str(local)))
            names = sorted(audit.model_info(base)["records"])
            (region / "keyframes.json").write_text(json.dumps(dict(selected_images=names[:3])))
            # Independently logged radius and split threshold from the regional run.
            (region / "densification.jsonl").write_text('{"scene_extent": 2, "split_sigma_threshold": 0.04}\n')
            xyz = np.array([[0,0,2], [1,0,2], [0,2,3], [10,0,2.]])
            def write_poses(path, centers, world_to_cam):
                q = world_to_cam.as_quat()[[3,0,1,2]]
                rows = [f"{i} " + " ".join(map(str, [*q, *(-world_to_cam.apply(c))])) + f" 1 {n}\n\n"
                        for i, (n, c) in enumerate(zip(names, centers), 1)]
                (path / "sparse/0/images.txt").write_text("".join(rows))
            rotation = Rotation.from_euler("xyz", [30,70,20], degrees=True)
            write_poses(original, xyz, Rotation.identity())
            for scale, noise in ((1., 0.), (4., 0.), (1., .002)):
                with self.subTest(scale=scale, noise=noise):
                    local_centers = rotation.apply(xyz / scale) + [7., -3., 5.]
                    local_centers[2, 1] += noise
                    write_poses(local, local_centers, rotation.inv())
                    buf = io.StringIO()
                    with redirect_stdout(buf):
                        audit.main(["--baseline", str(base), "--regional", str(region), "--plan-only"])
                    log = buf.getvalue()
                    plan = json.loads(log[log.index("{"):])
                    self.assertAlmostEqual(plan["matched_split_sigma"], .04 * scale, delta=.0001)
                    transfer = plan["regional_scale_transfer"]
                    self.assertEqual(transfer["method"], "shared-camera similarity")
                    self.assertEqual(transfer["shared_views"], 3)
                    self.assertLess(transfer["relative_rms_residual"], .002)
                    self.assertAlmostEqual(plan["percent_dense"] * plan["full_extent"], plan["matched_split_sigma"])

    def test_inconsistent_trajectories_use_only_verified_shared_photos(self):
        from PIL import Image
        from utils.densification_diagnostics import training_extent
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base_path, region_path, _ = fixture(root)
            base, region = (audit.model_info(path) for path in (base_path, region_path))
            names = sorted(base["records"])[:3]
            region["splits"]["train"] = set(names + ["regional-only.jpg"])
            for i, name in enumerate(names):
                path = root / (name + ".png")
                Image.new("RGB", (2,2), "red").save(path)
                base["records"][name]["image_path"] = region["records"][name]["image_path"] = str(path)
                base["records"][name]["T"] = [i, 0., 0.]
            region["records"][names[2]]["T"] = [0., 3., 0.]
            sigma, info = audit.regional_cutoff(base, region, 2., .01)
            self.assertIn("proposal", info["method"])
            self.assertEqual((info["shared_views"], info["regional_train_views"]), (3, 4))
            self.assertAlmostEqual(sigma, .005 * training_extent(base["records"], names))
            other = root/"other.png"
            Image.new("RGB", (2,2), "blue").save(other)
            region["records"][names[1]]["image_path"] = str(other)
            with self.assertRaisesRegex(ValueError, names[1]):
                audit.regional_cutoff(base, region, 2., .01)
            region["splits"]["train"] = {names[0], "regional-only.jpg"}
            with self.assertRaisesRegex(ValueError, "1/2"):
                audit.regional_cutoff(base, region, 2., .01)
            region["splits"]["train"] = set(names)
            for name in names:
                region["records"][name]["T"] = [2., 0., 0.]
            with self.assertRaisesRegex(ValueError, "nonzero distance"):
                audit.regional_cutoff(base, region, 2., .01)

    def test_three_way_gallery_uses_verified_splits_and_same_native_crop(self):
        import cv2
        with tempfile.TemporaryDirectory() as temp:
            base_path, _, matched = fixture(Path(temp))
            base = audit.model_info(base_path)
            train = base["splits"]["train"]
            for model, step in [(base_path, 30000), (matched, 30000), (matched, 50000)]:
                metric = model/f"psnr_all_frames/iteration_{step}"
                metric.mkdir(parents=True)
                audit.atomic_json(metric/"summary.json", dict(model=str(model), iteration=step,
                    model_type="3dgs", source=base["cfg"]["source_path"], antialiasing=False,
                    resolution_setting=1, counts=dict(all=4, selected=3, unselected=1),
                    mean_psnr_db=dict(all=22., selected=22., unselected=22.),
                    occlusion=dict(occlusion_mode="off"), camera_poses="original"))
                (metric/"per_frame.csv").write_text("image_name,selected_for_training,psnr_db\n" +
                    "".join(f"{name},{int(name in train)},22\n" for name in sorted(base["records"])))
            def render(spec, names, step, folder):
                folder.mkdir(parents=True)
                target = np.zeros((1536,2048,3), np.uint8)
                target[:,500:] = 255
                image = cv2.GaussianBlur(target, (5,5), 1)
                cv2.imwrite(str(folder/"gt.png"), target)
                cv2.imwrite(str(folder/"image.png"), image)
                audit.atomic_json(folder/"frames.json", {audit.FRAME: dict(gt="gt.png", image="image.png",
                                                                          target_sha256="same")})
            with patch("scripts.render_views_worker.render_views", side_effect=render), patch.object(audit, "run") as run:
                audit.evaluate(base, matched, "0", dict(method="full-coordinate proposal from shared photos"))
            self.assertEqual(run.call_count, 2)  # Paired 30k and extended 50k galleries.
            report = audit.read_json(matched/"comparison/results.json")
            self.assertTrue(report["held_out"])
            self.assertEqual([r["iteration"] for r in report["rows"]], [30000,30000,50000])
            self.assertEqual(report["rows"][1]["label"], "Proposed 30k")
            self.assertTrue(all(r["gradient_agreement"] > .9 for r in report["rows"]))
            self.assertTrue((matched/"comparison/index.html").is_file())

    def test_gpu_selection_prefers_idle_device(self):
        with patch.dict("os.environ", {}, clear=True), patch("subprocess.check_output",
                side_effect=["GPU-b\n", "0, GPU-b, 80000\n1, GPU-a, 40000\n"]):
            self.assertEqual(audit.select_gpu(None), "1")

    def test_independent_spacing_from_survivors_original_neighbors_and_scales(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, fixed = fixture(Path(temp))
            self.enterContext(patch.object(audit, "FLOOR_SOURCE", Path(temp)/"capture"))
            xyz = np.asarray([audit.CENTER + [.01*x, .01*y, 0] for x in range(5) for y in range(5)]
                             + [audit.CENTER + [0, 0, 1]], dtype=np.float32)
            ids = [0, 2, 4, 10, 12, 14, 20, 22, 24, 25]
            ply(fixed/"input.ply", xyz)
            ply(fixed/"point_cloud/iteration_30000/point_cloud.ply", xyz[ids], np.full(len(ids),np.log(.04)))
            r = audit.spacing_report(fixed)
            self.assertEqual((r["input_points_in_crop"], r["surviving_centers_in_crop"]), (25,9))
            self.assertAlmostEqual(r["input_nearest_neighbor"]["median"], .01, places=5)
            self.assertAlmostEqual(r["surviving_nearest_neighbor"]["median"], .02, places=5)
            self.assertAlmostEqual(r["learned_max_sigma"]["median"], .04, places=5)
            self.assertGreater(r["max_sigma_over_initial"]["median"], 3.)
            moved = xyz[ids].copy()
            moved[0,0] += .0002
            ply(fixed/"point_cloud/iteration_30000/point_cloud.ply", moved, np.zeros(len(ids)))
            with self.assertRaisesRegex(ValueError, "differs from"):
                audit.spacing_report(fixed)

    def test_spacing_fits_plane_in_fixed_model_coordinates(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, fixed = fixture(Path(temp))
            # Look down at a floor well away from the saved diagnostic's world plane.
            record = audit.model_info(fixed)["records"][audit.FRAME]
            origin = -np.asarray(record["R"]) @ record["T"]
            x, y, w, h = audit.ROI
            us, vs = np.meshgrid(np.linspace(x+10, x+w-10, 12), np.linspace(y+10, y+h-10, 6))
            rays = np.column_stack([(us.ravel()-record["cx"])/record["fx"],
                                    (vs.ravel()-record["cy"])/record["fy"], np.ones(us.size)])
            xyz = origin + 4 * (rays @ np.asarray(record["R"]).T)
            ply(fixed/"input.ply", xyz)
            ply(fixed/"point_cloud/iteration_30000/point_cloud.ply", xyz, np.zeros(len(xyz)))
            report = audit.spacing_report(fixed)
            self.assertIn("fitted", report["plane_source"])
            self.assertAlmostEqual(report["plane_center"][2], origin[2]-4, places=5)
            self.assertEqual(report["surviving_centers_in_crop"], len(xyz))


if __name__ == "__main__":
    unittest.main()
