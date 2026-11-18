import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
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
            audit.check_common_world(audit.model_info(base), spec)
            extent, target, source = audit.split_settings(spec)
            self.assertAlmostEqual(target, extent*.01)
            self.assertIn("proposed", source)

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
                audit.evaluate(base, matched, "0")
            self.assertEqual(run.call_count, 2)  # Paired 30k and extended 50k galleries.
            report = audit.read_json(matched/"comparison/results.json")
            self.assertTrue(report["held_out"])
            self.assertEqual([r["iteration"] for r in report["rows"]], [30000,30000,50000])
            self.assertTrue(all(r["gradient_agreement"] > .9 for r in report["rows"]))
            self.assertTrue((matched/"comparison/index.html").is_file())

    def test_gpu_selection_prefers_idle_device(self):
        with patch.dict("os.environ", {}, clear=True), patch("subprocess.check_output",
                side_effect=["GPU-b\n", "0, GPU-b, 80000\n1, GPU-a, 40000\n"]):
            self.assertEqual(audit.select_gpu(None), "1")

    def test_independent_spacing_from_survivors_original_neighbors_and_scales(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, fixed = fixture(Path(temp))
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


if __name__ == "__main__":
    unittest.main()
