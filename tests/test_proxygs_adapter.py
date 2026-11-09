"""Check LiDAR/camera provenance without requiring the CUDA training stack."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.proxygs_support import REVISION, prepare_dataset, resolution
from scripts.run_proxygs import main
from test_view_comparisons import dataset


class ProxyGSAdapterTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source, self.manifest, self.names, self.selected = dataset(self.root)
        self.model = self.root / "proxygs"

    def test_staging_preserves_splits_cloud_and_camera_projection(self):
        before = {str(p): p.read_bytes() for p in self.source.rglob("*") if p.is_file()}
        preparation = prepare_dataset(self.source, self.manifest, self.model)
        stage = preparation.parent
        data = json.loads(preparation.read_text())
        self.assertEqual((stage / "input.ply").resolve(), self.source / "sparse/points3D.ply")
        self.assertTrue((stage / "input.ply").is_symlink())
        by_alias = {r["alias"]: r for r in data["frames"]}
        self.assertEqual(len(by_alias), len(self.selected) + 2)
        for split, suffix, expected in (("train", "train", self.selected), ("eval", "test", self.names[::8])):
            frames = json.loads((stage / f"transforms_{suffix}.json").read_text())["frames"]
            self.assertEqual(len(frames), len(expected))
            for frame in frames:
                source_record = by_alias[Path(frame["file_path"]).stem]
                self.assertEqual(source_record["split"], split)
                self.assertIn(source_record["name"], expected)
                self.assertEqual((stage / frame["file_path"]).resolve(), Path(source_record["image_path"]))
                c2w = np.array(frame["transform_matrix"])
                c2w[:3, 1:3] *= -1
                w2c = np.linalg.inv(c2w)
                np.testing.assert_allclose(w2c[:3, :3], np.array(source_record["R"]).T, atol=1e-14)
                np.testing.assert_allclose(w2c[:3, 3], source_record["T"], atol=1e-14)
                with Image.open(stage / frame["file_path"]) as image:
                    self.assertAlmostEqual(2*np.arctan(image.width/(2*frame["fl_x"])), source_record["fov_x"])
                    self.assertAlmostEqual(2*np.arctan(image.height/(2*frame["fl_y"])), source_record["fov_y"])
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.source.rglob("*") if p.is_file()})
        self.assertEqual(prepare_dataset(self.source, self.manifest, self.model), preparation)

    def test_changed_cloud_or_manifest_is_not_silently_reused(self):
        prepare_dataset(self.source, self.manifest, self.model)
        (self.source / "sparse/points3D.ply").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "differs"):
            prepare_dataset(self.source, self.manifest, self.model)

    def test_missing_cloud_never_falls_back_to_random_points(self):
        (self.source / "sparse/points3D.ply").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "LiDAR"):
            prepare_dataset(self.source, self.manifest, self.model)
        self.assertFalse(self.model.exists())

    def test_mismatched_pose_source_is_rejected(self):
        data = json.loads(self.manifest.read_text())
        data["source_path"] = str(self.root / "different_poses")
        self.manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "pose set"):
            prepare_dataset(self.source, self.manifest, self.model)

    def test_no_eval_split_stages_only_selected_train_images(self):
        data = json.loads(self.manifest.read_text())
        data["eval"] = False
        self.manifest.write_text(json.dumps(data))
        path = prepare_dataset(self.source, self.manifest, self.model)
        self.assertEqual(json.loads((path.parent / "transforms_test.json").read_text())["frames"], [])
        self.assertEqual(len(json.loads(path.read_text())["frames"]), len(self.selected))

    def test_run_dry_run_has_checkpoints_proxy_culling_and_final_reports(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = main(["run", "--source", str(self.source), "--keyframes", str(self.manifest),
                           "--model", str(self.model), "--gpu", "4", "--dry-run"])
        self.assertEqual(result, 0)
        for argument in ("--ply_path", "--ply_mesh", "--depth_npy_dir", "--test_iterations 7000 30000",
                         "--save_iterations 7000 30000", "--eval", "--split both", "--gpu 4"):
            self.assertIn(argument, stdout.getvalue())
        self.assertFalse(self.model.exists())

    def test_cache_resolution_matches_camera_resizing(self):
        self.assertEqual(resolution(1920, 1080, -1), (1600, 900))
        self.assertEqual(resolution(1280, 720, -1), (1280, 720))
        self.assertEqual(resolution(1921, 1081, 2), (960, 540))
        self.assertEqual(resolution(1920, 1080, 1000), (1000, 562))

    def test_prepare_dispatches_to_isolated_environment(self):
        environment, repo = self.root / "venv", self.root / "upstream"
        environment.mkdir()
        (environment / "proxygs_install.json").write_text(json.dumps(dict(revision=REVISION, upstream=str(repo))))
        with patch("scripts.run_proxygs.subprocess.run") as worker:
            self.assertEqual(main(["prepare", "--source", str(self.source), "--keyframes", str(self.manifest),
                                   "--model", str(self.model), "--environment", str(environment), "--repo", str(repo)]), 0)
        command = worker.call_args.args[0]
        self.assertEqual(command[0], str(environment / "bin/python"))
        self.assertIn("prepare", command)
        self.assertFalse(self.model.exists())


if __name__ == "__main__":
    unittest.main()
