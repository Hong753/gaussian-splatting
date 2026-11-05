import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train.py"


class TrainLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "datasets"
        self.scene = self.root / "orange_4F/1_full"
        model = self.scene / "sparse"
        model.mkdir(parents=True)
        (model / "images.bin").touch()
        (model / "cameras.bin").touch()
        self.output = Path(self.temp.name) / "output"

    def run_launcher(self, root, *scenes, extra=()):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--data-root", str(root),
             "--output-root", str(self.output), "--scenes", *scenes,
             "--name", "test", "--select-keyframes", "--dry-run", *extra],
            text=True, capture_output=True,
        )

    def test_nested_scene_keeps_relative_path_in_output(self):
        result = self.run_launcher(self.root, "orange_4F/1_full")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(self.scene), result.stdout)
        self.assertIn(str(self.output / "orange_4F/1_full/test/keyframes.json"), result.stdout)
        self.assertIn("--antialiasing", result.stdout)
        self.assertIn("--train_test_exp", result.stdout)
        self.assertIn("--port 6009", result.stdout)
        self.assertFalse(self.output.exists())

    def test_data_root_itself_can_be_scene(self):
        result = self.run_launcher(self.scene, ".")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(self.output / "1_full/test/keyframes.json"), result.stdout)

    def test_reject_escape_and_duplicate_normalized_paths(self):
        for scenes in (("../other",), (str(self.scene),),
                       ("orange_4F/1_full", "orange_4F//1_full")):
            with self.subTest(scenes=scenes):
                self.assertNotEqual(self.run_launcher(self.root, *scenes).returncode, 0)

    def test_eval_requires_disabling_exposure_to_preserve_split(self):
        fail = self.run_launcher(self.root, "orange_4F/1_full", extra=("--eval",))
        self.assertNotEqual(fail.returncode, 0)
        self.assertIn("--no-exposure-compensation", fail.stderr)
        ok = self.run_launcher(self.root, "orange_4F/1_full", extra=("--eval", "--no-exposure-compensation", "--no-antialiasing"))
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertIn("--eval", ok.stdout)
        self.assertNotIn("--train_test_exp", ok.stdout)
        self.assertNotIn("--antialiasing", ok.stdout)

    def test_optional_depths_and_sparse_optimizer_are_forwarded(self):
        result = self.run_launcher(self.root, "orange_4F/1_full", extra=("--optimizer-type", "sparse_adam", "--depths", "depth_maps"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--optimizer_type sparse_adam", result.stdout)
        self.assertIn("--depths depth_maps", result.stdout)

    def test_viewer_port_and_disable_are_forwarded(self):
        port = self.run_launcher(self.root, "orange_4F/1_full", extra=("--viewer-port", "6010"))
        self.assertEqual(port.returncode, 0, port.stderr)
        self.assertIn("--port 6010", port.stdout)
        disabled = self.run_launcher(self.root, "orange_4F/1_full", extra=("--disable-viewer",))
        self.assertEqual(disabled.returncode, 0, disabled.stderr)
        self.assertIn("--disable_viewer", disabled.stdout)
        self.assertNotIn("--port 6009", disabled.stdout)
        bad = self.run_launcher(self.root, "orange_4F/1_full", extra=("--viewer-port", "65536"))
        self.assertNotEqual(bad.returncode, 0)


if __name__ == "__main__":
    unittest.main()
