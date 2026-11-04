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

    def run_launcher(self, root, *scenes):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--data-root", str(root),
             "--output-root", str(self.output), "--scenes", *scenes,
             "--name", "test", "--select-keyframes", "--dry-run"],
            text=True, capture_output=True,
        )

    def test_nested_scene_keeps_relative_path_in_output(self):
        result = self.run_launcher(self.root, "orange_4F/1_full")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(self.scene), result.stdout)
        self.assertIn(str(self.output / "orange_4F/1_full/test/keyframes.json"), result.stdout)
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


if __name__ == "__main__":
    unittest.main()
