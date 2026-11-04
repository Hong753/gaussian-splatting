import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("keyframes", ROOT / "scripts/keyframes.py")
keyframes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(keyframes)


class KeyframeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scene = self.root / "scene"
        self.model = self.scene / "sparse/0"
        self.model.mkdir(parents=True)
        (self.scene / "images").mkdir()
        (self.model / "cameras.txt").write_text("placeholder")
        records = []
        for index in range(1, 11):
            name = f"frame{index}.png"
            (self.scene / "images" / name).write_text(str(index))
            records.append(f"{index} 1 0 0 0 0 0 0 1 {name}\n\n")
        (self.model / "images.txt").write_text("".join(records))
        self.output = self.root / "output"

    def test_selection_respects_original_test_split_and_reuses_manifest(self):
        class Score:
            def __init__(self, value):
                self.value = value

            def var(self):
                return self.value

        fake_cv2 = types.SimpleNamespace(
            IMREAD_GRAYSCALE=0, CV_64F=6,
            imread=lambda path, mode: int(Path(path).read_text()),
            Laplacian=lambda image, depth: Score(image),
        )
        with patch.dict(sys.modules, {"cv2": fake_cv2}):
            path = keyframes.select_keyframes(self.scene, "images", self.output, 3, True)
        manifest = json.loads(path.read_text())
        self.assertEqual(manifest["selected_images"], ["frame4.png", "frame7.png", "frame10.png"])
        self.assertEqual(manifest["test_count"], 2)
        self.assertEqual(manifest["candidate_count"], 8)
        # Reuse does not need to decode any images again.
        self.assertEqual(keyframes.select_keyframes(self.scene, "images", self.output, 3, True), path)
        with self.assertRaisesRegex(ValueError, "does not match"):
            keyframes.select_keyframes(self.scene, "images", self.output, 2, True)
        self.assertFalse((self.model / "points3D.ply").exists())

    def test_binary_names_and_truncated_observations(self):
        binary = self.model / "images.bin"
        (self.model / "cameras.bin").write_bytes(b"placeholder")
        with binary.open("wb") as file:
            file.write(struct.pack("<Q", 2))
            for index in (1, 2):
                file.write(struct.pack("<i7di", index, 1, 0, 0, 0, 0, 0, 0, 1))
                file.write(f"frame{index}.png".encode() + b"\0")
                file.write(struct.pack("<Qddq", 1, 2.0, 3.0, -1))
        self.assertEqual(keyframes.registered_names(self.scene), ["frame1.png", "frame2.png"])
        binary.write_bytes(binary.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "Truncated"):
            keyframes.registered_names(self.scene)


if __name__ == "__main__":
    unittest.main()
