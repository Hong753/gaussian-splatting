import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


def write_component(folder, names):
    folder.mkdir(parents=True)
    (folder / "cameras.bin").write_bytes(struct.pack("<QiiQQ4d", 1, 1, 1, 64, 48, 50, 50, 32, 24))
    with (folder / "images.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", len(names)))
        for i, name in enumerate(names, 1):
            stream.write(struct.pack("<i7di", i, 1, 0, 0, 0, 0.1 * i, 0, 0, 1))
            stream.write(name.encode() + b"\0" + struct.pack("<Q", 0))
    with (folder / "points3D.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", 4))
        for i in range(4):
            stream.write(struct.pack("<Q3d3BdQii", i + 1, i, 0, 1, 2, 3, 4, 0, 1, 1, 0))


class MapperModelTests(unittest.TestCase):
    def test_direct_model_and_partial_coverage(self):
        names = [f"{i}.png" for i in range(9)]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source, run = base / "sfm", base / "training"
            write_component(source / "sparse/0", names)
            photos = source / "images"
            photos.mkdir()
            for name in names:
                Image.new("RGB", (64, 48)).save(photos / name)
            scan = source / "superpoint_superglue_lidar"
            raw = scan / "map.first/0"
            write_component(raw, names[1:])
            manifest = base / "keyframes.json"
            manifest.write_text(json.dumps(dict(selected_images=names[::3], group_size=3, eval=False,
                registered_names_sha256=hashlib.sha256(json.dumps(names).encode()).hexdigest())))
            cmd = [sys.executable, "-m", "scripts.orange4f_spirula_3dgs", "--source", str(source), "--run", str(run),
                   "--expected-frames", "9", "--prepare-only", "--mapper-model", "0"]
            def prepare(*extra):
                return subprocess.run(cmd + list(extra), capture_output=True, text=True)
            point_file = raw / "points3D.bin"
            data = point_file.read_bytes()
            point_file.write_bytes(data[:-1])
            self.assertIn("observations are incomplete", prepare("--keyframes", str(manifest)).stderr)
            point_file.write_bytes(data)
            result = prepare("--keyframes", str(manifest))
            self.assertEqual(result.returncode, 0, result.stderr)
            protocol = json.loads((run / "protocol.json").read_text())
            self.assertEqual(protocol["mapper_model"], str(raw))
            self.assertEqual(protocol["unregistered"], names[:1])
            self.assertEqual(protocol["train"], names[3::3])
            self.assertEqual(len(protocol["test"]), 6)
            self.assertTrue((run / "scene/sparse/0/images.bin").samefile(raw / "images.bin"))
            # A later mapper attempt must not change a prepared run on restart.
            write_component(scan / "map.later/0", names)
            result = prepare()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((run / "scene/sparse/0/images.bin").samefile(raw / "images.bin"))
            # A new run with no selector uses the live model, even without any BA marker.
            cmd = cmd[:-2]
            cmd[cmd.index(str(run))] = str(base / "live")
            result = prepare("--keyframes", str(manifest))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((base / "live/scene/sparse/0/images.bin").samefile(source / "sparse/0/images.bin"))
