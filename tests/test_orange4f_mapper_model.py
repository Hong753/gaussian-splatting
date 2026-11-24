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

    def test_native_learned_sfm_workspaces_and_training_sources(self):
        for method, detector, cap in (("flexglue", "aliked-n16rot", "--aliked-max-features"),
                                      ("loma-b", "loma-b", "--loma-max-features")):
            with self.subTest(method=method):
                self.check_native_model(method, detector, cap)

    def check_native_model(self, method, detector, cap):
        import os
        from unittest.mock import patch
        from scripts import orange4f_spirula_3dgs as runner

        names = [f"{i}.png" for i in range(9)]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            capture, scene, fixture = base / "capture", base / method, base / "fixture"
            (capture / "images").mkdir(parents=True)
            write_component(capture / "sparse/0", names)
            write_component(fixture, names[1:])
            for name in names:
                Image.new("RGB", (64, 48)).save(capture / "images" / name)
            # Fixture binary only exercises the shell/3DGS handoff.
            binary = base / "spirula"
            binary.write_text("""#!/usr/bin/env bash
set -eu
case "$1 $2" in
  'sfm --version') echo 'spirula sfm 2026.11.12 (667e353)' ;;
  '--accept-license flexglue') exit 0 ;;
  'sam devices') echo 'device fixture' ;;
  'sfm auto')
    printf '%s\\n' "$@" > "$TEST_ARGS"
    mkdir -p "$5/sparse/0"
    cp "$TEST_MODEL"/*.bin "$5/sparse/0/"
    exit 3 ;;
  *) exit 1 ;;
esac
""")
            binary.chmod(0o755)
            env = dict(os.environ, SPIRULA_BIN=str(binary), TEST_MODEL=str(fixture), TEST_ARGS=str(base/"args"))
            cmd = ["bash", "scripts/orange4f_spirula_sfm.sh", "--" + method, str(capture), str(scene)]
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 3, result.stderr)
            flags = (base/"args").read_text().splitlines()
            for flag, value in (("--features",detector), ("--matcher",method), ("--mapper","flat"),
                                (cap,"2048"), ("--max-image-size","1600")):
                self.assertEqual(flags[flags.index(flag) + 1], value)
            self.assertTrue((scene/"images").samefile(capture/"images"))
            (base/"args").unlink()
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 3, result.stderr)
            self.assertFalse((base/"args").exists())
            result = subprocess.run(cmd[:2] + cmd[3:], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("contains " + method, result.stderr)

            split = base / "keyframes.json"
            split.write_text(json.dumps(dict(selected_images=names[::3], group_size=3, eval=False,
                registered_names_sha256=hashlib.sha256(json.dumps(names).encode()).hexdigest())))
            # Exercise the default FlexGlue scene/run names and the original capture split.
            with patch.object(runner, "SOURCE", scene.with_name("rgb")), patch.object(runner, "RUN", base/"sift-gs"):
                scene.rename(base / f"spirula_sfm_{method}_w1600")
                with patch.object(sys, "argv", ["runner", "--" + method, "--expected-frames", "9",
                                                "--keyframes", str(split), "--prepare-only"]):
                    runner.main()
            protocol = json.loads((base/f"{method}_3dgs_stride3_w1600/protocol.json").read_text())
            self.assertIn("FlexGlue" if method == "flexglue" else "LoMa-B", protocol["pose_provenance"])
            self.assertEqual(protocol["unregistered"], names[:1])
            self.assertEqual(protocol["train"], names[3::3])
