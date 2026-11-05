import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


class Tensor:
    def __init__(self, value):
        self.value = value
    def clamp(self, *args):
        return self
    def to(self, *args):
        return self
    def mean(self):
        return self
    def item(self):
        return self.value


class EvalAllFramesTests(unittest.TestCase):
    def test_streams_all_frames_without_exposure_and_reports_groups(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("eval_all_frames_test", root / "scripts/eval_all_frames.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, model = root / "scene", root / "output"
            sparse = source / "sparse/0"
            sparse.mkdir(parents=True)
            (sparse / "images.bin").touch()
            (sparse / "cameras.bin").touch()
            ply = model / "point_cloud/iteration_7000/point_cloud.ply"
            ply.parent.mkdir(parents=True)
            ply.touch()
            (model / "cfg_args").write_text("Namespace(resolution=2, images='images', sh_degree=3, keyframe_manifest='keyframes.json')")
            (model / "keyframes.json").write_text(json.dumps({"selected_images": ["2.png"]}))
            camera_infos = [types.SimpleNamespace(image_name=f"{i}.png") for i in [1, 2, 3]]
            calls = []
            def module(name, **kwargs):
                m = types.ModuleType(name)
                m.__dict__.update(kwargs)
                return m
            class NoGrad:
                def __enter__(self): pass
                def __exit__(self, *args): pass
            class Model:
                def __init__(self, degree): self.degree = degree
                def load_ply(self, path): self.ply = path
            def load_cam(args, idx, info, *unused):
                calls.append(("load", info.image_name, args.resolution, args.data_device))
                return types.SimpleNamespace(image_name=info.image_name, original_image=Tensor(idx))
            def render(camera, model, pipe, background, **kwargs):
                calls.append(("render", camera.image_name, kwargs["use_trained_exp"], pipe.antialiasing))
                return {"render": Tensor(0)}
            mods = {
                "torch": module("torch", no_grad=NoGrad, tensor=lambda *a, **k: Tensor(0), float32=object()),
                "tqdm": module("tqdm", tqdm=lambda seq, **kw: seq),
                "scene": module("scene"),
                "scene.colmap_loader": module("scene.colmap_loader", **{n: lambda p: {} for n in ["read_extrinsics_binary", "read_intrinsics_binary", "read_extrinsics_text", "read_intrinsics_text"]}),
                "scene.dataset_readers": module("scene.dataset_readers", readColmapCameras=lambda *a: camera_infos),
                "scene.gaussian_model": module("scene.gaussian_model", GaussianModel=Model),
                "gaussian_renderer": module("gaussian_renderer", render=render),
                "utils": module("utils"),
                "utils.camera_utils": module("utils.camera_utils", loadCam=load_cam),
                "utils.colmap_paths": module("utils.colmap_paths", colmap_model_dir=lambda p: sparse),
                "utils.image_utils": module("utils.image_utils", psnr=lambda pred, target: Tensor(10 + target.value)),
                "diff_gaussian_rasterization": module("diff_gaussian_rasterization"),
            }
            argv = ["eval_all_frames.py", "--source", str(source), "--model", str(model), "--no-antialiasing"]
            with patch.dict(sys.modules, mods), patch.object(sys, "argv", argv):
                mod.main()
            output = model / "psnr_all_frames/iteration_7000"
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["counts"], {"all": 3, "selected": 1, "unselected": 2})
            self.assertEqual(summary["mean_psnr_db"], {"all": 11.0, "selected": 11.0, "unselected": 11.0})
            self.assertEqual([c[0] for c in calls], ["load", "render"] * 3)
            self.assertTrue(all(c[2:] == (False, False) for c in calls if c[0] == "render"))
            with (output / "per_frame.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r["selected_for_training"] for r in rows], ["0", "1", "0"])


if __name__ == "__main__":
    unittest.main()
