import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import render_ablation_videos as video
from scripts import collect_ablation_results as collect
from scripts.proxygs_support import prepare_dataset, proxy_depth_renderer
from test_view_comparisons import dataset, checkpoint


class AblationVideosTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source, self.manifest, self.names, self.selected = dataset(self.root)

    def model(self, name, source=None, iteration=30000):
        path = self.root / name
        path.mkdir()
        (path / "cfg_args").write_text(f"Namespace(source_path={str(source or self.source)!r}, eval=True, keyframe_manifest='keyframes.json', antialiasing=True)")
        (path / "keyframes.json").write_bytes(self.manifest.read_bytes())
        checkpoint(path, iteration)
        return path

    def refined(self):
        path = self.root / "refined_source"
        (path / "sparse").mkdir(parents=True)
        (path / "images").symlink_to(self.source / "images", target_is_directory=True)
        (path / "sparse/points3D.ply").symlink_to(self.source / "sparse/points3D.ply")
        (path / "sparse/cameras.txt").write_bytes((self.source / "sparse/cameras.txt").read_bytes())
        text = (self.source / "sparse/images.txt").read_text().replace(" 0 2 3 1 ", " 0.1 2 3 1 ", 1)
        (path / "sparse/images.txt").write_text(text)
        return path

    def args(self, models, mode="random", **options):
        argv = ["--video", mode, "--source", str(self.source), "--models", *map(str, models),
            "--output", str(self.root / "video"), "--width", "64", "--duration", "1", "--fps", "5", "--source-window", "12"]
        for key, value in options.items():
            argv += ["--" + key.replace("_", "-"), str(value)]
        return video.parser().parse_args(argv)

    def test_random_models_share_exact_trajectory_including_refined_poses(self):
        models = [self.model("baseline"), self.model("refined", self.refined())]
        plan = video.build_plan(self.args(models))
        self.assertEqual(plan["frame_count"], 5)
        self.assertEqual({m["trajectory_sha256"] for m in plan["models"]}, {plan["trajectory_sha256"]})
        again = video.build_plan(self.args(models))
        self.assertEqual(plan["trajectory_sha256"], again["trajectory_sha256"])
        self.assertFalse(Path(plan["output"]).exists())

    def test_actual_models_can_use_own_calibration_with_identical_gt(self):
        models = [self.model("baseline"), self.model("refined", self.refined())]
        common = video.build_plan(self.args(models, "actual"))
        own = video.build_plan(self.args(models, "actual", actual_poses="model"))
        self.assertEqual(common["models"][0]["cameras"], common["models"][1]["cameras"])
        self.assertNotEqual(own["models"][0]["trajectory_sha256"], own["models"][1]["trajectory_sha256"])
        self.assertEqual([c["image_name"] for c in own["models"][0]["cameras"]], [c["image_name"] for c in own["models"][1]["cameras"]])

    def test_refuses_unrelated_coordinate_system(self):
        other = self.refined()
        (other / "sparse/points3D.ply").unlink()
        (other / "sparse/points3D.ply").write_text("other cloud")
        models = [self.model("baseline"), self.model("unrelated", other)]
        with self.assertRaisesRegex(ValueError, "common world frame"):
            video.build_plan(self.args(models))

    def test_common_checkpoint_or_explicit_disclosed_iterations(self):
        a, b = self.model("a"), self.model("b", iteration=60000)
        checkpoint(a, 7000)
        checkpoint(b, 7000)
        args = self.args([a, b])
        plan = video.build_plan(args)
        self.assertEqual([m["iteration"] for m in plan["models"]], [7000, 7000])
        args.iterations = [30000, 60000]
        plan = video.build_plan(args)
        self.assertEqual([m["iteration"] for m in plan["models"]], [30000, 60000])
        args.iterations = [30000, 40000]
        with self.assertRaises(FileNotFoundError):
            video.build_plan(args)
        self.assertFalse((self.root / "video").exists())

    def test_proxy_is_validated_and_uses_environment_without_python_supervisor(self):
        native = self.model("native")
        proxy = self.root / "proxy"
        prep = prepare_dataset(self.source, self.manifest, proxy)
        (proxy / "cfg_args").write_text(f"Namespace(source_path={str(prep.parent)!r}, appearance_dim=0)")
        checkpoint(proxy, 30000)
        folder = proxy / "point_cloud/iteration_30000"
        for name in ("opacity_mlp.pt", "cov_mlp.pt", "color_mlp.pt"):
            (folder / name).touch()
        mesh = prep.parent / "proxy.ply"
        mesh.touch()
        interpreter = self.root / "proxy_env/bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(sys.executable)
        (proxy / "proxygs_run.json").write_text(json.dumps(dict(preparation=str(prep), mesh=str(mesh),
            source=str(self.source), python=str(interpreter), upstream=str(self.root))))
        plan = video.build_plan(self.args([native, proxy]))
        self.assertFalse(plan["models"][1]["antialiasing"])
        self.assertEqual(plan["models"][1]["train_count"], len(self.selected))
        with patch.object(video.os, "execv") as execute:
            video.replace_with_model(self.root / "trajectory.json", plan, 1)
        self.assertEqual(execute.call_args.args[0], str(interpreter))
        self.assertIn("--_plan", execute.call_args.args[1])
        with patch("scripts.proxygs_support.proxy_renderer", return_value=("renderer", None)) as load:
            self.assertEqual(video.core.load_renderer({"type": "proxygs", "iteration": 30000}, False), "renderer")
            self.assertTrue(load.call_args.kwargs["dynamic_depth"])
        (folder / "color_mlp.pt").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Incomplete checkpoint"):
            video.build_plan(self.args([native, proxy]))

    def test_proxy_depth_uses_each_requested_pose_and_output_intrinsics(self):
        calls = []
        class Renderer:
            def __init__(self, **kwargs): pass
            def render_depth_batched(self, mesh, parameters, **kwargs):
                calls.append((parameters, kwargs))
                return "depth", None
        helper = SimpleNamespace(DepthRenderer=Renderer, Load_ply_resource=lambda *args: "mesh",
            Build_Ply_Render_Camera_Parameters_colmap_correct=lambda *args: args)
        with patch.dict(sys.modules, {"Mesh2DepthHelper": helper}):
            depth = proxy_depth_renderer("mesh.ply")
            cam = SimpleNamespace(image_width=128, image_height=64, FoVx=1., FoVy=.5,
                znear=.01, zfar=100., R=np.array([[0,1,0],[-1,0,0],[0,0,1]]), T=np.array([1,2,3]))
            self.assertEqual(depth(cam), "depth")
            cam.T = np.array([4,5,6])
            depth(cam)
        self.assertEqual(len(calls), 2)
        self.assertAlmostEqual(calls[0][0][0], 128 / (2*np.tan(.5)))
        np.testing.assert_array_equal(calls[0][0][8], cam.R.T)
        np.testing.assert_array_equal(calls[1][0][9], [4,5,6])
        self.assertTrue(calls[0][1]["linear_depth"])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg required")
    def test_real_gt_and_multiple_panel_video_encoding(self):
        models = [self.model("baseline"), self.model("refined", self.refined())]
        seen = []
        def frames(spec, cameras, width, height, aa):
            seen.append((spec["path"].name, video.core.trajectory_record(cameras)))
            color = [0,255,0] if spec["path"] == models[0] else [255,0,0]
            for cam in cameras:
                yield np.broadcast_to(np.array(color, dtype=np.uint8), (height, width, 3)).copy()
        def replace(path, plan, index):
            video.execute_model(path, index)
        argv = ["--video", "actual", "--source", str(self.source), "--models", *map(str, models),
            "--output", str(self.root / "encoded"), "--actual-poses", "model", "--width", "64", "--max-frames", "2"]
        with patch.object(video, "replace_with_model", side_effect=replace), \
                patch.object(video.core, "rendered_frames", side_effect=frames), \
                patch.dict(sys.modules, {"tqdm": SimpleNamespace(tqdm=lambda xs, **kwargs: xs)}):
            self.assertEqual(video.main(argv), 0)
        output = self.root / "encoded"
        plan = json.loads((output / "trajectory.json").read_text())
        self.assertTrue(plan["complete"])
        self.assertEqual(plan["completed_models"], [0, 1])
        self.assertNotEqual(seen[0][1], seen[1][1])
        data = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
            "stream=width,height,nb_read_frames", "-of", "json", str(output / "side_by_side.mp4")], text=True))["streams"][0]
        self.assertEqual((data["width"], data["height"], int(data["nb_read_frames"])), (192,72,2))
        raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(output / "side_by_side.mp4"),
            "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(72,192,3)
        self.assertGreater(pixels[56,96,1], 240)
        self.assertGreater(pixels[56,160,0], 240)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            video.main(argv)

    def test_collector_keeps_unmeasured_runs_and_distinguishes_split_scores(self):
        a, b = self.model("measured"), self.model("pending")
        folder = a / "psnr_all_frames/iteration_30000"
        folder.mkdir(parents=True)
        (folder / "summary.json").write_text(json.dumps(dict(iteration=30000, antialiasing=True,
            counts=dict(all=16, selected=6, unselected=10), mean_psnr_db=dict(all=22., selected=23., unselected=21.))))
        views = b / "view_comparisons/iteration_30000/train"
        views.mkdir(parents=True)
        (views / "summary.json").write_text(json.dumps(dict(split="train", rendered_views=3, total_shared_views=6,
            models=[dict(path=str(b), iteration=30000, mean_psnr_db=24.)])))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            collect.main(["--root", str(self.root)])
        text = output.getvalue()
        self.assertIn("22.000", text)
        self.assertIn("| pending | 30000 | Y | N | 6 | — | — | — |", text)
        self.assertIn("| train | 3 | 6 | 24.000 |", text)


if __name__ == "__main__":
    unittest.main()
