"""CPU regression tests for split fidelity and comparable report generation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts import render
from scripts.view_comparisons import (
    atomic_json, image_id, model_info, read_colmap_records, shared_iteration,
    split_names, write_gallery,
)


def dataset(root):
    source = root / "source"
    sparse = source / "sparse"
    sparse.mkdir(parents=True)
    # Actual images are half the calibration resolution, as in pre-resized data.
    (sparse / "cameras.txt").write_text("1 PINHOLE 32 16 24 20 16 8\n")
    names = [f"cam{index // 8}/frame{index % 8:02d}.png" for index in range(16)]
    rows = []
    for index, name in enumerate(names):
        image = source / "images" / name
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 8), (index * 8, 20, 80)).save(image)
        rows.append(f"{index + 1} 0.7071067811865476 0 0 0.7071067811865476 {index} 2 3 1 {name}\n\n")
    (sparse / "images.txt").write_text("".join(rows))
    (sparse / "points3D.ply").write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
    selected = [names[i] for i in (1, 2, 7, 9, 10, 15)]
    manifest = root / "keyframes.json"
    atomic_json(manifest, dict(source_path=str(source), eval=True, selected_images=selected))
    return source, manifest, names, selected


def checkpoint(path, iteration):
    saved = path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.touch()


class ViewComparisonsTests(unittest.TestCase):
    def test_concurrent_reports_use_independent_atomic_temporary_files(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        barrier = Barrier(2)
        original_replace = Path.replace
        def synchronized_replace(path, target):
            barrier.wait(timeout=5)
            return original_replace(path, target)
        with tempfile.TemporaryDirectory() as tmp, patch.object(Path, 'replace', synchronized_replace):
            target = Path(tmp)/'summary.json'
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(atomic_json, target, dict(worker=i, values=list(range(100)))) for i in range(2)]
                for future in futures:
                    future.result()
            saved = json.loads(target.read_text())
            self.assertIn(saved['worker'], [0, 1])
            self.assertEqual(saved['values'], list(range(100)))
            self.assertEqual(list(Path(tmp).glob('*.tmp')), [])

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source, self.manifest, self.names, self.selected = dataset(self.root)

    def model(self, name):
        path = self.root / name
        path.mkdir(parents=True)
        (path / "cfg_args").write_text(
            f"Namespace(source_path={str(self.source)!r}, eval=True, keyframe_manifest='keyframes.json', antialiasing=True)")
        (path / "keyframes.json").write_bytes(self.manifest.read_bytes())
        checkpoint(path, 7000)
        return path

    def test_train_membership_and_heldout_never_include_unused_candidates(self):
        info = model_info(self.model("a"))
        self.assertEqual(info["splits"]["train"], set(self.selected))
        self.assertEqual(info["splits"]["eval"], set(self.names[::8]))
        self.assertEqual(info["splits"]["all"], set(self.names))
        self.assertNotIn(self.names[3], info["splits"]["eval"])
        for selected in ([], [self.names[0]], [self.selected[0]] * 2, ["missing.png"]):
            with self.assertRaises(ValueError):
                split_names(self.names, {"eval": True}, {"selected_images": selected})

    def test_common_checkpoint_and_unique_nested_run_discovery(self):
        a, b = self.model("original/kf_original"), self.model("refined/kf_refined")
        checkpoint(a, 30000)
        specs = [model_info(a), model_info(b)]
        self.assertEqual(shared_iteration(specs, -1), 7000)
        with self.assertRaises(FileNotFoundError):
            shared_iteration(specs, 30000)
        self.assertEqual(render.resolve_models(None, self.root, ["kf_original", "kf_refined"]), [a, b])
        self.model("duplicate/kf_original")
        with self.assertRaises(ValueError):
            render.resolve_models(None, self.root, ["kf_original"])

    def test_cpu_reader_keeps_paths_and_nontrivial_calibration(self):
        records = read_colmap_records(self.source)
        self.assertEqual(set(records), set(self.names))
        record = records[self.names[1]]
        np.testing.assert_allclose(record["R"], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], atol=1e-14)
        self.assertEqual(record["T"], [1, 2, 3])
        self.assertAlmostEqual(record["fov_x"], 2*np.arctan(32/48))

    def fake_worker(self, mismatched=False):
        def run(command, **kwargs):
            folder = Path(command[command.index("--output") + 1])
            selection = Path(command[command.index("--selection") + 1])
            rows = {}
            for name in json.loads(selection.read_text())["names"]:
                key = image_id(name)
                files = {kind: f"{kind}/{key}.png" for kind in ("gt", "image", "error", "comparison")}
                for relative in files.values():
                    target = folder / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (16, 8), "red").save(target)
                rows[name] = dict(**files, psnr_db=22.5,
                                  target_sha256=folder.name if mismatched else name)
            atomic_json(folder / "frames.json", rows)
        return run

    def test_controller_generates_both_reports_with_shared_targets(self):
        a, b, output = self.model("a"), self.model("b"), self.root / "report"
        with patch.object(render.subprocess, "run", side_effect=self.fake_worker()) as worker, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(render.main(["--models", str(a), str(b), "--split", "both", "--output", str(output)]), 0)
        self.assertEqual(worker.call_count, 4)
        for split, expected in (("train", self.selected), ("eval", self.names[::8])):
            summary = json.loads((output / split / "summary.json").read_text())
            self.assertEqual([row["name"] for row in summary["frames"]], sorted(expected))
            self.assertEqual(summary["models"][0]["mean_psnr_db"], 22.5)
            for row in summary["frames"]:
                self.assertTrue((output / split / row["gt"]).is_file())
                self.assertEqual(len(row["renders"]), 2)
        self.assertTrue((output / "index.html").is_file())

    def test_controller_rejects_unmatched_ground_truth(self):
        a, b, output = self.model("a"), self.model("b"), self.root / "bad"
        with patch.object(render.subprocess, "run", side_effect=self.fake_worker(True)), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                render.main(["--models", str(a), str(b), "--train", "--output", str(output)])
        self.assertEqual(result.exception.code, 1)
        self.assertFalse((output / "train/index.html").exists())

    def test_all_frame_metrics_preserve_train_membership_and_ignore_sampled_runs(self):
        model = self.model("all_frames")
        with patch.object(render.subprocess, "run", side_effect=self.fake_worker()), contextlib.redirect_stdout(io.StringIO()):
            render.main(["--model", str(model), "--split", "all"])
        summary_path = model / "psnr_all_frames/iteration_7000/summary.json"
        before = summary_path.read_bytes()
        summary = json.loads(before)
        self.assertEqual(summary["counts"], dict(all=16, selected=6, unselected=10))
        self.assertEqual(summary["mean_psnr_db"], dict(all=22.5, selected=22.5, unselected=22.5))
        csv_path = summary_path.with_name("per_frame.csv")
        self.assertEqual(len(csv_path.read_text().splitlines()), 17)
        with patch.object(render.subprocess, "run", side_effect=self.fake_worker()), contextlib.redirect_stdout(io.StringIO()):
            render.main(["--model", str(model), "--split", "all", "--max-views", "3"])
        self.assertEqual(summary_path.read_bytes(), before)
        self.assertEqual(len(csv_path.read_text().splitlines()), 17)

    def test_shared_subset_does_not_claim_full_coverage_for_larger_model(self):
        a, b = self.model("subset"), self.model("full")
        specs = {p: model_info(p) for p in (a, b)}
        specs[a]["splits"]["all"] = set(self.names[:4])
        specs[a]["splits"]["train"] = set(self.selected) & set(self.names[:4])
        with patch.object(render, "model_info", side_effect=lambda p: specs[p]), \
                patch.object(render.subprocess, "run", side_effect=self.fake_worker()), contextlib.redirect_stdout(io.StringIO()):
            render.main(["--models", str(a), str(b), "--split", "all", "--output", str(self.root / "subset_report")])
        self.assertTrue((a / "psnr_all_frames/iteration_7000/summary.json").is_file())
        self.assertFalse((b / "psnr_all_frames").exists())

    def test_dry_run_makes_no_output_and_no_gpu_call(self):
        a, output = self.model("a"), self.root / "dry"
        with patch.object(render.subprocess, "run") as worker, contextlib.redirect_stdout(io.StringIO()):
            render.main(["train", "--model", str(a), "--output", str(output), "--dry-run"])
        worker.assert_not_called()
        self.assertFalse(output.exists())

    def test_in_process_report_renders_both_splits_without_worker_processes(self):
        model, output = self.model("inline"), self.root / "inline_report"
        def render_inline(spec, names, iteration, destination, antialiasing):
            self.assertEqual(spec["path"], model)
            self.assertEqual(iteration, 7000)
            self.fake_worker()(["--output", str(destination), "--selection", str(destination.parent / "selection.json")])
        with patch("scripts.render_views_worker.render_views", side_effect=render_inline) as inline, \
                patch.object(render.subprocess, "run") as worker, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(render.main(["--model", str(model), "--split", "both", "--in-process", "--output", str(output)]), 0)
        self.assertEqual(inline.call_count, 2)
        worker.assert_not_called()
        self.assertTrue((output / "train/index.html").is_file())
        self.assertTrue((output / "eval/index.html").is_file())

    def test_gallery_escapes_embedded_filenames(self):
        rows = [dict(name="</script><script>alert(1)</script>.png", gt="gt.png", renders=[])]
        write_gallery(self.root / "html", "A < B", rows, [], {})
        html = (self.root / "html/index.html").read_text()
        self.assertNotIn("<script>alert(1)", html)
        self.assertIn("A &lt; B", html)


if __name__ == "__main__":
    unittest.main()
