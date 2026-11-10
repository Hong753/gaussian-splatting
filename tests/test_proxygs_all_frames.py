"""Exercise full-source Proxy-GS reporting with real CPU image metrics/encoding."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from scripts import render
from scripts.proxygs_support import prepare_dataset
from test_view_comparisons import checkpoint, dataset


class ProxyGSAllFramesTests(unittest.TestCase):
    def test_full_render_evaluates_unstaged_views_and_exports_separate_means(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, manifest, names, selected = dataset(root)
            model = root / "proxy"
            prep = prepare_dataset(source, manifest, model)
            (model / "cfg_args").write_text(f"Namespace(source_path={str(prep.parent)!r}, appearance_dim=0, resolution=-1)")
            (model / "proxygs_run.json").write_text(json.dumps(dict(preparation=str(prep), source=str(source))))
            checkpoint(model, 30000)
            seen, expected = [], {}

            def make_camera(record, index):
                seen.append(record["name"])
                with Image.open(record["image_path"]) as image:
                    pixels = np.asarray(image).copy().astype(np.float32) / 255
                # Independent numpy metric, before image encoding/rounding.
                prediction = pixels * .6 + .1
                error = np.mean(np.square(prediction - pixels), axis=(0, 1))
                expected[record["name"]] = float(np.mean(10 * np.log10(1 / error)))
                return SimpleNamespace(original_image=torch.from_numpy(pixels).permute(2, 0, 1))

            def prediction(camera):
                return camera.original_image * .6 + .1

            with patch("scripts.proxygs_support.proxy_renderer", return_value=(prediction, make_camera)) as load, \
                    patch.object(render.subprocess, "run") as supervisor, \
                    patch.dict(sys.modules, {"tqdm": SimpleNamespace(tqdm=lambda xs, **kwargs: xs)}), \
                    contextlib.redirect_stdout(io.StringIO()):
                render.main(["--model", str(model), "--split", "all", "--in-process", "--iteration", "30000"])
            supervisor.assert_not_called()
            self.assertTrue(load.call_args.kwargs["dynamic_depth"])
            self.assertEqual(seen, sorted(names))
            summary = json.loads((model / "psnr_all_frames/iteration_30000/summary.json").read_text())
            self.assertEqual(summary["source"], str(source))
            self.assertEqual(summary["counts"], dict(all=16, selected=6, unselected=10))
            self.assertFalse(summary["antialiasing"])
            for group, group_names in dict(all=names, selected=selected,
                                           unselected=set(names) - set(selected)).items():
                self.assertAlmostEqual(summary["mean_psnr_db"][group],
                                       np.mean([expected[name] for name in group_names]), places=4)
            with (model / "psnr_all_frames/iteration_30000/per_frame.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({r["image_name"] for r in rows if r["selected_for_training"] == "1"}, set(selected))
            gallery = model / "view_comparisons/iteration_30000/all"
            self.assertEqual(len(list((gallery / "model_0/renders").glob("*.png"))), 16)
            self.assertTrue((gallery / "index.html").is_file())


if __name__ == "__main__":
    unittest.main()
