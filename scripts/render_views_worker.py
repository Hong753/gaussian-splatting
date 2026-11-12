#!/usr/bin/env python3
"""One-model streaming renderer; Proxy-GS runs in its own Python environment."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.view_comparisons import atomic_json, image_id, model_info, save_panel
from scripts.occlusion_options import add_arguments, apply_overrides
from scripts import pose_options


def render_views(spec, names, iteration, output, antialiasing=None):
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
    cfg = spec["cfg"]
    pose_cache=pose_options.load_cache(spec,iteration)
    if spec["type"] == "proxygs":
        if antialiasing:
            raise ValueError("Proxy-GS does not support Graphdeco AA; use --no-antialiasing")
        from scripts.proxygs_support import proxy_renderer
        # Unselected original frames have no training depth cache. Rasterize the
        # proxy at the exact requested pose/resolution for every evaluated view.
        render, make_camera = proxy_renderer(spec, iteration, dynamic_depth=True)
    else:
        from scripts.render_model_comparison import load_renderer
        from utils.camera_utils import loadCam
        aa = antialiasing if antialiasing is not None else cfg.get("antialiasing", spec["type"] != "textured")
        if antialiasing is None and "antialiasing" not in cfg:
            print(f"Legacy config has no AA setting: using antialiasing={aa}; override with --no-antialiasing if needed.")
        filename = "textured.pt" if spec["type"] == "textured" else "point_cloud.ply"
        runtime = {**spec, "ply": spec["path"] / f"point_cloud/iteration_{iteration}" / filename}
        render = load_renderer(runtime, aa)
        camera_args = SimpleNamespace(resolution=cfg.get("resolution", -1), data_device="cpu", train_test_exp=False)
        def make_camera(record, index):
            info = SimpleNamespace(uid=index, image_path=record["image_path"],
                    image_name=record["name"], R=np.array(record["R"]), T=np.array(record["T"]),
                    FovX=record["fov_x"], FovY=record["fov_y"], depth_path="", depth_params=None,
                    is_test=False)
            return loadCam(camera_args, index, info, 1., False, False)
    output = Path(output)
    for name in ("gt", "renders", "errors", "comparisons"):
        (output / name).mkdir(parents=True, exist_ok=True)
    records = {}
    with torch.no_grad():
        for index, name in enumerate(tqdm(names, desc=f"Rendering {spec['path'].name}")):
            camera = pose_options.apply_camera(make_camera(spec["records"][name], index),pose_cache)
            pred = render(camera).clamp(0, 1)
            target = camera.original_image[:3].to(pred.device).clamp(0, 1)
            if pred.shape != target.shape or not torch.isfinite(pred).all():
                raise ValueError(f"Invalid rendered image for {name}")
            mse = (pred-target).square().flatten(1).mean(1)
            psnr = (10 * torch.log10(1 / mse)).mean().item()
            def rgb(tensor):
                return (tensor.detach().cpu().permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
            gt_array, pred_array = rgb(target), rgb(pred)
            gt, rendered = Image.fromarray(gt_array), Image.fromarray(pred_array)
            key = image_id(name)
            gt.save(output / "gt" / f"{key}.png")
            rendered.save(output / "renders" / f"{key}.png")
            Image.fromarray(rgb((pred-target).abs().mul(4).clamp(0, 1))).save(output / "errors" / f"{key}.png")
            label=spec['path'].name+(' | cached train pose' if pose_cache is not None and name in pose_cache['frames'] else ' | original pose')
            save_panel(gt, rendered, output / "comparisons" / f"{key}.jpg", label, psnr)
            records[name] = dict(gt=f"gt/{key}.png", image=f"renders/{key}.png",
                                 error=f"errors/{key}.png", comparison=f"comparisons/{key}.jpg",
                                 psnr_db=psnr if math.isfinite(psnr) else "inf",
                                 camera_pose='refined' if pose_cache is not None and name in pose_cache['frames'] else 'original',
                                 target_sha256=hashlib.sha256(str(gt_array.shape).encode() + gt_array.tobytes()).hexdigest())
            del camera, pred, target
    atomic_json(output / "frames.json", records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--antialiasing", action=argparse.BooleanOptionalAction, default=None)
    add_arguments(parser)
    pose_options.add_arguments(parser)
    args = parser.parse_args()
    render_views(pose_options.apply_overrides(apply_overrides(model_info(args.model), args),args), json.loads(args.selection.read_text())["names"],
                 args.iteration, args.output, args.antialiasing)


if __name__ == "__main__":
    main()
