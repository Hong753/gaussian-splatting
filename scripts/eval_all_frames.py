#!/usr/bin/env python3
"""Evaluate a saved COLMAP reconstruction on every registered image.

Images are loaded one at a time. No per-image exposure is applied, giving all
frames the same base-scene rendering protocol, including unseen images.
"""
import argparse
import ast
import csv
import json
import os
from pathlib import Path
import sys
import uuid
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.occlusion_options import add_arguments, apply_overrides, configure_pipeline, metric_directory, settings


def read_config(path):
    node = ast.parse(path.read_text(), mode="eval").body
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "Namespace" and not node.args
            and all(k.arg is not None for k in node.keywords)):
        raise ValueError(f"Unsupported config format: {path}")
    return {k.arg: ast.literal_eval(k.value) for k in node.keywords}


def choose_iteration(model, iteration, model_type="3dgs"):
    root = model / "point_cloud"
    filename = "textured.pt" if model_type == "textured" else "point_cloud.ply"
    if iteration < 0:
        available = [int(p.name.removeprefix("iteration_")) for p in root.glob("iteration_*")
                     if p.name.removeprefix("iteration_").isdigit() and (p / filename).is_file()]
        if not available:
            raise FileNotFoundError(f"No saved {filename} under {root}")
        iteration = max(available)
    ply = root / f"iteration_{iteration}" / filename
    if not ply.is_file():
        raise FileNotFoundError(f"Saved Gaussian model not found: {ply}")
    return iteration, ply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=-1, help="Saved iteration, or -1 for latest")
    parser.add_argument("--antialiasing", action=argparse.BooleanOptionalAction, default=True)
    add_arguments(parser)
    args = parser.parse_args()
    model, source = args.model.resolve(), args.source.resolve()
    cfg = read_config(model / "cfg_args")
    model_type = cfg.get("model_type", "3dgs")
    spec = apply_overrides(dict(path=model, cfg=cfg, type=model_type), args)
    cfg = spec["cfg"]
    iteration, ply = choose_iteration(model, args.iteration, model_type)
    manifest_path = model / "keyframes.json"
    manifest_name = cfg.get("keyframe_manifest", "")
    if manifest_name and not manifest_path.is_file():
        raise FileNotFoundError(f"Training used keyframes but {manifest_path} is missing")
    selected = set(json.loads(manifest_path.read_text())["selected_images"]) if manifest_name else None
    images = cfg.get("images", "images")

    import torch
    from tqdm import tqdm
    from scene.colmap_loader import (read_extrinsics_binary, read_intrinsics_binary,
                                     read_extrinsics_text, read_intrinsics_text)
    from scene.dataset_readers import readColmapCameras
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from utils.camera_utils import loadCam
    from utils.colmap_paths import colmap_model_dir
    from utils.image_utils import psnr
    try:
        from diff_gaussian_rasterization import SparseGaussianAdam
        separate_sh = True
    except ImportError:
        separate_sh = False

    colmap = Path(colmap_model_dir(source))
    if (colmap / "images.bin").is_file() and (colmap / "cameras.bin").is_file():
        extrinsics = read_extrinsics_binary(str(colmap / "images.bin"))
        intrinsics = read_intrinsics_binary(str(colmap / "cameras.bin"))
    else:
        extrinsics = read_extrinsics_text(str(colmap / "images.txt"))
        intrinsics = read_intrinsics_text(str(colmap / "cameras.txt"))
    cameras = sorted(readColmapCameras(extrinsics, intrinsics, None, str(source / images), "", []),
                     key=lambda camera: camera.image_name)
    if not cameras:
        raise ValueError("No registered COLMAP images found")
    if selected is None:
        if cfg.get("eval", False):
            selected = {camera.image_name for index, camera in enumerate(cameras) if index % 8 != 0}
        else:
            selected = {camera.image_name for camera in cameras}
    if selected is not None and not selected.issubset({c.image_name for c in cameras}):
        raise ValueError("Some selected training images are absent from the current COLMAP model")

    if model_type == "scaffold":
        from models.scaffold_gaussian import ScaffoldGaussianModel
        from models.scaffold_renderer import render as render_scaffold, prefilter_voxel
        gaussians = ScaffoldGaussianModel(feat_dim=cfg.get("scaffold_feat_dim", 32),
                                           n_offsets=cfg.get("scaffold_n_offsets", 10),
                                           voxel_size=cfg.get("scaffold_voxel_size", 0.001),
                                           update_init_factor=16, appearance_dim=0)
        gaussians.load_ply(str(ply))
        gaussians.eval()
    elif model_type == "3dgs":
        gaussians = GaussianModel(cfg.get("sh_degree", 3))
        gaussians.load_ply(str(ply))
    elif model_type == "textured":
        if args.antialiasing:
            raise ValueError("Textured Gaussians does not support Graphdeco anti-aliasing; pass --no-antialiasing")
        from models.textured_gaussian import TexturedGaussianModel
        gaussians = TexturedGaussianModel.load(ply)
    else:
        raise ValueError(f"Unsupported saved model type: {model_type}")
    pipeline = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False,
                               debug=False, antialiasing=args.antialiasing)
    configure_pipeline(pipeline, cfg)
    camera_args = SimpleNamespace(resolution=cfg.get("resolution", -1), data_device="cpu", train_test_exp=False)
    color = [1, 1, 1] if cfg.get("white_background", False) else [0, 0, 0]
    background = torch.tensor(color, dtype=torch.float32, device="cuda")
    output = metric_directory(model, iteration, spec.get("render_variant", ""))
    output.mkdir(parents=True, exist_ok=True)
    # Concurrent ablations can evaluate the same baseline on different GPUs.
    token = uuid.uuid4().hex
    csv_tmp, json_tmp = output / f"per_frame.csv.{token}.tmp", output / f"summary.json.{token}.tmp"
    sums, counts = {k: 0.0 for k in ("all", "selected", "unselected")}, {k: 0 for k in ("all", "selected", "unselected")}
    culled_sum, culled_views = 0., 0
    with torch.no_grad(), csv_tmp.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_name", "selected_for_training", "psnr_db", "occlusion_culled_fraction"])
        for idx, info in enumerate(tqdm(cameras, desc="Evaluating all frames", unit="frame")):
            culled_fraction = None
            camera = loadCam(camera_args, idx, info, 1.0, False, False)
            if model_type == "scaffold":
                visible = prefilter_voxel(camera, gaussians, pipeline, background)
                pred = render_scaffold(camera, gaussians, pipeline, background, visible_mask=visible)["render"].clamp(0, 1)
            elif model_type == "textured":
                pred = gaussians.render(camera, background)[0].clamp(0, 1)
            else:
                rendered = render(camera, gaussians, pipeline, background, separate_sh=separate_sh,
                                  use_trained_exp=False)
                pred = rendered["render"].clamp(0, 1)
                if "occlusion_keep" in rendered:
                    culled_fraction = 1-rendered["occlusion_keep"].float().mean().item()
                    culled_sum += culled_fraction
                    culled_views += 1
            target = camera.original_image.to("cuda").clamp(0, 1)
            value = psnr(pred, target).mean().item()
            is_selected = (info.image_name in selected) if selected is not None else None
            writer.writerow([info.image_name, "" if is_selected is None else int(is_selected), value, culled_fraction])
            sums["all"] += value
            counts["all"] += 1
            if is_selected is not None:
                group = "selected" if is_selected else "unselected"
                sums[group] += value
                counts[group] += 1
            del camera, pred, target
    means = {k: sums[k] / counts[k] if counts[k] else None for k in sums}
    means = {k: "inf" if value == float("inf") else value for k, value in means.items()}
    summary = {"iteration": iteration, "source": str(source), "model": str(model), "model_type": model_type,
               "exposure_protocol": "base appearance; no per-image exposure applied",
               "antialiasing": args.antialiasing, "resolution_setting": camera_args.resolution,
               "selection_manifest": str(manifest_path) if manifest_name else None,
               "occlusion": settings(cfg), "render_variant": spec.get("render_variant", ""),
               "training_occlusion_mode": spec.get("training_occlusion_mode", "off"),
               "inference_occlusion_policy": cfg.get("occlusion_inference_mode", "saved"),
               "training_depth_occlusion_dir": cfg.get("train_depth_occlusion_dir", ""),
               "mean_occlusion_culled_fraction": culled_sum/culled_views if culled_views else None,
               "counts": counts, "mean_psnr_db": means}
    json_tmp.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    os.replace(csv_tmp, output / "per_frame.csv")
    os.replace(json_tmp, output / "summary.json")
    print(json.dumps(summary, indent=2))
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()


