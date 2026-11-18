#!/usr/bin/env python3
"""Render a fixed-LiDAR checkpoint before and after a local scale cap."""
import argparse
import html
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_all_frames import choose_iteration
from scripts.view_comparisons import atomic_json, image_id, model_info
from utils.local_scale import distribution


def initial_sigma(source, centers):
    """3DGS initializes each axis from the RMS distance to three other input points."""
    from scipy.spatial import cKDTree
    from tqdm import trange
    if len(source) < 4 or not len(centers):
        raise ValueError("Need at least four input points and one surviving Gaussian")
    tree = cKDTree(source)
    result = np.empty(len(centers), np.float32)
    for start in trange(0, len(centers), 65536, desc="Matching fixed centers to original LiDAR"):
        rows = slice(start, start + 65536)
        distances, _ = tree.query(centers[rows], k=4, workers=4)
        if np.any(distances[:, 0] != 0):
            raise ValueError("Checkpoint contains centers absent from the original LiDAR. Check the model and input PLY.")
        result[rows] = np.sqrt(np.maximum(np.mean(distances[:, 1:] ** 2, axis=1), 1e-7))
    return result


def cap_outliers(log_axes, spacing, ratio, percent):
    if not math.isfinite(ratio) or ratio <= 0 or not 0 < percent <= 100:
        raise ValueError("Cap ratio must be positive and tail percentage in (0, 100]")
    if not np.isfinite(log_axes).all() or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Invalid checkpoint scales or LiDAR spacing")
    log_growth = log_axes.max(axis=1) - np.log(spacing)
    count = math.ceil(len(spacing) * percent / 100)
    ids = np.flatnonzero(log_growth > math.log(ratio))
    if len(ids) > count:
        ids = ids[np.argsort(-log_growth[ids], kind="stable")[:count]]
    capped = np.minimum(log_axes[ids], (np.log(spacing[ids]) + math.log(ratio))[:, None])
    stats = dict(total=len(spacing), capped=len(ids), max_percent=percent, cap_ratio=ratio,
                 initial_sigma=distribution(spacing), learned_max_sigma=distribution(np.exp(log_axes.max(axis=1))),
                 largest_axis_over_initial=distribution(np.exp(log_growth)))
    return ids, capped, stats


def training_policy(spec, iteration):
    from utils.densification_diagnostics import training_extent
    opt = spec["cfg"].get("optimization_settings", {})
    extent = training_extent(spec["records"], spec["splits"]["train"])
    policy = dict(camera_extent=extent, clone_split_sigma_if_growth_enabled=opt.get("percent_dense", .01) * extent,
                  world_prune_sigma=opt.get("prune_world_scale_ratio", .1) * extent,
                  screen_prune_radius_px=opt.get("prune_screen_size", 20), saved_optimization_settings=opt)
    log = spec["path"] / "densification.jsonl"
    if log.is_file():
        with log.open() as stream:
            for line in stream:
                if not line.endswith("\n"):
                    break  # A running trainer may still be appending the last event.
                event = json.loads(line)
                if event["iteration"] > iteration:
                    break
                if event.get("prune_world_sigma", 0) > 0:
                    policy["last_logged_size_check"] = {key: event.get(key) for key in
                        ("iteration", "scene_extent", "split_sigma_threshold", "growth_enabled",
                         "prune_world_sigma", "prune_world_candidates", "prune_screen_radius",
                         "prune_screen_candidates", "radius_history_preserved")}
    return policy


def select_frames(spec, reference, count):
    train = sorted(spec["splits"]["train"])
    if reference and reference not in spec["records"]:
        raise ValueError(f"Unregistered reference image: {reference}")
    frames = [reference] if reference else []
    candidates = [name for name in train if name != reference]
    if reference and candidates and count:
        # Include a nearby training view in filename/trajectory order for this sequence.
        order = {name: i for i, name in enumerate(sorted(spec["records"]))}
        nearest = min(candidates, key=lambda name: abs(order[name] - order[reference]))
        frames.append(nearest)
        candidates.remove(nearest)
        count -= 1
    for i in np.linspace(0, len(candidates) - 1, min(count, len(candidates)), dtype=int):
        frames.append(candidates[i])
    if not frames:
        raise ValueError("No frames selected")
    return frames


def rgb8(array):
    return np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)


def compare_area(output, key, gt, base, capped, alpha, capped_alpha):
    import cv2
    from scripts.diagnose_floor_blur import salience
    # Ignore boundary padding in the Sobel measurement.
    mask = np.zeros(gt.shape[:2], bool)
    mask[1:-1, 1:-1] = True
    def gray(rgb):
        return cv2.cvtColor(rgb8(rgb), cv2.COLOR_RGB2GRAY)
    reference = gray(gt)
    def metrics(rgb):
        mse = float(np.mean((rgb - gt) ** 2, dtype=np.float64))
        return dict(psnr_db=-10 * math.log10(mse) if mse else "inf",
                    **salience(reference, gray(rgb), mask))
    loss = np.maximum(alpha - capped_alpha, 0)
    metrics_row = dict(baseline=metrics(base), capped=metrics(capped),
                       mean_absolute_rgb_change=float(np.abs(capped - base).mean()),
                       mean_alpha_loss=float(loss.mean()),
                       new_hole_percent=100 * float(np.mean((alpha >= .9) & (capped_alpha < .5))))
    imgs = {"Actual": gt, "Original": base, "Capped": capped,
            "Original error ×4": 4 * np.abs(gt - base), "Capped error ×4": 4 * np.abs(gt - capped),
            "Opacity lost ×4": 4 * loss}
    files = {}
    for i, (title, array) in enumerate(imgs.items()):
        path = output / f"{key}_{i}.png"
        Image.fromarray(rgb8(array)).save(path)
        files[title] = path.name
    return dict(**metrics_row, images=files)


def gallery(output, report):
    stats, policy = report["scales"], report["training"]
    sections = []
    for row in report["views"]:
        for area, result in row["areas"].items():
            def value(state, metric):
                v = result[state][metric]
                return f"{v:.3f}" if isinstance(v, (float, int)) else ("n/a" if v is None else v)
            caption = " · ".join(f"{label} {value('baseline', metric)} → {value('capped', metric)}"
                                for label, metric in (("PSNR (dB)", "psnr_db"), ("Edge agreement", "gradient_agreement")))
            pictures = "".join(f'<figure><figcaption>{title}</figcaption><a href="{path}"><img src="{path}" loading="lazy"></a></figure>'
                               for title, path in result["images"].items())
            sections.append(f'<h2>{html.escape(row["name"])} · {row["split"]} · {area}</h2><p>{caption}'
                            f' · New holes {result["new_hole_percent"]:.3f}%</p><section>{pictures}</section>')
    page = f"""<!doctype html><meta charset="utf-8"><title>LiDAR scale comparison</title>
<style>body{{background:#17212a;color:#eee;font:16px system-ui;margin:2vw}}section{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}}figure{{margin:0}}img{{width:100%}}a{{color:#9cf}}</style>
<h1>LiDAR scale comparison · iteration {report['iteration']}</h1>
<p>{stats['capped']:,} of {stats['total']:,} splats capped. Largest {stats['max_percent']:g}% by growth eligible.
Each selected axis is limited to {stats['cap_ratio']:g}× its initialization sigma.</p>
<p>Training camera extent {policy['camera_extent']:.4g} scene units. Configured world-prune sigma {policy['world_prune_sigma']:.4g} scene units.</p>
<p>Compare edges against the actual image. White in the opacity-loss map marks reduced coverage.
New holes count pixels whose accumulated opacity went from ≥0.9 to &lt;0.5. Click any image for full resolution.</p>
<a href="report.json">Metrics and settings</a>{''.join(sections)}"""
    (output / "index.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--reference", help="Optional registered image, including held-out frames")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"), help="Native pixels of reference only")
    parser.add_argument("--train-views", type=int, default=4)
    parser.add_argument("--cap-ratio", type=float, default=4)
    parser.add_argument("--top-percent", type=float, default=1)
    parser.add_argument("--output", type=Path, help="Empty directory, default: new directory under MODEL/scale_cap_audit")
    args = parser.parse_args()
    if args.train_views < 0 or (args.roi and not args.reference):
        parser.error("Use a nonnegative --train-views and supply --reference with --roi")
    spec = model_info(args.model)
    cfg = spec["cfg"]
    opt = cfg.get("optimization_settings", {})
    if spec["type"] != "3dgs" or (opt.get("freeze_lidar_positions") is False and not opt.get("fixed_lidar")):
        parser.error("Choose a native fixed-center LiDAR checkpoint")
    names = select_frames(spec, args.reference, args.train_views)
    if args.roi:
        x, y, w, h = args.roi
        cam = spec["records"][args.reference]
        if min(x, y) < 0 or min(w, h) < 3 or x + w > cam['width'] or y + h > cam['height']:
            parser.error("Reference ROI is outside the native image or too small for edge metrics")
    iteration, ply = choose_iteration(spec["path"], args.iteration)
    import torch
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from scripts.render_views_worker import load_record_camera
    from scripts.occlusion_options import configure_pipeline, inference_settings
    from plyfile import PlyData
    from tqdm import tqdm
    print(f"Loading {ply}", flush=True)
    model = GaussianModel(cfg.get("sh_degree", 3))
    model.load_ply(str(ply))
    source = spec["path"] / "input.ply"
    vertices = PlyData.read(str(source), mmap="r")["vertex"]
    xyz = np.column_stack([vertices[k] for k in ("x", "y", "z")]).astype(np.float32, copy=False)
    spacing = initial_sigma(xyz, model.get_xyz.detach().cpu().numpy())
    del xyz, vertices
    ids, capped, stats = cap_outliers(model._scaling.detach().cpu().numpy(), spacing, args.cap_ratio, args.top_percent)
    device = model.get_xyz.device
    ids = torch.as_tensor(ids, device=device)
    originals = model._scaling.detach()[ids]
    capped = torch.as_tensor(capped, device=device)
    policy = training_policy(spec, iteration)
    print(f"Camera extent {policy['camera_extent']:.5g}; clone/split sigma (adaptive only) "
          f"{policy['clone_split_sigma_if_growth_enabled']:.5g}; world-prune sigma "
          f"{policy['world_prune_sigma']:.5g} scene units. Capping {stats['capped']:,}/{stats['total']:,} splats",
          flush=True)
    root = spec["path"] / "scale_cap_audit"
    if args.output:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise ValueError(f"Choose an empty output directory: {output}")
    else:
        root.mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix=f"iter{iteration}_", dir=root))
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False,
                           antialiasing=cfg.get("antialiasing", True))
    configure_pipeline(pipe, inference_settings(cfg))
    camera_args = SimpleNamespace(resolution=1, data_device="cpu", train_test_exp=False)
    black = torch.zeros(3, device=device)
    bg = torch.ones_like(black) if cfg.get("white_background") else black
    import diff_gaussian_rasterization
    separate_sh = hasattr(diff_gaussian_rasterization, "SparseGaussianAdam")
    white_splats = torch.ones_like(model.get_xyz)
    def render_pair(camera):
        pred = render(camera, model, pipe, bg, separate_sh=separate_sh)["render"]
        coverage = render(camera, model, pipe, black, override_color=white_splats)["render"][0]
        if not torch.isfinite(pred).all() or not torch.isfinite(coverage).all():
            raise ValueError(f"Nonfinite render for {camera.image_name}")
        return pred.permute(1, 2, 0).cpu().numpy(), coverage.cpu().numpy()
    report = dict(model=str(spec["path"]), input_ply=str(source), iteration=iteration, roi=args.roi,
                  training=policy, scales=stats, antialiasing=pipe.antialiasing,
                  protocol="Same original COLMAP camera at native resolution and base exposure for both renders", views=[])
    with torch.no_grad():
        for index, name in enumerate(tqdm(names, desc="Rendering original, capped, and coverage")):
            camera = load_record_camera(camera_args, spec["records"][name], index)
            gt = camera.original_image[:3].permute(1, 2, 0).cpu().numpy()
            base, alpha = render_pair(camera)
            try:
                model._scaling[ids] = capped
                changed, changed_alpha = render_pair(camera)
            finally:
                model._scaling[ids] = originals
            arrays = (gt, base, changed, alpha, changed_alpha)
            key = image_id(name)
            areas = {}
            if args.roi and name == args.reference:
                areas["Selected crop"] = compare_area(output, key + "_roi", *(a[y:y+h, x:x+w] for a in arrays))
            areas["Full frame"] = compare_area(output, key, *arrays)
            report["views"].append(dict(name=name, split="train" if name in spec["splits"]["train"] else "held-out", areas=areas))
    atomic_json(output / "report.json", report)
    gallery(output, report)
    print(output / "index.html", flush=True)


if __name__ == "__main__":
    main()
