#!/usr/bin/env python3
"""Orange4F physical split-size ablation and CPU-only floor spacing audit."""
import argparse
import shutil
from contextlib import contextmanager
import fcntl
import html
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

# The downloadable script can be run from a checkout without installing it.
ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "scripts/eval_all_frames.py").is_file():
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from scripts.eval_all_frames import choose_iteration
from scripts.view_comparisons import atomic_json, model_info, read_colmap_records, split_names
from utils.checkpoint_snapshots import digest, read_snapshot

SUITE = Path("/data/3dgs_outputs/orange4f/3dgs_finetuning/stride3_noaa_v1")
REGIONS = Path("/data/3dgs_outputs/orange_4F/1_full/3dgs_finetuning/regions")
# Native image pixels and fitted LiDAR plane from the user's saved diagnostic.
FLOOR_SOURCE = Path("/data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full")
FRAME = "12569.jpg"
ROI = (96, 1260, 1275, 267)
CENTER = np.array([-4.985959985044178, 1.807263625483172, -1.0653450191492186])
NORMAL = np.array([-0.009408616379497275, 0.010765564382005217, 0.9998977850567328])
TOLERANCE = .025
STEPS = (30000, 50000)
PROBE_STEPS = (2000,)


def read_json(path):
    return json.loads(path.read_text())


def regional_model(path):
    if path is not None:
        return path.expanduser().resolve()
    names = ("crop_adaptive_lr05_v1", "crop_adaptive_lr0.05_v1")
    for region in ("wall_06663_v1", "wall_06663_auto_v1"):
        for name in names:
            candidate = REGIONS / region / "runs" / name
            if (candidate / "cfg_args").is_file():
                return candidate
    matches = sorted(REGIONS.glob("wall_06663*/runs/crop*adaptive*/cfg_args"))
    if len(matches) == 1:
        return matches[0].parent
    original_export = REGIONS / "wall_06663_v1"
    if not matches and (original_export / "selection.json").is_file():
        return original_export
    raise ValueError(f"Specify --regional PATH to the sharp region. Candidates: {[str(p.parent) for p in matches]}")


def regional_reference(path):
    if (path / "cfg_args").is_file():
        return model_info(path)
    # If only the original export exists, the physical threshold is a proposed control.
    # No adaptive operation or measured training cutoff is attributed to a fixed run.
    selection = read_json(path / "selection.json")
    manifest = read_json(path / "keyframes.json")
    if set(manifest["selected_images"]) != {r["name"] for r in selection["selection"] if r["split"] == "train"}:
        raise ValueError("Regional export disagrees with its training manifest")
    records = read_colmap_records(path)
    return dict(path=path, cfg={"optimization_settings": {"percent_dense": .01}}, type="3dgs",
                export_only=True, records=records, splits=split_names(records, {}, manifest))


def split_settings(spec):
    cfg = spec["cfg"]
    opt = cfg.get("optimization_settings", {})
    if spec["type"] != "3dgs" or any(opt.get(k) for k in ("freeze_lidar_positions", "fixed_lidar", "discrete_lidar")):
        raise ValueError(f"Choose an adaptive 3DGS model for the split comparison: {spec['path']}")
    log = spec["path"] / "densification.jsonl"
    if log.is_file() and not spec.get("export_only"):
        with log.open() as lines:
            for line in lines:
                if not line.endswith("\n"):
                    break
                event = json.loads(line)
                if event.get("growth_enabled") is False:
                    raise ValueError(f"Growth disabled in {log}")
                if all(k in event for k in ("scene_extent", "split_sigma_threshold")):
                    return positive(event["scene_extent"]), positive(event["split_sigma_threshold"]), str(log)
    from utils.densification_diagnostics import training_extent
    extent = positive(training_extent(spec["records"], spec["splits"]["train"]))
    origin = ("proposed 0.01 × regional training-camera extent (export only; no measured adaptive run)"
              if spec.get("export_only") else "recomputed from selected cameras and saved percent_dense")
    return extent, extent * positive(opt["percent_dense"]), origin


def positive(value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid physical scale {value}")
    return value


def regional_cutoff(base, region, extent, sigma):
    """Express a regional length in full-scene units using the same named cameras."""
    names = region["splits"]["train"]
    shared = sorted(base["records"].keys() & names)
    if len(shared) < 2:
        raise ValueError(f"Only {len(shared)}/{len(names)} regional training views occur in the full model. "
                         "Select a regional export of the same capture with --regional.")
    def centers(spec):
        return np.stack([-np.asarray(spec["records"][n]["R"]) @ spec["records"][n]["T"] for n in shared])
    full, local = centers(base), centers(region)
    x, y = local - local.mean(0), full - full.mean(0)
    if not np.isfinite([x, y]).all() or any(not np.any(p != p[0]) for p in (full, local)):
        raise ValueError("Shared camera centers must be finite and span a nonzero distance in both exports")
    # Least-squares similarity with a proper rotation. The residual is relative
    # to the full trajectory's RMS radius, so it is independent of scene units.
    u, _, vt = np.linalg.svd(x.T @ y)
    u[:, -1] *= np.linalg.det(u @ vt)
    rotated = x @ (u @ vt)
    scale = float(np.sum(rotated * y) / np.sum(x * x))
    error = float(np.linalg.norm(scale * rotated - y) / np.linalg.norm(y))
    info = dict(shared_views=len(shared), regional_train_views=len(names), regional_sigma=sigma,
                fitted_scale=scale, relative_rms_residual=error)
    if error <= .05 and scale > 0:
        if np.allclose(full, local, atol=1e-4, rtol=0) and math.isclose(scale, 1., rel_tol=1e-5):
            scale = 1.  # Preserve the exact recorded cutoff for an original-pose export.
        method = "shared-camera baseline length" if len(shared) == 2 else "shared-camera similarity"
        return positive(sigma * scale), dict(info, method=method, applied_scale=scale)

    # With different trajectories there is no supported physical unit transfer.
    # Use the regional split fraction on the full-model footprint of the same photos.
    print(f"Camera trajectories disagree after similarity alignment (relative RMS {error:.1%}). "
          f"Checking the {len(shared)} shared photos before deriving a full-coordinate cutoff.", flush=True)
    for index, name in enumerate(shared, 1):
        a, b = (Path(spec["records"][name]["image_path"]) for spec in (base, region))
        if not same_photo(a, b):
            raise ValueError(f"Cannot verify the shared photo {name} across incompatible camera trajectories: "
                             f"{a} and {b}. Select the corresponding regional capture with --regional.")
        if index % 100 == 0:
            print(f"Verified shared photos {index}/{len(shared)}", flush=True)
    from utils.densification_diagnostics import training_extent
    footprint = positive(training_extent(base["records"], shared))
    fraction = sigma / extent
    return positive(fraction * footprint), dict(info, method="full-coordinate proposal from shared photos",
                                                 shared_full_extent=footprint, regional_split_fraction=fraction)


def same_photo(a, b):
    if not a.is_file() or not b.is_file():
        return False
    if a.samefile(b) or digest(a) == digest(b):
        return True
    # Metadata-only changes may leave different file bytes for identical training pixels.
    from PIL import Image
    with Image.open(a) as first, Image.open(b) as second:
        return first.size == second.size and np.array_equal(first.convert("RGB"), second.convert("RGB"))


def training_command(base, output, ratio=None, fixed_scale_ratio=None, *, steps=STEPS):
    """Recreate the actual native model and optimizer settings, including the 30k LR horizon."""
    from arguments import ModelParams, OptimizationParams, PipelineParams
    parser = argparse.ArgumentParser()
    ModelParams(parser)
    OptimizationParams(parser)
    PipelineParams(parser)
    defaults = vars(parser.parse_args([]))
    cfg = base["cfg"]
    opt = cfg["optimization_settings"]
    if (cfg.get("model_type", "3dgs") != "3dgs" or cfg.get("loss_mask_dir") or cfg.get("depths")
            or cfg.get("train_test_exp") or cfg.get("occlusion_mode", "off") != "off"
            or cfg.get("train_depth_occlusion_dir") or opt.get("pose_refinement")
            or (fixed_scale_ratio is None and opt.get("iterations") != STEPS[0]) or not cfg.get("keyframe_manifest")):
        raise ValueError("Expected a native, full-scene RGB keyframe baseline (30k for the adaptive comparison)")
    settings = {**cfg, **opt, "source_path": str(Path(cfg["source_path"]).resolve()),
                "model_path": str(output), "initial_point_cloud": str(base["path"] / "input.ply"),
                "keyframe_manifest": str(output / "keyframes.json"), "iterations": steps[-1]}
    if fixed_scale_ratio is None:
        settings["percent_dense"] = positive(ratio)
    else:
        if not (opt.get("freeze_lidar_positions") or opt.get("fixed_lidar")):
            raise ValueError("--fixed-model must identify a fixed-center LiDAR baseline")
        settings.update(fixed_lidar=True, freeze_lidar_positions=False, scale_control="clamp",
                        max_scale_ratio=positive(fixed_scale_ratio), max_scale_world=0.,
                        densify_until_iter=0, position_lr_init=0., position_lr_final=0.)
    command = [sys.executable, "-u", "train.py"]
    for key, default in defaults.items():
        value = settings.get(key, default)
        if value != default:
            command.append("--" + key)
            if not isinstance(default, bool):
                command.append(str(value))
            elif not value:
                raise ValueError(f"Cannot disable true-by-default option {key}")
    return command + ["--disable_viewer", "--publish_checkpoint_snapshots", "--test_iterations", "-1",
                      "--save_iterations", *map(str, steps), "--checkpoint_iterations", *map(str, steps),
                      "--debug-preview-every", "500", "--debug-preview-train", "2", "--debug-preview-test", "2",
                      "--debug-preview-seed", str(cfg.get("debug_preview_options", {}).get("seed", 0))]


def roi_indices(xyz, record, center, normal):
    from scripts.keyframe_pipeline.common import project
    x, y, w, h = ROI
    world_to_cam = np.asarray(record["R"]).T
    origin = -world_to_cam.T @ record["T"]
    intrinsics = np.array([record[k] for k in ("fx", "fy", "cx", "cy")])
    chunks = []
    for start in range(0, len(xyz), 262144):
        points = xyz[start:start + 262144]
        plane = np.flatnonzero(np.abs((points - center) @ normal) <= TOLERANCE)
        uv, depth = project(points[plane], world_to_cam, origin, intrinsics)
        u, v = (uv - .5).T
        visible = (depth > .05) & (u >= x) & (u < x+w) & (v >= y) & (v < y+h)
        chunks.append(plane[visible] + start)
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=int)


def quantiles(values):
    return dict(zip(("p10", "median", "p90", "p99"), map(float, np.percentile(values, (10, 50, 90, 99)))))


def estimate_fixed_scale(cloud, ratio):
    from plyfile import PlyData
    from scripts.audit_lidar_scale import initial_sigma
    ratio = positive(ratio)
    v = PlyData.read(str(cloud), mmap="r")["vertex"].data
    if not set("xyz") <= set(v.dtype.names) or {"scale_0", "f_dc_0"} & set(v.dtype.names):
        raise ValueError(f"Expected an original XYZ point cloud in {cloud}")
    xyz = np.column_stack([v[k] for k in "xyz"]).astype(np.float32, copy=False)
    if len(xyz) <= 100000:
        query = xyz
    else:
        query = xyz[np.random.default_rng(0).choice(len(xyz), 100000, replace=False)]
    print(f"Measuring {len(query):,} anchors against all {len(xyz):,} input points ...", flush=True)
    sigma = initial_sigma(xyz, query)
    return dict(point_cloud=str(cloud), input_points=len(xyz), measured_anchors=len(query),
                units="input PLY scene units, per-axis standard deviation",
                initial_3nn_sigma=quantiles(sigma), proposed_ceiling_over_initial=ratio,
                proposed_per_anchor_axis_ceiling=quantiles(sigma * ratio))


def floor_reference(spec):
    names = [name for name in spec["records"] if Path(name).name == FRAME]
    if FRAME in names:
        names = [FRAME]
    if len(names) != 1:
        return None, f"{FRAME} has {len(names)} matching registered images"
    name = names[0]
    record = spec["records"][name]
    if ROI[0]+ROI[2] > record["width"] or ROI[1]+ROI[3] > record["height"]:
        return None, f"saved floor crop is outside {name} at its native resolution"
    original = FLOOR_SOURCE/"images"/FRAME
    try:
        if original.is_file() and not same_photo(original, Path(record["image_path"])):
            return None, f"{name} differs from the source photograph used to select the floor crop"
    except OSError as exc:
        return None, f"could not verify the saved crop photograph ({exc})"
    split = "training" if name in spec["splits"]["train"] else "held-out"
    return name, split


def fixed_floor_plane(source, cloud, record):
    if Path(source).resolve() == FLOOR_SOURCE.resolve():
        return CENTER, NORMAL, "saved diagnostic"
    from scripts.diagnose_floor_blur import fit_plane, floor_points, plane_grid
    from scripts.keyframe_pipeline.common import Camera, Image, project
    rotation = np.asarray(record["R"])
    view = Image(0, 0, FRAME, rotation.T, -rotation @ record["T"])
    intrinsics = np.array([record[k] for k in ("fx", "fy", "cx", "cy")])
    camera = Camera(0, record["width"], record["height"], intrinsics)
    points = floor_points(cloud, view, camera, ROI, (.05, float("inf")))
    center, normal, inliers = fit_plane(points, TOLERANCE)
    plane_grid(view, camera, ROI, center, normal)  # Check that the plane is in front across the crop.
    uv, _ = project(inliers, view.R, view.C, intrinsics)
    cells = np.floor((uv - .5 - ROI[:2]) / ROI[2:] * 4).astype(int).clip(0, 3)
    coverage = len(np.unique(cells, axis=0))
    if coverage < 8:
        raise ValueError(f"Fixed-model plane inliers cover only {coverage}/16 floor crop cells")
    return center, normal, "dominant plane fitted in the fixed model's floor crop"


def spacing_report(model):
    from plyfile import PlyData
    from scipy.spatial import cKDTree
    spec = model_info(model)
    opt = spec["cfg"].get("optimization_settings", {})
    if spec["type"] != "3dgs" or not (opt.get("freeze_lidar_positions") or opt.get("fixed_lidar")):
        raise ValueError("Floor spacing requires a native fixed-LiDAR checkpoint")
    name, split = floor_reference(spec)
    if name is None:
        raise ValueError(split)
    record = spec["records"][name]
    iteration, ply = choose_iteration(spec["path"], -1)
    def read_xyz(path):
        v = PlyData.read(str(path), mmap="r")["vertex"].data
        return v, np.column_stack([v[k] for k in "xyz"]).astype(np.float32, copy=False)
    print(f"Measuring floor returns and surviving Gaussians at {iteration:,} ...", flush=True)
    center, normal, plane_source = fixed_floor_plane(spec["cfg"]["source_path"], model / "input.ply", record)
    source_vertices, xyz = read_xyz(model / "input.ply")
    del source_vertices
    ids = roi_indices(xyz, record, center, normal)
    if len(ids) < 4:
        raise ValueError("Insufficient original floor anchors in the saved plane and crop")
    tree = cKDTree(xyz)
    def neighbors(tree, points):
        distances = tree.query(points, k=4, workers=4)[0]
        if np.any(distances[:, 0] != 0):
            raise ValueError("A floor Gaussian center differs from its input LiDAR point")
        return distances[:, 1:]
    distances = neighbors(tree, xyz[ids])
    v, learned = read_xyz(ply)
    live = roi_indices(learned, record, center, normal)
    result = dict(iteration=iteration, model=str(model), source=spec["cfg"]["source_path"],
                  frame=name, split=split, roi_xywh=ROI, plane_center=center.tolist(), plane_normal=normal.tolist(),
                  plane_tolerance=TOLERANCE, plane_source=plane_source, units="fixed model scene units",
                  selection="Centers within floor plane tolerance projecting into the crop; occlusion is not tested",
                  input_points_in_crop=len(ids), surviving_centers_in_crop=len(live),
                  input_nearest_neighbor=quantiles(distances[:, 0]), input_third_neighbor=quantiles(distances[:, 2]))
    if len(live):
        distances = neighbors(tree, learned[live])
        initial = np.sqrt(np.maximum((distances**2).mean(1), 1e-7))
        largest = np.exp(np.maximum.reduce([v[f"scale_{axis}"][live] for axis in range(3)]))
        if not np.isfinite(largest).all():
            raise ValueError("Nonfinite learned scales")
        result.update(initial_sigma_at_survivors=quantiles(initial), learned_max_sigma=quantiles(largest),
                      max_sigma_over_initial=quantiles(largest / initial))
    del tree
    if len(learned) >= 4 and len(live):
        distances = neighbors(cKDTree(learned), learned[live])
        result["surviving_nearest_neighbor"] = quantiles(distances[:, 0])
    return result


def measured_frames(model, step, source, base):
    from scripts.run_soft_surface_long import verify_metrics
    metric = model / f"psnr_all_frames/iteration_{step}"
    count = len(base["records"])
    aa = base["cfg"].get("antialiasing", True)
    meta = dict(model=str(model), source=source, frames=sorted(base["records"]),
                selected=sorted(base["splits"]["train"]), plan=dict(antialiasing=aa))
    def verified():
        s = read_json(metric / "summary.json")
        verify_metrics(meta, step, s, metric / "per_frame.csv")
        if (s.get("loss_mask_dir") or s.get("camera_poses", "original") != "original"
                or s.get("resolution_setting") != base["cfg"]["resolution"]):
            raise ValueError("Different evaluation protocol")
        return s["mean_psnr_db"]
    try:
        return verified()
    except (FileNotFoundError, ValueError, KeyError):
        run("scripts/eval_all_frames.py", "--source", source, "--model", model, "--iteration", step,
            "--expected-frames", count, "--antialiasing" if aa else "--no-antialiasing", "--occlusion", "off")
        return verified()


def run(*args):
    subprocess.run([sys.executable, "-u", *map(str, args)], cwd=ROOT, check=True)


def floor_render(model, step, output, base, expected_gt, name):
    import cv2
    from scripts.diagnose_floor_blur import salience
    from scripts.render_views_worker import render_views
    spec = model_info(model)
    if (spec["records"][name] != base["records"][name] or spec["splits"]["train"] != base["splits"]["train"]
            or spec["cfg"].get("antialiasing") != base["cfg"].get("antialiasing")):
        raise ValueError("Camera, AA or training split differs")
    # Native-pixel floor crop even if the training image was downsampled.
    spec["cfg"] = {**spec["cfg"], "resolution": 1}
    render_views(spec, [name], step, output)
    r = read_json(output / "frames.json")[name]
    if expected_gt and r["target_sha256"] != expected_gt:
        raise ValueError("Ground truth RGB differs between renders")
    x,y,w,h = ROI
    crops = []
    for key in ("gt", "image"):
        pixels = cv2.imread(str(output / r[key]))
        crop = pixels[y:y+h, x:x+w]
        if crop.shape != (h,w,3) or not cv2.imwrite(str(output / (key+"_crop.png")), crop):
            raise ValueError("Failed to extract the native-pixel crop")
        crops.append(crop)
    a,b = crops
    mse = np.mean(((a.astype(float)-b)/255.)**2)
    mask = np.zeros((h,w), bool)
    mask[2:-2,2:-2] = True
    edges = salience(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), mask)
    return r["target_sha256"], dict(crop_psnr_db=-10*math.log10(mse) if mse else "inf", **edges)


def evaluate(base, output, gpu, transfer):
    report = output / "comparison"
    report.mkdir(exist_ok=True)
    import torch
    rows, gt = [], None
    name, floor_status = floor_reference(base)
    control = transfer.get("label") or ("Matched" if "applied_scale" in transfer else "Proposed")
    comparisons = [(f"{control} 30k", output, 30000), (f"{control} 50k", output, 50000)]
    has_baseline = (base["path"]/"point_cloud/iteration_30000/point_cloud.ply").is_file()
    if has_baseline:
        comparisons.insert(0, ("Baseline 30k", base["path"], 30000))
    for label, model, step in comparisons:
        averages = measured_frames(model, step, str(Path(base["cfg"]["source_path"]).resolve()), base)
        folder = report / label.replace(" ", "_")
        metrics = {}
        if name is not None:
            gt, metrics = floor_render(model, step, folder, base, gt, name)
            metrics["crop_image"] = f"{folder.name}/image_crop.png"
        torch.cuda.empty_cache()  # The next full-frame evaluation runs in a separate process.
        rows.append(dict(label=label, model=str(model), iteration=step, mean_psnr_db=averages, **metrics))
    models = [base["path"], output] if has_baseline else [output]
    has_holdout = bool(base["records"].keys() - base["splits"]["train"])
    split = "both" if has_holdout else "train"
    run("scripts/render.py", "--models", *models, "--iteration", 30000, "--gpu", gpu,
        "--split", split, "--eval-set", "unselected", "--max-views", 8, "--output", report/"paired_30k")
    run("scripts/render.py", "--model", output, "--iteration", 50000, "--gpu", gpu,
        "--split", split, "--eval-set", "unselected", "--max-views", 8, "--output", report/"matched_50k")
    policy_key = "fixed_scale_policy" if transfer.get("label") == "Fixed cap" else "regional_scale_transfer"
    atomic_json(report/"results.json", dict(frame=name, roi=ROI if name else None, floor_status=floor_status,
                                             held_out=(floor_status == "held-out") if name else None,
                                             identical_original_poses=True, rows=rows, **{policy_key: transfer}))
    lines = []
    group = "unselected" if has_holdout else "selected"
    group_name = "Held-out" if has_holdout else "Training"
    gallery_split = "eval" if has_holdout else "train"
    for r in rows:
        crop_cells = (f'<td>{float(r["crop_psnr_db"]):.3f}</td><td>{r["gradient_agreement"]}</td>' if name else '')
        lines.append(f'<tr><td>{r["label"]}</td><td>{r["mean_psnr_db"][group]:.3f}</td>{crop_cells}</tr>')
    first_folder = rows[0]["label"].replace(" ", "_")
    pictures = ([("Source photograph", f"{first_folder}/gt_crop.png")] +
                [(r["label"], r["crop_image"]) for r in rows]) if name else []
    crop_headers = '<th>Floor PSNR (dB)</th><th>Floor edge agreement</th>' if name else ''
    page = ('<!doctype html><meta charset="utf-8"><title>Orange4F detail comparison</title>'
            '<style>body{font:16px system-ui;max-width:1280px;margin:2em auto}td,th{padding:.5em}img{width:100%}</style>'
            f'<h1>Orange4F detail comparison</h1><p>Floor crop: {html.escape(name or FRAME)} ({html.escape(floor_status)}).</p>'
            f'<p>{html.escape(transfer["method"])}. '
            '<a href="../detail_plan.json">Training settings</a></p>'
            f'<table><tr><th>Run</th><th>{group_name} PSNR (dB)</th>{crop_headers}</tr>'+''.join(lines)+'</table>'+''.join(
                f'<h2>{html.escape(label)}</h2><a href="{img}"><img src="{img}"></a>' for label,img in pictures)
            +f'<p><a href="paired_30k/{gallery_split}/index.html">{group_name} views at 30k</a> · '
             f'<a href="matched_50k/{gallery_split}/index.html">{group_name} views at 50k</a> · <a href="results.json">Metrics</a></p>')
    (report/"index.html").write_text(page)
    for row in rows:
        floor = f", {floor_status} floor crop {float(row['crop_psnr_db']):.3f} dB" if name else ""
        print(f"{row['label']}: {group_name} {row['mean_psnr_db'][group]:.3f} dB{floor}", flush=True)
    print(report / "index.html", flush=True)


@contextmanager
def experiment_lock(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / ("."+output.name+".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield lock


def select_gpu(requested):
    if requested is not None:
        return requested
    allowed = os.environ.get("CUDA_VISIBLE_DEVICES")
    if allowed is not None:
        if not allowed or allowed == "-1":
            raise RuntimeError("CUDA devices are hidden; run with --gpu N on the training host")
        return allowed.split(",")[0]
    def query(option):
        return subprocess.check_output(["nvidia-smi", option, "--format=csv,noheader,nounits"], text=True).splitlines()
    used = set(query("--query-compute-apps=gpu_uuid"))
    available = []
    for row in query("--query-gpu=index,uuid,memory.free"):
        index, uuid, free = map(str.strip, row.split(","))
        if uuid not in used:
            available.append((int(free), index))
    if not available:
        raise RuntimeError("All GPUs have active compute processes. Run when one is available, or choose --gpu N.")
    return max(available)[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=SUITE/"3dgs")
    parser.add_argument("--fixed-model", type=Path, default=SUITE/"3dgs_fixed_lidar")
    parser.add_argument("--regional", type=Path, help="Regional adaptive run or exported region containing selection.json")
    parser.add_argument("--output", type=Path, help="Separate model directory (default chosen by experiment)")
    parser.add_argument("--gpu", help="Physical CUDA device. Default honors CUDA_VISIBLE_DEVICES or selects a free GPU")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--spacing-only", action="store_true", help="Audit the fixed checkpoint's saved floor crop")
    mode.add_argument("--estimate-fixed-scale", action="store_true", help="Estimate per-anchor sigma from input.ply on CPU")
    mode.add_argument("--train-fixed-scale", action="store_true", help="Train fixed centers with per-anchor KNN scale caps")
    mode.add_argument("--probe-image", help="Fit one original training image for 2,000 steps with the fixed-center recipe")
    parser.add_argument("--fixed-scale-ratio", type=float, default=2., help="Per-axis sigma ceiling / native 3-NN sigma")
    parser.add_argument("--plan-only", action="store_true", help="Print measured extents and training command without training")
    args = parser.parse_args(argv)
    if args.estimate_fixed_scale:
        print(json.dumps(estimate_fixed_scale(args.fixed_model.expanduser().resolve()/"input.ply",
                                              args.fixed_scale_ratio), indent=2))
        return
    if args.spacing_only:
        print(json.dumps(spacing_report(args.fixed_model.resolve()), indent=2))
        return
    probe = args.probe_image
    fixed = args.train_fixed_scale or probe is not None
    steps = PROBE_STEPS if probe is not None else STEPS
    base = model_info(args.fixed_model if fixed else args.baseline)
    if probe is not None:
        if probe not in base["splits"]["train"]:
            parser.error(f"--probe-image must exactly match a saved training image: {probe!r}")
        source_image = Path(base["records"][probe]["image_path"])
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        default_model = f"fixed_lidar_one_view_{Path(probe).stem}_knn{args.fixed_scale_ratio:g}x"
    elif fixed:
        default_model = f"fixed_lidar_knn{args.fixed_scale_ratio:g}x_50k_v1"
    else:
        default_model = "matched_split_50k_v1"
    output = (args.output or SUITE.parent/default_model).expanduser().resolve()
    if not (base["path"]/"input.ply").is_file():
        raise FileNotFoundError(base["path"]/"input.ply")
    if probe is None:
        name, floor_status = floor_reference(base)
        print(f"Floor crop {name or FRAME}: {floor_status}.", flush=True)
    if fixed:
        ratio = positive(args.fixed_scale_ratio)
        transfer = dict(method=f"Fixed LiDAR, each principal sigma capped at {ratio:g} times its initial 3-NN sigma",
                        label="Fixed cap", max_scale_ratio=ratio, constant_population=True, opacity_resets=False)
        print(f"{transfer['method']}. All input anchors are retained.", flush=True)
        command = training_command(base, output, fixed_scale_ratio=ratio, steps=steps)
        plan = dict(fixed_scale_policy=transfer)
        if probe is not None:
            print(f"One-view fit of {probe} for {steps[-1]:,} steps using the original image and pose.", flush=True)
            plan["probe_keyframes"] = {"selected_images": [probe]}
    else:
        choose_iteration(base["path"], STEPS[0])
        region = regional_reference(regional_model(args.regional))
        extent, old_sigma, full_source = split_settings(base)
        region_extent, regional_sigma, target_source = split_settings(region)
        target, transfer = regional_cutoff(base, region, region_extent, regional_sigma)
        if math.isclose(old_sigma, target, rel_tol=1e-6):
            print("The physical thresholds already coincide; this run tests the iteration extension only.", flush=True)
        ratio = target / extent
        command = training_command(base, output, ratio)
        plan = dict(regional=str(region["path"]), full_extent=extent, regional_extent=region_extent,
                    extent_source=full_source, target_source=target_source, regional_scale_transfer=transfer,
                    baseline_split_sigma=old_sigma, matched_split_sigma=target, percent_dense=ratio)
        print(f"Full extent {extent:.6g}, regional extent {region_extent:.6g}, "
              f"split sigma {old_sigma:.6g} → {target:.6g}, percent_dense={ratio:.8g}.", flush=True)
        print(f"Regional reference: {region['path']} ({target_source})", flush=True)
        print(f"{transfer['method']}: {transfer['shared_views']}/{transfer['regional_train_views']} views, "
              f"relative RMS {transfer['relative_rms_residual']:.2%}, regional sigma {regional_sigma:.6g}, "
              f"full-coordinate target {target:.6g}.", flush=True)
    plan.update(baseline=str(base["path"]), selected_keyframes_sha256=digest(base["path"]/"keyframes.json"),
                input_ply_sha256=digest(base["path"]/"input.ply"), command=command)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    with experiment_lock(output) as lock:
        protocol = output / "detail_plan.json"
        if output.exists():
            previous_plan = read_json(protocol) if protocol.is_file() else {}
            # Earlier successful plans had no transfer metadata and used the raw cutoff.
            if "regional_scale_transfer" not in previous_plan and transfer.get("applied_scale") == 1.:
                previous_plan["regional_scale_transfer"] = transfer
            if previous_plan != plan:
                raise FileExistsError(f"Unrelated output or different protocol: {output}")
        else:
            output.mkdir()
            atomic_json(protocol, plan)
        manifest = output / "keyframes.json"
        if not manifest.exists():
            if probe is None:
                shutil.copyfile(base["path"]/"keyframes.json", manifest)
            else:
                atomic_json(manifest, plan["probe_keyframes"])
        elif (digest(manifest) != plan["selected_keyframes_sha256"] if probe is None
              else read_json(manifest) != plan["probe_keyframes"]):
            raise ValueError(f"Training manifest changed: {manifest}")
        if probe is not None:
            suite = base["path"].parent
            if output.parent == suite.parent and ((suite/"suite.json").is_file() or (suite/"index.html").is_file()):
                from scripts.refresh_orange4f_dashboard import refresh
                try:
                    if output.name in refresh(suite):
                        print(f"One-view previews on the suite dashboard: {suite/'one_view'/output.name/'index.html'}",
                              flush=True)
                except (OSError, ValueError) as exc:
                    print(f"Could not update the preview dashboard: {exc}", flush=True)
        gpu = select_gpu(args.gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        print(f"Using CUDA device {gpu}.", flush=True)
        snapshots = [step for step in steps if read_snapshot(output, step)]
        if steps[-1] not in snapshots:
            if snapshots:
                command = command + ["--start_checkpoint", str(output/f"chkpnt{max(snapshots)}.pth")]
            print("Training with live GT/render/error previews", flush=True)
            subprocess.run(command, cwd=ROOT, check=True, pass_fds=(lock.fileno(),))
        if probe is not None:
            run("scripts/render.py", "--model", output, "--iteration", steps[-1], "--gpu", gpu,
                "--split", "both", "--eval-set", "unselected", "--max-views", 2, "--output", output/"comparison")
            print(output/"debug_previews/index.html", flush=True)
            return
        floor_out = output/"fixed_floor_spacing.json"
        try:
            fixed = args.fixed_model.expanduser().resolve()
            iteration, _ = choose_iteration(fixed, -1)
            previous = read_json(floor_out) if floor_out.is_file() else {}
            if ((previous.get("model"), previous.get("iteration")) != (str(fixed), iteration)
                    or not previous.get("plane_source")):
                previous = spacing_report(fixed)
                atomic_json(floor_out, previous)
            print(json.dumps(previous, indent=2), flush=True)
        except (OSError, ValueError, KeyError) as exc:
            print(f"Fixed-floor measurement unavailable: {exc}. Continuing comparison.", flush=True)
        evaluate(base, output, gpu, transfer)


if __name__ == "__main__":
    main()
