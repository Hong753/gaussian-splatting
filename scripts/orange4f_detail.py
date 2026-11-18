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


def training_command(base, output, ratio):
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
            or opt.get("iterations") != STEPS[0] or not cfg.get("keyframe_manifest")):
        raise ValueError("--baseline must be the full-scene, native 30k RGB keyframe control")
    settings = {**cfg, **opt, "source_path": str(Path(cfg["source_path"]).resolve()),
                "model_path": str(output), "initial_point_cloud": str(base["path"] / "input.ply"),
                "keyframe_manifest": str(output / "keyframes.json"), "iterations": STEPS[-1],
                "percent_dense": ratio}
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
                      "--save_iterations", *map(str, STEPS), "--checkpoint_iterations", *map(str, STEPS),
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
    record = spec["records"][FRAME]
    if record["width"] < ROI[0]+ROI[2] or record["height"] < ROI[1]+ROI[3]:
        raise ValueError("The saved floor crop is outside the reference image")
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
                  frame=FRAME, roi_xywh=ROI, plane_center=center.tolist(), plane_normal=normal.tolist(),
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


def floor_render(model, step, output, base, expected_gt):
    import cv2
    from scripts.diagnose_floor_blur import salience
    from scripts.render_views_worker import render_views
    spec = model_info(model)
    if (spec["records"][FRAME] != base["records"][FRAME] or spec["splits"]["train"] != base["splits"]["train"]
            or spec["cfg"].get("antialiasing") != base["cfg"].get("antialiasing")):
        raise ValueError("Camera, AA or training split differs")
    # Native-pixel floor crop even if the training image was downsampled.
    spec["cfg"] = {**spec["cfg"], "resolution": 1}
    render_views(spec, [FRAME], step, output)
    r = read_json(output / "frames.json")[FRAME]
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
    control = "Matched" if "applied_scale" in transfer else "Proposed"
    for label, model, step in (("Baseline 30k",base["path"],30000), (f"{control} 30k",output,30000),
                               (f"{control} 50k",output,50000)):
        averages = measured_frames(model, step, str(Path(base["cfg"]["source_path"]).resolve()), base)
        folder = report / label.replace(" ", "_")
        gt, metrics = floor_render(model, step, folder, base, gt)
        torch.cuda.empty_cache()  # The next full-frame evaluation runs in a separate process.
        rows.append(dict(label=label, model=str(model), iteration=step, mean_psnr_db=averages,
                         crop_image=f"{folder.name}/image_crop.png", **metrics))
    run("scripts/render.py", "--models", base["path"], output, "--iteration", 30000, "--gpu", gpu,
        "--split", "both", "--eval-set", "unselected", "--max-views", 8, "--output", report/"paired_30k")
    run("scripts/render.py", "--model", output, "--iteration", 50000, "--gpu", gpu,
        "--split", "both", "--eval-set", "unselected", "--max-views", 8, "--output", report/"matched_50k")
    atomic_json(report/"results.json", dict(frame=FRAME, roi=ROI, held_out=FRAME not in base["splits"]["train"],
                                             identical_original_poses=True, regional_scale_transfer=transfer, rows=rows))
    lines = []
    for r in rows:
        lines.append(f'<tr><td>{r["label"]}</td><td>{r["mean_psnr_db"]["unselected"]:.3f}</td>'
                     f'<td>{float(r["crop_psnr_db"]):.3f}</td><td>{r["gradient_agreement"]}</td></tr>')
    pictures = [("Source photograph", "Baseline_30k/gt_crop.png")] + [(r["label"], r["crop_image"]) for r in rows]
    page = ('<!doctype html><meta charset="utf-8"><title>Orange4F detail comparison</title>'
            '<style>body{font:16px system-ui;max-width:1280px;margin:2em auto}td,th{padding:.5em}img{width:100%}</style>'
            f'<h1>{FRAME} floor comparison</h1><p>Split cutoff from {html.escape(transfer["method"])}. '
            '<a href="../detail_plan.json">Training settings</a></p>'
            '<table><tr><th>Run</th><th>Held-out PSNR (dB)</th><th>Floor PSNR (dB)</th>'
            '<th>Floor edge agreement</th></tr>'+''.join(lines)+'</table>'+''.join(
                f'<h2>{html.escape(label)}</h2><a href="{img}"><img src="{img}"></a>' for label,img in pictures)
            +'<p><a href="paired_30k/eval/index.html">Shared held-out views at 30k</a> · '
             '<a href="matched_50k/eval/index.html">Held-out views at 50k</a> · <a href="results.json">Metrics</a></p>')
    (report/"index.html").write_text(page)
    for row in rows:
        print(f"{row['label']}: held-out {row['mean_psnr_db']['unselected']:.3f} dB; "
              f"floor {float(row['crop_psnr_db']):.3f} dB", flush=True)
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
    parser.add_argument("--output", type=Path, default=SUITE.parent/"matched_split_50k_v1")
    parser.add_argument("--gpu", help="Physical CUDA device. Default honors CUDA_VISIBLE_DEVICES or selects a free GPU")
    parser.add_argument("--spacing-only", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="Print measured extents and training command without training")
    args = parser.parse_args(argv)
    if args.spacing_only:
        print(json.dumps(spacing_report(args.fixed_model.resolve()), indent=2))
        return
    output = args.output.expanduser().resolve()
    base = model_info(args.baseline)
    region = regional_reference(regional_model(args.regional))
    if not (base["path"]/"input.ply").is_file():
        raise FileNotFoundError(base["path"]/"input.ply")
    if FRAME not in base["records"] or FRAME in base["splits"]["train"]:
        raise ValueError("Expected the original split with the saved floor reference held out")
    choose_iteration(base["path"], STEPS[0])
    extent, old_sigma, full_source = split_settings(base)
    region_extent, regional_sigma, target_source = split_settings(region)
    target, transfer = regional_cutoff(base, region, region_extent, regional_sigma)
    if math.isclose(old_sigma, target, rel_tol=1e-6):
        print("The physical thresholds already coincide; this run tests the iteration extension only.", flush=True)
    ratio = target / extent
    command = training_command(base, output, ratio)
    plan = dict(baseline=str(base["path"]), regional=str(region["path"]), full_extent=extent,
                regional_extent=region_extent, extent_source=full_source, target_source=target_source,
                regional_scale_transfer=transfer,
                baseline_split_sigma=old_sigma, matched_split_sigma=target, percent_dense=ratio,
                selected_keyframes_sha256=digest(base["path"]/"keyframes.json"),
                input_ply_sha256=digest(base["path"]/"input.ply"), command=command)
    print(f"Full extent {extent:.6g}, regional extent {region_extent:.6g}, "
          f"split sigma {old_sigma:.6g} → {target:.6g}, percent_dense={ratio:.8g}.", flush=True)
    print(f"Regional reference: {region['path']} ({target_source})", flush=True)
    print(f"{transfer['method']}: {transfer['shared_views']}/{transfer['regional_train_views']} views, "
          f"relative RMS {transfer['relative_rms_residual']:.2%}, regional sigma {regional_sigma:.6g}, "
          f"full-coordinate target {target:.6g}.", flush=True)
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
            shutil.copyfile(base["path"]/"keyframes.json", manifest)
        elif digest(manifest) != plan["selected_keyframes_sha256"]:
            raise ValueError(f"Training manifest changed: {manifest}")
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
            print(f"Fixed-floor measurement unavailable: {exc}. Continuing adaptive comparison.", flush=True)
        gpu = select_gpu(args.gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        print(f"Using CUDA device {gpu}.", flush=True)
        snapshots = [step for step in STEPS if read_snapshot(output,step)]
        if 50000 not in snapshots:
            if snapshots:
                command = command + ["--start_checkpoint", str(output/f"chkpnt{max(snapshots)}.pth")]
            print("Training with live GT/render/error previews", flush=True)
            subprocess.run(command, cwd=ROOT, check=True, pass_fds=(lock.fileno(),))
        evaluate(base, output, gpu, transfer)


if __name__ == "__main__":
    main()
