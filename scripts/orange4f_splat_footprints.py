"""Read-only native 3DGS contribution audit of the same Orange4F floor patch in two views."""

import argparse
from html import escape
import json
from pathlib import Path
import os
import sys
import tempfile
from types import SimpleNamespace
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_floor_blur import floor_points, load_rgb, plane_grid, target_grid
from scripts.keyframe_pipeline.common import atomic_json, image_path, read_model
from scripts.orange4f_floor_neighbors import (
    REFERENCE,
    ROI_FRACTIONS,
    TOLERANCE,
    fit_visible_patch,
    local_cloud,
    visible_floor,
)
from scripts.orange4f_detail import SUITE, select_gpu
from scripts.view_comparisons import model_info, shared_iteration
from utils.checkpoint_snapshots import atomic_write
from utils.densification_diagnostics import projected_covariance

NEIGHBOR = "09884.jpg"
REPORT = "floor_splat_audit_v1"


def bright_cores(rgb, visible):
    """Apparent bright-core diameter, in native pixels, used as a local comparison scale."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    contrast = gray - cv2.GaussianBlur(gray, (0, 0), 4.0)
    cut = max(5.0, float(np.percentile(contrast[visible], 90)))
    mask = contrast >= cut
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    # Only use intact components away from crop and visibility boundaries.
    inside = cv2.erode(
        visible.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0
    ).astype(bool)
    area = stats[:, cv2.CC_STAT_AREA]
    good = (area >= 2) & (area <= 150)
    good[0] = False
    good[np.unique(labels[~inside])] = False
    widths = 2 * np.sqrt(area[good] / np.pi)
    diameter = float(np.median(widths)) if len(widths) >= 12 else None
    return good[labels], dict(count=int(good.sum()), median_diameter_px=diameter, contrast_threshold=cut)


def core_contrast_ratio(rgb, pred, cores):
    """Mean signed local contrast at the same GT bright pixels, in render / photo units."""
    if not cores.any():
        return None

    def contrast(image):
        gray = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)
        return float((gray - cv2.GaussianBlur(gray, (0, 0), 4.0))[cores].mean())

    measured = contrast(rgb)
    return contrast(pred) / measured if measured > 0 else None


def patches(suite, output):
    meta = json.loads((suite / "suite.json").read_text())
    source, cloud = Path(meta["source"]), Path(meta["cloud"]["cached"])
    folder = json.loads(Path(meta["selection"]).read_text()).get("image_directory", "images")
    _, cameras, images, _ = read_model(source)
    views = {v.name: v for v in images if v.name in (REFERENCE, NEIGHBOR)}
    missing = {REFERENCE, NEIGHBOR} - views.keys()
    if missing:
        raise ValueError(f"Original camera records are missing for {sorted(missing)} in {source}")
    ref = views[REFERENCE]
    camera = cameras[ref.camera_id]
    x, y, r, b = [round(a * n) for a, n in zip(ROI_FRACTIONS, (camera.width, camera.height) * 2)]
    roi = (x, y, r - x, b - y)
    ref_photo = load_rgb(image_path(source, folder, REFERENCE), camera)
    points = floor_points(cloud, ref, camera, roi, (0.05, np.inf))
    fit, plane, reason = fit_visible_patch(points, ref, camera, roi, ref_photo[y:b, x:r], output)
    if fit is None:
        raise ValueError(f"Cannot measure depths relative to the floor: {reason}")
    center, normal, world = fit
    cloud_points = local_cloud(cloud, world, [v.C for v in views.values()])
    corners = world[[0, 0, -1, -1], [0, -1, -1, 0]].reshape(2, 2, 3)
    data = {}
    for name in (REFERENCE, NEIGHBOR):
        view = views[name]
        cam = cameras[view.camera_id]
        if name == REFERENCE:
            rect, grid, photo = roi, world, ref_photo
        else:
            xy, _ = target_grid(corners, view, cam)
            bounds = xy.reshape(-1, 2)
            lo = np.maximum(0, np.floor(bounds.min(axis=0)).astype(int))
            hi = np.minimum([cam.width, cam.height], np.ceil(bounds.max(axis=0)).astype(int) + 1)
            if (hi <= lo).any():
                raise ValueError(f"Floor patch projects outside {name}")
            rect = tuple(map(int, [*lo, *(hi - lo)]))
            grid = plane_grid(view, cam, rect, center, normal)
            photo = load_rgb(image_path(source, folder, name), cam)
        mask = np.ones(grid.shape[:2], bool)
        for other in views.values():
            k = cameras[other.camera_id]
            xy, valid = target_grid(grid, other, k)
            if other.name == REFERENCE:
                # Allow roundoff at the boundaries of the same calibrated crop.
                valid &= ((xy >= [x - 1e-5, y - 1e-5]) & (xy <= [r - 1 + 1e-5, b - 1 + 1e-5])).all(axis=2)
            mask &= visible_floor(cloud_points, center, normal, other, k, xy, valid)
        if not mask.any():
            raise ValueError(f"No mutually LiDAR-supported floor pixels in {name}")
        left, top, w, h = rect
        rgb = photo[top : top + h, left : left + w, ::-1].copy()
        cores, detail = bright_cores(rgb, mask)
        prefix = Path(name).stem
        Image.fromarray(rgb).save(output / f"{prefix}_gt.png")
        # Native pixel positions. Green marks the detected bright cores and gray the excluded pixels.
        marked = rgb.copy()
        marked[~mask] //= 4
        marked[cores] = [40, 240, 80]
        Image.fromarray(marked).save(output / f"{prefix}_cores.png")
        data[name] = dict(
            rect=rect,
            rgb=rgb,
            mask=mask,
            cores=cores,
            detail=detail,
            floor_z=float(np.median((grid[mask] - view.C) @ view.R[2])),
            path=image_path(source, folder, name).resolve(),
            view=view,
            cam=cam,
        )
    return data, plane


def contribution_weights(render_tags, xyz, rect, mask, cores):
    """d(sum of tagged pixel colors)/d(tag_i) = sum_p T_i(p)*alpha_i(p)."""
    import torch

    x, y, w, h = rect
    with torch.enable_grad():
        tags = torch.full_like(xyz, 0.5, requires_grad=True)
        crop = render_tags(tags)[:2, y : y + h, x : x + w]
        masks = torch.as_tensor(np.stack([mask, cores]), device=crop.device, dtype=crop.dtype)
        sums = (crop * masks).sum(dim=(1, 2))
        (grad,) = torch.autograd.grad(sums.sum(), tags)
    weights = grad[:, :2].detach()
    if not torch.isfinite(weights).all() or torch.any(weights < 0):
        raise ValueError("Nonfinite or negative compositing weights")
    total = weights.sum(dim=0)
    # The .5-color forward pass against black independently measures accumulated opacity.
    if not torch.allclose(total, 2 * sums.detach(), rtol=0.003, atol=0.002):
        raise ValueError(
            f"Native backward/forward compositing mass mismatch: {total.tolist()} vs {(2 * sums).tolist()}"
        )
    ids = (weights[:, 0] > 0).nonzero().flatten()
    return ids, weights[ids].cpu().numpy()


def distribution(values, weights):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    idx = np.searchsorted(cumulative, cumulative[-1] * np.array([0.1, 0.5, 0.9]))
    return dict(zip(("p10", "p50", "p90"), map(float, values[order[idx]])))


def summarize(weights, minor, major, depth, offset, diameter, pixels):
    mass = weights.sum(dtype=np.float64)
    if mass <= 0 or not pixels:
        return dict(mean_alpha=0.0, verdict="Insufficient compositing mass.")

    def share(condition):
        return float(weights[condition].sum(dtype=np.float64) / mass)

    off = np.abs(offset) > TOLERANCE
    row = dict(
        mean_alpha=float(mass / pixels),
        contributing_centers=int(np.count_nonzero(weights)),
        minor_fwhm_px=distribution(minor, weights),
        major_fwhm_px=distribution(major, weights),
        camera_z=distribution(depth, weights),
        signed_plane_offset=distribution(offset, weights),
        off_plane_share=share(off),
        front_of_plane_share=share(offset > TOLERANCE),
        behind_plane_share=share(offset < -TOLERANCE),
    )
    if diameter is not None:
        row["broad_share"] = share(minor > diameter)
        row["broad_on_plane_share"] = share((minor > diameter) & ~off)
    if row["mean_alpha"] < 0.5:
        verdict = "Low accumulated opacity. Background contributes substantially."
    elif row["off_plane_share"] > 0.5:
        verdict = "Most weight comes from centers outside the fitted floor band. Inspect depth placement."
    elif diameter is None:
        verdict = "Too few distinct bright cores to compare footprint widths."
    elif row["broad_on_plane_share"] > 0.5:
        verdict = "Most weight comes from on-plane centers with footprints wider than a bright core on both axes."
    elif row["major_fwhm_px"]["p50"] <= diameter:
        verdict = (
            "Typical footprints fit within a bright core. Check view sampling and conflicting observations next."
        )
    else:
        verdict = "Footprints or depths are mixed. The measurements do not isolate a single cause."
    return dict(row, verdict=verdict)


def run_model(spec, step, data, plane, output):
    import torch
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from scripts.render_views_worker import load_record_camera
    from scripts.eval_all_frames import choose_iteration

    cfg = spec["cfg"]
    _, ply = choose_iteration(spec["path"], step)
    model = GaussianModel(cfg.get("sh_degree", 3))
    print(f"Loading {ply}", flush=True)
    model.load_ply(str(ply))
    for value in vars(model).values():
        if torch.is_tensor(value) and value.requires_grad:
            value.requires_grad_(False)
    bg = model.get_xyz.new_full((3,), float(cfg.get("white_background", False)))
    pipe = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=bool(cfg.get("antialiasing", False)),
        occlusion_mode="off",
    )
    args = SimpleNamespace(resolution=1, data_device="cpu", train_test_exp=False, loss_mask_dir="")
    rows = []
    for name, patch in data.items():
        record = spec["records"][name]
        view, cam = patch["view"], patch["cam"]
        if (
            Path(record["image_path"]).resolve() != patch["path"]
            or (record["width"], record["height"]) != (cam.width, cam.height)
            or not np.allclose([record[k] for k in ("fx", "fy", "cx", "cy")], cam.params, rtol=0, atol=1e-6)
            or not np.allclose(record["R"], view.R.T, rtol=0, atol=1e-6)
            or not np.allclose(record["T"], -view.R @ view.C, rtol=0, atol=1e-6)
        ):
            raise ValueError(f"Model camera or image differs from the original {name}")
        camera = load_record_camera(args, record, 0)
        rect, mask = patch["rect"], patch["mask"]
        x, y, w, h = rect

        def tags(colors):
            return render(camera, model, pipe, torch.zeros_like(bg), override_color=colors)["render"]

        print(f"Measuring {spec['path'].name} · {step:,} · {name}", flush=True)
        with torch.no_grad():
            pred = (
                render(camera, model, pipe, bg)["render"][:, y : y + h, x : x + w].permute(1, 2, 0).cpu().numpy()
            )
        ids, weights = contribution_weights(tags, model.get_xyz, rect, mask, patch["cores"])
        xyz = model.get_xyz[ids].detach().cpu().numpy()
        scale = model.get_scaling[ids].detach().cpu().numpy()
        rotation = model.get_rotation[ids].detach().cpu().numpy()
        view_matrix = camera.world_view_transform.cpu().numpy()
        widths = np.empty((len(xyz), 2))
        for start in range(0, len(xyz), 65536):
            block = slice(start, start + 65536)
            cov = projected_covariance(
                xyz[block],
                scale[block],
                rotation[block],
                view_matrix,
                cam.width,
                cam.height,
                camera.FoVx,
                camera.FoVy,
            )
            widths[block] = np.sqrt(np.maximum(np.linalg.eigvalsh(cov), 0)) * np.sqrt(8 * np.log(2))
        z = (xyz - view.C) @ view.R[2]
        center, normal = np.asarray(plane["center"]), np.asarray(plane["normal"])
        offset = (xyz - center) @ normal * np.sign((view.C - center) @ normal)
        if not all(np.isfinite(v).all() for v in (widths, z, offset, pred)):
            raise ValueError(f"Nonfinite footprint, depth or RGB for a contributor in {name}")
        diameter = patch["detail"]["median_diameter_px"]
        row = dict(
            model=spec["path"].name,
            checkpoint=str(ply),
            iteration=step,
            frame=name,
            split="train" if name in spec["splits"]["train"] else "held-out",
            original_pose=True,
            aa=pipe.antialiasing,
            floor_z=patch["floor_z"],
            centered_minus_colmap_px=[cam.width / 2 - cam.params[2], cam.height / 2 - cam.params[3]],
            crop_xywh=rect,
            supported_pixels=int(mask.sum()),
            texture=patch["detail"],
        )
        for i, (label, area) in enumerate([("floor", mask), ("bright_cores", patch["cores"])]):
            row[label] = summarize(weights[:, i], *widths.T, z, offset, diameter, int(area.sum()))
        target = patch["rgb"].astype(np.float32) / 255.0
        row["core_contrast_ratio"] = (
            core_contrast_ratio(target, pred, patch["cores"]) if diameter is not None else None
        )
        mse = float(np.mean((pred[mask] - target[mask]) ** 2))
        row["crop_psnr_db"] = -10 * np.log10(mse) if mse else "inf"
        stem = f"{spec['path'].name}_{step}_{Path(name).stem}"
        Image.fromarray(np.rint(pred.clip(0, 1) * 255).astype(np.uint8)).save(output / f"{stem}.png")
        row["render"] = f"{stem}.png"
        error = (np.abs(pred - target) * 4).clip(0, 1)
        error[~mask] = 0
        Image.fromarray(np.rint(error * 255).astype(np.uint8)).save(output / f"{stem}_error.png")
        row["error"] = f"{stem}_error.png"
        # A one-hot color per contributor gives its class's composited weight at each pixel.
        # Black means missing opacity or excluded pixels. Anisotropic footprints count
        # as broad only when the minor axis alone is wider than a measured bright core.
        off = np.abs(offset) > TOLERANCE
        broad = (widths[:, 0] > diameter) if diameter is not None else np.zeros_like(off)
        category = np.where(off, 0, np.where(broad, 2, 1))
        with torch.no_grad():
            colors = torch.zeros_like(model.get_xyz)
            colors[ids, torch.as_tensor(category, device=ids.device)] = 1.0
            mass_map = tags(colors)[:, y : y + h, x : x + w].permute(1, 2, 0).cpu().numpy()
            del colors
        mass_map[~mask] = 0
        Image.fromarray(np.rint(mass_map.clip(0, 1) * 255).astype(np.uint8)).save(output / f"{stem}_mass.png")
        row["mass_map"] = f"{stem}_mass.png"
        row["verdict"] = row["bright_cores" if diameter is not None else "floor"]["verdict"]
        print(row["verdict"], flush=True)
        metric = row["bright_cores"] if diameter is not None else row["floor"]
        if "off_plane_share" in metric:
            print(
                f"  Center weight off the floor {metric['off_plane_share']:.1%}. "
                f"Median footprint {metric['minor_fwhm_px']['p50']:.2f} × "
                f"{metric['major_fwhm_px']['p50']:.2f} px.",
                flush=True,
            )
        if row["core_contrast_ratio"] is not None:
            print(
                f"  Median photo spot diameter {diameter:.2f} px. "
                f"Rendered spot contrast {row['core_contrast_ratio']:.0%} of the photo.",
                flush=True,
            )
        rows.append(row)
    return rows


def checkpoint_plan(suite, requested, plain_only):
    def present(path):
        return any((path / "point_cloud").glob("iteration_*/point_cloud.ply"))

    def native(path):
        spec = model_info(path)
        if spec["type"] != "3dgs":
            raise ValueError(f"Expected native 3DGS at {path}, found {spec['type']}")
        for name in (REFERENCE, NEIGHBOR):
            if name not in spec["records"]:
                raise ValueError(f"{name} is missing from the original cameras for {path}")
        return spec

    main = native(suite / "3dgs")
    specs = [main]
    notices = []
    fixed = suite / "3dgs_fixed_lidar"
    if not plain_only:
        if present(fixed):
            other = native(fixed)
            if main["splits"]["train"] != other["splits"]["train"]:
                raise ValueError(
                    "Adaptive and fixed-LiDAR checkpoints use different training images. Use --plain-only for adaptive alone."
                )
            specs.append(other)
        else:
            notices.append("The fixed-LiDAR checkpoint is unavailable. Showing adaptive 3DGS.")
    step = shared_iteration(specs, requested)
    plan = [(spec, step) for spec in specs]
    if not plain_only:
        control = suite.parent / "fixed_lidar_one_view_09882_knn2x"
        if (control / "point_cloud/iteration_2000/point_cloud.ply").is_file():
            probe = native(control)
            if probe["splits"]["train"] == {REFERENCE}:
                plan.append((probe, 2000))
            else:
                notices.append("Skipped the one-view control because its saved training split differs.")
        else:
            notices.append("The optional 2,000-step one-view control is unavailable.")
    return plan, notices


def write_report(output, rows, plane, notices):
    def number(v, digits=2):
        return f"{v:.{digits}f}" if isinstance(v, (int, float)) else str(v)

    def picture(label, file):
        return (
            f'<figure><figcaption>{escape(label)}</figcaption><a href="{escape(file)}">'
            f'<img src="{escape(file)}" alt="{escape(label)}"></a></figure>'
        )

    sections = []
    for row in rows:
        scope = "bright_cores" if row["texture"]["median_diameter_px"] is not None else "floor"
        name = Path(row["frame"]).stem
        images = "".join(
            picture(label, file)
            for label, file in [
                ("Photo", f"{name}_gt.png"),
                ("Render", row["render"]),
                ("Absolute error ×4", row["error"]),
                ("Compositing weights", row["mass_map"]),
                ("Detected cores in green", f"{name}_cores.png"),
            ]
        )
        entries = []
        for key, label in [("floor", "Supported floor"), ("bright_cores", "Bright cores")]:
            m = row[key]
            if "minor_fwhm_px" not in m:
                entries.append(f'<tr><th>{label}</th><td colspan="5">Insufficient weight or pixels</td></tr>')
                continue
            broad = f"{m['broad_on_plane_share']:.0%}" if "broad_on_plane_share" in m else "—"
            entries.append(
                f"<tr><th>{label}</th><td>{m['minor_fwhm_px']['p50']:.2f} × {m['major_fwhm_px']['p50']:.2f} px</td>"
                f"<td>{broad}</td><td>{m['front_of_plane_share']:.0%} / {m['behind_plane_share']:.0%}</td>"
                f"<td>{m['camera_z']['p50']:.3f} m</td><td>{m['mean_alpha']:.0%}</td></tr>"
            )
        core = row["texture"]["median_diameter_px"]
        core_text = f"{core:.2f} px" if core is not None else "insufficient samples"
        contrast = row["core_contrast_ratio"]
        contrast_text = f"{contrast:.0%}" if contrast is not None else "insufficient samples"
        shift = ", ".join(f"{v:+.2f}" for v in row["centered_minus_colmap_px"])
        sections.append(
            f"<section><h2>{escape(row['model'])} · {row['iteration']:,} steps · {escape(row['frame'])} ({row['split']})</h2>"
            f"<p><strong>{escape(row[scope]['verdict'])}</strong></p><p>Bright-core diameter {core_text} "
            f"({row['texture']['count']} components). Rendered spot contrast {contrast_text} of the photo. Supported floor PSNR {number(row['crop_psnr_db'])} dB. "
            f"LiDAR median camera depth {row['floor_z']:.3f} m. "
            f"Centered minus calibrated principal point ({shift}) px. AA {'on' if row['aa'] else 'off'}.</p>"
            f'<div class="photos">{images}</div><table><tr><th>Pixels</th><th>Median minor × major FWHM</th>'
            "<th>Broad on-floor weight</th><th>Off-floor weight, front / behind</th><th>Median center depth</th>"
            f"<th>Mean opacity</th></tr>{''.join(entries)}</table></section>"
        )
    style = """body{font:16px system-ui;background:#121a24;color:#e8edf5;margin:2vw}a{color:#96daff}
section{margin:2em 0}table{border-collapse:collapse}th,td{padding:.5em;text-align:left;border-bottom:1px solid #495660}
.photos{display:flex;flex-wrap:wrap;gap:10px}figure{margin:0}img{max-width:min(430px,90vw);height:auto}figcaption{margin:.5em 0}"""
    notes = "".join(f"<p>{escape(n)}</p>" for n in notices)
    html = (
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>3DGS floor footprint audit</title><style>{style}</style><h1>3DGS floor footprint audit</h1>"
        "<p>These native-resolution crops cover the same physical floor patch in two original photos. "
        "The comparisons use the saved Gaussians and the native centered 3DGS camera. The LiDAR visibility mask uses "
        "the recorded camera calibration. The principal-point offset below quantifies their difference. Spot contrast compares "
        "brightness above the local average at the same marked photo pixels. A shifted spot can also reduce this value.</p>"
        "<p>Each splat is weighted by its opacity times the remaining transmittance at each measured pixel. "
        "In the weight image, red comes from centers more than 2.5 cm from the fitted floor, blue from floor centers "
        "whose smaller projected full width at half maximum (FWHM) exceeds the measured bright-core diameter, "
        "and green from the remaining floor centers. If no diameter is available all on-floor weight is green. "
        "Dark areas have little opacity or lack common LiDAR support. Click any crop for native pixels.</p>"
        "<p>The one-view run, when present, is a separate control. A large footprint or an off-plane center can "
        "explain smoothing but its presence alone cannot establish the cause. "
        '<a href="report.json">Full measurements</a> · <a href="plane_support.png">Reference LiDAR support</a></p>'
        + notes
        + "".join(sections)
    )
    (output / "index.html").write_text(html)
    atomic_json(
        output / "report.json",
        dict(
            reference=REFERENCE,
            neighbor=NEIGHBOR,
            plane=plane,
            plane_tolerance_m=TOLERANCE,
            notices=notices,
            measurements=rows,
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=SUITE)
    parser.add_argument("--gpu", help="Physical GPU. Defaults to CUDA_VISIBLE_DEVICES or an idle GPU.")
    parser.add_argument("--iteration", type=int, default=-1, help="Shared main checkpoint (default latest common)")
    parser.add_argument("--plain-only", action="store_true", help="Measure only ordinary adaptive 3DGS")
    args = parser.parse_args(argv)
    suite = args.suite.expanduser().resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(select_gpu(args.gpu))
    plan, notices = checkpoint_plan(suite, args.iteration, args.plain_only)
    for message in notices:
        print(message, flush=True)
    root = suite / REPORT
    root.mkdir(parents=True, exist_ok=True)
    print(
        f"Using GPU {os.environ['CUDA_VISIBLE_DEVICES']}. "
        + ", ".join(f"{s['path'].name} at {i:,}" for s, i in plan),
        flush=True,
    )
    print(f"Fitting the common visible floor patch. Report: {root / 'index.html'}", flush=True)
    with tempfile.TemporaryDirectory(dir=root) as temp:
        output = Path(temp) / "report"
        output.mkdir()
        data, plane = patches(suite, output)
        rows = []
        for spec, step in plan:
            rows.extend(run_model(spec, step, data, plane, output))
        write_report(output, rows, plane, notices)
        published = root / f"iteration_{plan[0][1]}_{uuid4().hex[:10]}"
        output.rename(published)
        target = published.name + "/index.html"
        atomic_write(
            root / "index.html",
            lambda path: path.write_text(
                f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url={target}">'
                f'<title>3DGS floor footprint audit</title><a href="{target}">Open completed audit</a>'
            ),
        )
    print(f"Saved {root / 'index.html'}", flush=True)
    from scripts.refresh_orange4f_dashboard import refresh

    refresh(suite)


if __name__ == "__main__":
    main()
