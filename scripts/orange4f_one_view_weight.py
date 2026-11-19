"""Paired native 3DGS one-photo test of surface-resolution weighting on the audited floor."""

import argparse
from html import escape
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_floor_blur import plane_grid
from scripts.orange4f_detail import SUITE, experiment_lock, select_gpu, split_settings, training_command
from scripts.orange4f_splat_footprints import (
    REFERENCE,
    NEIGHBOR,
    core_contrast_ratio,
    patches,
    validate_audit_camera,
)
from scripts.view_comparisons import image_id, model_info
from utils.checkpoint_snapshots import atomic_json, atomic_write, digest, read_snapshot
from utils.region_loss import mask_path

REPORT = "one_view_floor_weight_v1"
ARMS = ("uniform", "weighted")
STEP = 2000


def surface_weights(patch, plane):
    """Area of a single native pixel projected onto the fitted surface (square meters)."""
    view, cam, rect = patch["view"], patch["cam"], patch["rect"]
    grid = plane_grid(view, cam, rect, np.asarray(plane["center"]), np.asarray(plane["normal"]))
    dy, dx = np.gradient(grid, axis=(0, 1))
    area = np.linalg.norm(np.cross(dx, dy), axis=-1)
    mask = patch["mask"]
    measured = area[mask]
    if not measured.size or not np.isfinite(measured).all() or measured.min() <= 0:
        raise ValueError("No finite positive pixel areas on the supported floor")
    best_area = np.percentile(measured, 10)
    extra = np.zeros(mask.shape, np.uint8)
    extra[mask] = np.rint(255 * np.minimum(1.0, best_area / measured)).astype(np.uint8)
    x, y, w, h = rect
    full = np.zeros((cam.height, cam.width), np.uint8)
    full[y : y + h, x : x + w] = extra
    weights = 1.0 + extra[mask] / 255.0
    return full, dict(
        pixel_area_mm2_p10_p50_p90=(np.percentile(measured, [10, 50, 90]) * 1e6).tolist(),
        floor_weight_min_median_max=[float(weights.min()), float(np.median(weights)), float(weights.max())],
        supported_pixels=int(mask.sum()),
        image_pixels=int(full.size),
    )


def crop_result(path, patch, output):
    x, y, w, h = patch["rect"]
    with Image.open(path) as image:
        if image.size != (patch["cam"].width, patch["cam"].height):
            raise ValueError(f"Expected native-resolution render at {path}")
        rgb = np.asarray(image.convert("RGB").crop((x, y, x + w, y + h)))
    Image.fromarray(rgb).save(output)
    mask = patch["mask"]
    err = (rgb.astype(np.float64) - patch["rgb"]) / 255.0
    mse = float(np.mean(err[mask] ** 2))
    return dict(
        render=output.name,
        psnr_db=float(-10 * np.log10(mse)) if mse else "inf",
        spot_contrast=core_contrast_ratio(patch["rgb"], rgb, patch["cores"])
        if patch["detail"]["median_diameter_px"] is not None
        else None,
    )


def html_index(root, status, rows):
    def pic(file, caption):
        return f'<figure><figcaption>{caption}</figcaption><a href="{file}"><img src="{file}"></a></figure>'

    sections = []
    for name, label in [(REFERENCE, "Training photo"), (NEIGHBOR, "Neighbor excluded from training")]:
        subset = [r for r in rows if r["frame"] == name]
        if not subset:
            continue
        final = {r["arm"]: r["spot_contrast"] for r in subset if r["step"] == STEP}
        summary = ""
        if all(final.get(a) is not None for a in ARMS):
            baseline, weighted = final["uniform"], final["weighted"]
            summary = (
                f"<p>At {STEP:,} steps the uniform run reproduces {baseline:.1%} of the photo's spot "
                f"contrast and the weighted run reproduces {weighted:.1%} "
                f"({100 * (weighted - baseline):+.1f} percentage points).</p>"
            )
        panels = [pic(f"{Path(name).stem}_gt.png", "Photo")]
        for r in subset:
            m = r["spot_contrast"]
            contrast = "unavailable" if m is None else f"{m:.1%}"
            panels.append(
                pic(
                    r["render"],
                    f"{r['arm']} · {r['step']:,} steps · {float(r['psnr_db']):.2f} dB · spot contrast {contrast}",
                )
            )
        sections.append(f"<h2>{label} · {name}</h2>{summary}<div>{''.join(panels)}</div>")
    previews = " · ".join(f'<a href="{a}/debug_previews/index.html">{a} live previews</a>' for a in ARMS)
    text = (
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        + ('<meta http-equiv="refresh" content="15">' if status != "Complete" else "")
        + "<title>One-view floor weighting</title><style>body{font:16px system-ui;margin:2vw;background:#14202b;color:white}"
        "a{color:#9de0ff}div{display:flex;gap:10px;flex-wrap:wrap}figure{margin:0}img{max-width:min(420px,92vw)}</style>"
        f"<h1>One-view floor weighting</h1><p>{escape(status)}</p>"
        "<p>Both fits use the full 09882.jpg photo and original pose. Only the weighted run gives "
        "LiDAR-supported floor pixels extra RGB loss. White in the weight map means 2× and black means 1×. "
        "Spot contrast reports how much of the photographed speckle contrast appears at the same pixels. "
        "Measurements use the same visible floor area in each run.</p>"
        '<p><a href="weights.png">Training weight map</a> · <a href="results.json">Measurements</a></p>'
        + previews
        + (
            ' · <a href="comparison/index.html">Full photos, renders and errors</a>'
            if status == "Complete"
            else ""
        )
        + "".join(sections)
    )
    atomic_write(root / "index.html", lambda p: p.write_text(text))


def collect(root, data):
    rows = []
    for index, arm in enumerate(ARMS):
        preview = root / arm / "debug_previews"
        history = preview / "history.json"
        if history.is_file():
            for snap in json.loads(history.read_text())["snapshots"]:
                step = snap["iteration"]
                if step >= STEP:
                    continue
                for frame in snap["frames"]:
                    if frame["name"] == REFERENCE and frame["split"] == "train":
                        name = REFERENCE
                        row = crop_result(
                            preview / frame["render"], data[name], root / f"{arm}_{step}_{Path(name).stem}.png"
                        )
                        rows.append(dict(arm=arm, frame=name, step=step, **row))
        for split, name in [("train", REFERENCE), ("eval", NEIGHBOR)]:
            path = root / "comparison" / split / f"model_{index}" / "renders" / (image_id(name) + ".png")
            if path.is_file():
                row = crop_result(path, data[name], root / f"{arm}_{STEP}_{Path(name).stem}.png")
                rows.append(dict(arm=arm, frame=name, step=STEP, **row))
    return sorted(rows, key=lambda r: (r["step"], r["frame"], r["arm"]))


def launch(suite, gpu):
    root = suite / REPORT
    base = model_info(suite / "3dgs")
    if REFERENCE not in base["splits"]["train"]:
        raise ValueError(f"{REFERENCE} is absent from the original stride-3 split")
    extent, _, _ = split_settings(base)
    ratio = base["cfg"]["optimization_settings"]["percent_dense"]
    with experiment_lock(root) as lock:
        root.mkdir(exist_ok=True)
        data, plane = patches(suite, root)
        patch = data[REFERENCE]
        full, stats = surface_weights(patch, plane)
        for name, datum in data.items():
            validate_audit_camera(base["records"][name], name, datum)
        if base["cfg"].get("resolution") != 1:
            raise ValueError("This probe expects the native-resolution stride-3 suite")
        weight_dir = root / "loss_weights"
        commands = {}
        for arm in ARMS:
            cmd = training_command(base, root / arm, ratio, steps=(STEP,))
            cmd += ["--training_scene_extent", str(extent)]
            if arm == "weighted":
                cmd += ["--rgb_loss_weight_dir", str(weight_dir)]
            commands[arm] = cmd
        protocol = dict(
            commands=commands,
            source=str(base["path"]),
            extent=extent,
            train=[REFERENCE],
            eval_only=NEIGHBOR,
            selection_sha256=digest(base["path"] / "keyframes.json"),
            init_sha256=digest(base["path"] / "input.ply"),
            train_pixels_sha256=digest(patch["path"]),
            weights_sha256=hashlib.sha256(full.tobytes()).hexdigest(),
            weight_stats=stats,
        )
        plan = root / "protocol.json"
        if plan.exists() and json.loads(plan.read_text()) != protocol:
            raise ValueError(f"Inputs or settings differ from {plan}")
        if not plan.exists() and any((root / a).exists() for a in ARMS):
            raise FileExistsError(f"Unrecognized run directories in {root}")
        atomic_json(plan, protocol)
        weight_file = mask_path(weight_dir, REFERENCE)
        for path in [weight_file, root / "weights.png"]:
            atomic_write(path, lambda p: Image.fromarray(full).save(p, format="PNG"))
        rows = collect(root, data)

        def publish(status):
            atomic_json(root / "results.json", dict(status=status, weight_stats=stats, measurements=rows))
            html_index(root, status, rows)

        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)

        def run(cmd):
            try:
                subprocess.run(cmd, cwd=ROOT, env=env, check=True, pass_fds=(lock.fileno(),))
            except subprocess.CalledProcessError as exc:
                publish(f"Stopped with exit code {exc.returncode}. See the terminal for the traceback.")
                raise

        print(
            f"GPU {gpu}. Original extent {extent:.4g}. Floor weights {stats['floor_weight_min_median_max']}. "
            f"Report: {root / 'index.html'}",
            flush=True,
        )
        publish("Preparing both runs")
        from scripts.refresh_orange4f_dashboard import refresh

        refresh(suite)
        for arm in ARMS:
            out = root / arm
            out.mkdir(exist_ok=True)
            manifest = out / "keyframes.json"
            if manifest.exists() and json.loads(manifest.read_text()) != {"selected_images": [REFERENCE]}:
                raise ValueError(f"Unexpected training split in {manifest}")
            atomic_json(manifest, {"selected_images": [REFERENCE]})
            if not read_snapshot(out, STEP):
                publish(f"Training {arm} on {REFERENCE}")
                print(f"Training {arm} on one photo for {STEP:,} steps", flush=True)
                run(commands[arm])
            if not read_snapshot(out, STEP, verify=False):
                raise RuntimeError(f"Missing final {arm} snapshot")
            rows = collect(root, data)
        publish("Rendering both original cameras")
        run(
            [
                sys.executable,
                "-u",
                "scripts/render.py",
                "--models",
                *[str(root / a) for a in ARMS],
                "--iteration",
                str(STEP),
                "--split",
                "both",
                "--eval-set",
                "unselected",
                "--frames",
                REFERENCE,
                NEIGHBOR,
                "--gpu",
                gpu,
                "--output",
                str(root / "comparison"),
            ]
        )
        rows = collect(root, data)
        final = [r for r in rows if r["step"] == STEP]
        if len(final) != 4:
            raise RuntimeError("Expected both final crops for both models")
        for r in final:
            value = r["spot_contrast"]
            contrast = f"{value:.1%}" if value is not None else "unavailable"
            print(f"{r['frame']} {r['arm']}: {float(r['psnr_db']):.2f} dB, spot contrast {contrast}", flush=True)
        publish("Complete")
        print(root / "index.html", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=SUITE)
    parser.add_argument("--gpu", help="Physical CUDA device (default uses an idle GPU)")
    args = parser.parse_args(argv)
    launch(args.suite.expanduser().resolve(), select_gpu(args.gpu))


if __name__ == "__main__":
    main()
