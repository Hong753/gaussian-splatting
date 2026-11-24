"""Atomic, file://-compatible live previews and streaming evaluation outputs."""

from contextlib import contextmanager
import html
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw


def publish(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def write_json(path, obj):
    publish(path, json.dumps(obj, indent=2, allow_nan=False))


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}{path.suffix}")
    image.save(tmp, **({"quality": 92} if path.suffix == ".jpg" else {}))
    tmp.replace(path)


def to_image(t):
    return Image.fromarray((t.detach().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8))


def panel(images, labels, max_width=None):
    size = images[0].size
    w = min(size[0], max_width or size[0])
    h = round(size[1] * w / size[0])
    out = Image.new("RGB", (w * len(images), h + 26), "#141719")
    draw = ImageDraw.Draw(out)
    for i, (image, label) in enumerate(zip(images, labels, strict=True)):
        out.paste(image.resize((w, h), Image.Resampling.LANCZOS), (i * w, 26))
        draw.text((i * w + 6, 5), label, fill="white")
    return out


def score(gt, pred):
    return -10 * math.log10(max(float((gt - pred).square().mean()), 1e-12))


def summary(rows):
    return {
        k: dict(frames=len(vals), psnr_db=float(np.mean(vals)))
        for k in ("train", "test", "all")
        if (vals := [r["psnr_db"] for r in rows if k == "all" or r["split"] == k])
    }


def page(title, body, live=False):
    refresh = '<meta http-equiv="refresh" content="15">' if live else ""
    return (
        '<!doctype html><meta charset="utf-8">'
        + refresh
        + '<meta name="viewport" content="width=device-width">'
        "<style>body{background:#17191b;color:#eee;font:16px system-ui;margin:24px}img{max-width:100%;height:auto}"
        "a{color:#abd8ff}figure{margin:18px 0}p{line-height:1.5}</style><title>"
        + html.escape(title)
        + "</title><h1>"
        + html.escape(title)
        + "</h1>"
        + body
    )


def dashboard(run, message, rows=(), metrics=None):
    body = "<p>" + html.escape(message) + "</p>"
    if metrics:
        body += (
            "<p>"
            + " &nbsp; ".join(f"{k} {v['psnr_db']:.2f} dB ({v['frames']} views)" for k, v in metrics.items())
            + "</p>"
        )
    snapshots = sorted((run / "previews").glob("*/metrics.json"), reverse=True)
    if snapshots:
        body += "<details><summary>Preview history and PSNR</summary><ul>"
        for path in snapshots:
            item = json.loads(path.read_text())
            m = item["metrics"]
            link = path.with_name("index.html").relative_to(run).as_posix()
            body += (
                f'<li><a href="{link}">Step {item["step"]:,}</a> · train {m["train"]["psnr_db"]:.2f} dB'
                f" · held-out {m['test']['psnr_db']:.2f} dB</li>"
            )
        body += "</ul></details>"
    if (run / "density/index.html").is_file():
        body += '<p><a href="density/index.html">Mesh density before and after expansion</a></p>'
    meshes = sorted((run / "checkpoints").glob("mesh_*.ply"))
    if meshes:
        link = meshes[-1].relative_to(run).as_posix()
        body += f'<p><a href="{link}">Latest checkpoint mesh ({meshes[-1].stem[5:]})</a></p>'
    for r in rows:
        body += (
            f"<figure><figcaption>{html.escape(r['name'])} · {r['split']} · {r['psnr_db']:.2f} dB</figcaption>"
            f'<img src="{html.escape(r["panel"], quote=True)}"></figure>'
        )
    if (run / "final/index.html").is_file():
        body += '<p><a href="final/index.html">All frames, mesh and trajectories</a></p>'
    publish(run / "index.html", page("Orange4F joint training", body, live=True))


def density_report(run):
    body = "<p>The same checkpoint Gaussians supply camera-z on rays with alpha above 0.7. Agreement is within 5% of Gaussian z. Both percentages include missing mesh hits in the denominator.</p>"
    for stage in ("before", "after"):
        item = json.loads((run / "density" / f"{stage}.json").read_text())
        rows = item["frames"]
        body += f"<h2>{stage.title()} at step {item['step']:,}</h2><p>{item['sites']:,} sites · {item['pivots']:,} pivots · {item['faces']:,} faces</p>"
        for split in ("train", "test"):
            split_rows = [r for r in rows if r["split"] == split]
            rays = sum(r["reference_rays"] for r in split_rows)
            body += f"<p>{split} · {rays:,} reference rays"
            if rays:
                hit = sum(r["mesh_hits"] for r in split_rows) / rays
                agree = sum(r["agreeing_hits"] for r in split_rows) / rays
                body += f" · mesh hit {hit:.2%} · depth agreement {agree:.2%}"
            body += "</p>"
        body += (
            "<p>"
            + " · ".join(
                f'<a href="../{html.escape(r["panel"], quote=True)}">{html.escape(r["name"])}</a>'
                for r in rows
            )
            + "</p>"
        )
    publish(run / "density/index.html", page("Mesh density comparison", body))


def depth_images(gs, mesh):
    import torch

    good = (gs > 0) & gs.isfinite()
    zs = gs[good][::64]
    limit = float(torch.quantile(zs, 0.95)) if zs.numel() else 1.0
    return [to_image((z / max(limit, 1e-6)).nan_to_num()) for z in (gs, mesh)]


@contextmanager
def video(path, width, height, fps=30):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial.mp4")
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-threads",
        "4",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def append(image):
        if image.size != (width, height):
            canvas = Image.new("RGB", (width, height))
            canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
            image = canvas
        proc.stdin.write(image.convert("RGB").tobytes())

    try:
        yield append
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError(f"ffmpeg failed writing {path}")
        tmp.replace(path)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        tmp.unlink(missing_ok=True)


def export_mesh(path, verts, faces, colors):
    from plyfile import PlyData, PlyElement

    if not len(faces):
        raise RuntimeError("The trained mesh has zero faces")
    used, remap = np.unique(faces, return_inverse=True)
    verts, colors = verts[used], colors[used]
    points = np.empty(
        len(verts),
        dtype=[(k, "<f4") for k in ("x", "y", "z")] + [(k, "u1") for k in ("red", "green", "blue")],
    )
    for i, k in enumerate(("x", "y", "z")):
        points[k] = verts[:, i]
    for i, k in enumerate(("red", "green", "blue")):
        points[k] = np.rint(255 * np.clip(colors[:, i], 0, 1)).astype(np.uint8)
    triangles = np.empty(len(faces), dtype=[("vertex_indices", "<i4", (3,))])
    triangles["vertex_indices"] = remap.reshape(-1, 3)
    tmp = path.with_suffix(".partial.ply")
    PlyData(
        [PlyElement.describe(points, "vertex"), PlyElement.describe(triangles, "face")], text=False
    ).write(str(tmp))
    tmp.replace(path)


def gallery(out, rows, metrics, has_mesh):
    body = (
        "<p>" + " · ".join(f"{k} {v['psnr_db']:.2f} dB ({v['frames']})" for k, v in metrics.items()) + "</p>"
    )
    body += '<p><a href="original.mp4">Original trajectory</a> · <a href="random.mp4">Random trajectory</a>'
    if has_mesh:
        body += ' · <a href="mesh.ply">Joint mesh</a> · <a href="random_mesh.mp4">Mesh normals trajectory</a>'
    body += (
        "</p><ol>"
        + "".join(
            f'<li><a href="{html.escape(r["comparison"], quote=True)}">{html.escape(r["name"])}</a> '
            f"· {r['split']} · {r['psnr_db']:.2f} dB</li>"
            for r in rows
        )
        + "</ol>"
    )
    publish(out / "index.html", page("Orange4F evaluation", body))
