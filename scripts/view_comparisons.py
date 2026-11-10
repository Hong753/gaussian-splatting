"""Camera selection and portable, image-by-image reconstruction reports.

This module deliberately does not import CUDA or the scene package. Split
membership is determined before any rendering, from the saved training inputs.
"""
import csv
import hashlib
import html
import importlib.util
import json
import math
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_all_frames import read_config, choose_iteration
from utils.colmap_paths import colmap_model_dir
from scripts.occlusion_options import metric_directory, settings


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def image_id(name):
    return hashlib.sha256(name.encode("utf8")).hexdigest()[:24]


def read_colmap_records(source, images="images"):
    """Read calibration without importing scene.__init__ or loading image pixels."""
    import numpy as np
    spec = importlib.util.spec_from_file_location("_comparison_colmap", ROOT / "scene/colmap_loader.py")
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    source = Path(source).resolve()
    model = colmap_model_dir(source)
    binary = (model / "images.bin").is_file() and (model / "cameras.bin").is_file()
    extrinsics = (loader.read_extrinsics_binary(str(model / "images.bin")) if binary
                  else loader.read_extrinsics_text(str(model / "images.txt")))
    intrinsics = (loader.read_intrinsics_binary(str(model / "cameras.bin")) if binary
                  else loader.read_intrinsics_text(str(model / "cameras.txt")))
    records = {}
    for extr in extrinsics.values():
        intr = intrinsics[extr.camera_id]
        if intr.model == "PINHOLE":
            fx, fy, cx, cy = map(float, intr.params)
        elif intr.model == "SIMPLE_PINHOLE":
            fx, cx, cy = map(float, intr.params)
            fy = fx
        else:
            raise ValueError(f"Undistorted PINHOLE images required; got {intr.model}")
        name = extr.name
        if name in records or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Duplicate or unsafe image name: {name}")
        records[name] = dict(name=name, image_path=str(source / images / name),
                             R=loader.qvec2rotmat(extr.qvec).T.tolist(),
                             T=np.asarray(extr.tvec).tolist(), width=intr.width, height=intr.height,
                             fx=fx, fy=fy, cx=cx, cy=cy,
                             fov_x=2 * math.atan(intr.width / (2 * fx)),
                             fov_y=2 * math.atan(intr.height / (2 * fy)))
    if not records:
        raise ValueError(f"No registered images in {source}")
    return records


def split_names(names, cfg, manifest=None):
    """Match the native lexical 1-in-8 split, then apply the training manifest."""
    names = sorted(names)
    available = set(names)
    evaluation = set(names[::8]) if cfg.get("eval", False) else set()
    train = available if cfg.get("train_test_exp", False) else available - evaluation
    if manifest is not None:
        chosen = manifest["selected_images"]
        if not chosen or len(chosen) != len(set(chosen)) or not set(chosen) <= train:
            raise ValueError("Keyframe manifest is empty, duplicated, or outside the training split")
        train = set(chosen)
    return {"train": train, "eval": evaluation, "all": available}


def model_info(path):
    path = Path(path).expanduser().resolve()
    cfg = read_config(path / "cfg_args")
    marker = path / "proxygs_run.json"
    if marker.is_file():
        metadata = json.loads(marker.read_text())
        prepared = json.loads(Path(metadata["preparation"]).read_text())
        manifest_path = Path(prepared["keyframes"])
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        records = read_colmap_records(prepared["source"], manifest.get("image_directory", "images"))
        staged = {r["name"]: r for r in prepared["frames"]}
        if len(staged) != len(prepared["frames"]) or not set(staged) <= set(records):
            raise ValueError("Staged Proxy-GS frames are duplicated or absent from the original COLMAP source")
        # Preserve saved training calibration/aliases; unstaged cameras come from
        # the complete original source, never the reduced proxygs_data directory.
        records.update(staged)
        splits = {s: {r["name"] for r in prepared["frames"] if r["split"] == s}
                  for s in ("train", "eval")}
        splits["all"] = set(records)
        return dict(path=path, cfg=cfg, type="proxygs", records=records,
                    splits=splits, metadata=metadata)
    source = Path(cfg["source_path"]).expanduser().resolve()
    records = read_colmap_records(source, cfg.get("images", "images"))
    manifest = None
    if cfg.get("keyframe_manifest"):
        manifest_path = path / "keyframes.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing saved training manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
    return dict(path=path, cfg=cfg, type=cfg.get("model_type", "3dgs"),
                records=records, splits=split_names(records, cfg, manifest))


def write_all_frame_metrics(spec, rows, iteration, antialiasing=None):
    """Export the common PSNR protocol only when every registered frame was rendered."""
    available, selected = spec["splits"]["all"], spec["splits"]["train"]
    if not available or set(rows) != set(available):
        return None  # A sampled/shared subset must never masquerade as full coverage.
    if not selected <= available:
        raise ValueError("Training frames are absent from the registered camera set")
    groups = dict(all=available, selected=selected, unselected=available - selected)
    means = {}
    for group, names in groups.items():
        value = sum(float(rows[name]["psnr_db"]) for name in sorted(names)) / len(names) if names else None
        means[group] = "inf" if value == float("inf") else value
    cfg, meta = spec["cfg"], spec.get("metadata", {})
    aa = False if spec["type"] == "proxygs" else (
        antialiasing if antialiasing is not None else cfg.get("antialiasing", spec["type"] != "textured"))
    summary = dict(iteration=iteration, source=meta.get("source", cfg.get("source_path")),
        model=str(spec["path"]), model_type=spec["type"], antialiasing=aa,
        resolution_setting=cfg.get("resolution", -1),
        exposure_protocol="base appearance; no per-image exposure applied",
        metric="mean per-channel PSNR, then arithmetic mean over images",
        selection_manifest=meta.get("preparation") if spec["type"] == "proxygs" else (
            str(spec["path"] / "keyframes.json") if cfg.get("keyframe_manifest") else None),
        coverage="all original registered COLMAP frames",
        occlusion=settings(cfg), render_variant=spec.get("render_variant", ""),
        counts={group: len(names) for group, names in groups.items()}, mean_psnr_db=means)
    output = metric_directory(spec["path"], iteration, spec.get("render_variant", ""))
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / f"per_frame.csv.{uuid.uuid4().hex}.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_name", "selected_for_training", "psnr_db"])
        for name in sorted(available):
            writer.writerow([name, int(name in selected), rows[name]["psnr_db"]])
    temporary.replace(output / "per_frame.csv")
    atomic_json(output / "summary.json", summary)
    return summary


def shared_iteration(specs, requested):
    if requested != -1:
        for spec in specs:
            choose_iteration(spec["path"], requested, spec["type"])
        return requested
    sets = []
    for spec in specs:
        filename = "textured.pt" if spec["type"] == "textured" else "point_cloud.ply"
        sets.append({int(p.name[10:]) for p in (spec["path"] / "point_cloud").glob("iteration_*")
                     if p.name[10:].isdigit() and (p / filename).is_file()})
    common = set.intersection(*sets)
    if not common:
        raise ValueError("Models have no common saved iteration; compare matching checkpoints")
    return max(common)


def save_panel(gt, pred, path, label, score):
    from PIL import Image, ImageDraw
    width = min(960, gt.width)
    height = max(1, round(gt.height * width / gt.width))
    panel = Image.new("RGB", (2 * width, height + 36), "#111827")
    panel.paste(gt.resize((width, height)), (0, 36))
    panel.paste(pred.resize((width, height)), (width, 36))
    draw = ImageDraw.Draw(panel)
    draw.text((12, 10), "Actual image", fill="white")
    draw.text((width + 12, 10), f"{label} | PSNR {score:.2f} dB", fill="white")
    panel.save(path, quality=93)


def write_gallery(output, title, rows, models, metadata):
    """No server/CDN required; all images and metadata use relative paths."""
    output = Path(output)
    atomic_json(output / "summary.json", {**metadata, "models": models, "frames": rows})
    with (output / "per_frame.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_name", *[m["label"] + "_psnr_db" for m in models]])
        for row in rows:
            writer.writerow([row["name"], *[r["psnr_db"] for r in row["renders"]]])
    payload = json.dumps({"rows": rows, "models": models}, allow_nan=False).replace("<", "\\u003c")
    page = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
body{margin:0;background:#101827;color:#e5edf7;font:15px system-ui}
header{padding:20px 24px;border-bottom:1px solid #334155}h1{font-size:22px;margin:0 0 8px}
small{color:#a8bbd0}nav{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:16px}
button,select,input{background:#203149;border:1px solid #47617d;color:inherit;padding:9px;border-radius:6px}
select{max-width:45vw}input{width:220px}main{display:grid;gap:12px;padding:20px}
figure{margin:0;min-width:0;background:#17253a;border-radius:8px;overflow:hidden}
figcaption{padding:12px;font-weight:600}img{display:block;width:100%;height:auto}
#name{padding:0 24px;overflow-wrap:anywhere}a{color:#91c6ff}#empty{padding:24px}
</style></head><body><header><h1>__TITLE__</h1>
<small>Matched image names; each model uses its own saved camera pose.
Base appearance, with no per-image exposure correction. Click an image for full resolution.</small>
<nav><button id="prev">Previous</button><button id="next">Next</button>
<select id="frame"></select><input id="filter" placeholder="Filter image name">
<select id="sort"><option value="name">Image order</option><option value="worst">Lowest PSNR first</option></select>
<label><input style="width:auto" type="checkbox" id="errors"> Show absolute error ×4</label>
<span id="count"></span></nav></header><p id="name"></p><main id="panels"></main><div id="empty"></div>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent);
let rows=data.rows.slice(),index=0;
const el=id=>document.getElementById(id);
function display(){
 el('panels').replaceChildren();el('empty').textContent=rows.length?'':'No matching images.';
 if(!rows.length)return;
 index=(index+rows.length)%rows.length;el('frame').value=String(index);
 const row=rows[index];el('name').textContent=row.name;
 el('count').textContent=(index+1)+' / '+rows.length;
 const entries=[{title:'Actual image',path:row.gt},...row.renders.map((r,i)=>({
 title:data.models[i].label+' · '+(typeof r.psnr_db==='number'?r.psnr_db.toFixed(2):r.psnr_db)+' dB',path:el('errors').checked?r.error:r.image}))];
 el('panels').style.gridTemplateColumns='repeat('+entries.length+',minmax(0,1fr))';
 for(const item of entries){
 const box=document.createElement('figure'),caption=document.createElement('figcaption');
 caption.textContent=item.title;const link=document.createElement('a');link.href=item.path;link.target='_blank';
 const img=document.createElement('img');img.src=item.path;img.alt=item.title;link.append(img);box.append(caption,link);el('panels').append(box);}
}
function rebuild(){
 rows=data.rows.filter(r=>r.name.toLowerCase().includes(el('filter').value.toLowerCase()));
 if(el('sort').value==='worst')rows.sort((a,b)=>Number(a.renders[0].psnr_db)-Number(b.renders[0].psnr_db));
 el('frame').replaceChildren();rows.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=r.name;el('frame').append(o);});
 index=0;display();
}
el('prev').onclick=()=>{index--;display();};el('next').onclick=()=>{index++;display();};
el('frame').onchange=()=>{index=Number(el('frame').value);display();};
el('filter').oninput=rebuild;el('sort').onchange=rebuild;el('errors').onchange=display;
document.onkeydown=e=>{if(['INPUT','SELECT'].includes(e.target.tagName))return;if(e.key==='ArrowRight'){index++;display();}if(e.key==='ArrowLeft'){index--;display();}};
rebuild();</script></body></html>"""
    page = page.replace("__TITLE__", html.escape(title)).replace("__DATA__", payload)
    (output / "index.html").write_text(page)

