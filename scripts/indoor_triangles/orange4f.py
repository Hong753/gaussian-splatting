"""Prepare a local Orange4F chart directly from the capture's RGB and COLMAP poses."""

from argparse import Namespace
from html import escape
import json
from pathlib import Path
import re

import numpy as np
from scipy.spatial.distance import cdist

from utils.colmap_paths import colmap_model_dir

from .experiment import save_json
from .room import HOLDOUT_GAP_M, fixed_pose, model_cameras, prepare

SOURCE = Path("/data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full")
REFERENCE_VIEW = "12569.jpg"


def frame_order(name):
    return tuple((part.isdigit(), int(part) if part.isdigit() else part.casefold())
                 for part in re.split(r"(\d+)", name))


def choose_split(names, centers, anchor, fit_views, edit_views, final_views):
    """Reserve translated validation views, then sample the surviving fitting sequence."""
    distances = cdist(centers, centers)
    fit_available = np.ones(len(names), dtype=bool)
    minimum_fit = min(8, fit_views)
    selected = {}
    reserved = []
    for role, count in (("final", final_views), ("edit", edit_views)):
        held_out = []
        gap = HOLDOUT_GAP_M[role]
        for position in np.linspace(0, len(names)-1, count+2)[1:-1]:
            for index in np.argsort(np.abs(np.arange(len(names))-position), kind="stable"):
                # Keep holdouts separated from each other and from the reference.
                if distances[index, anchor] < gap or any(
                        distances[index, other] < max(gap, other_gap)
                        for other, other_gap in reserved):
                    continue
                allowed = fit_available & (distances[index] >= gap)
                if np.count_nonzero(allowed) < minimum_fit or distances[anchor, allowed].max() < .18:
                    continue
                fit_available = allowed
                held_out.append(index)
                reserved.append((index,gap))
                break
            else:
                raise ValueError(f"Could not reserve {count} translated {role} views while retaining "
                                 f"{minimum_fit} fitting views. Try a wider camera window or another reference.")
        selected[role] = sorted(held_out)

    available = np.flatnonzero(fit_available & (np.arange(len(names)) != anchor))
    indices = np.linspace(0, len(available)-1, min(len(available), fit_views-1)).round().astype(int)
    selected["fit"] = sorted([anchor,*available[indices]])
    return {role:[names[index] for index in selected[role]] for role in ("fit","edit","final")}


def prepare_orange4f(args):
    if args.fit_views < 4 or min(args.edit_views, args.final_views, args.window) < 1:
        raise ValueError("Use at least four fitting views and positive edit, final and window sizes.")
    if not np.isfinite(args.radius) or args.radius <= .18:
        raise ValueError("Camera radius must be greater than 0.18 m.")
    source = args.source.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Orange4F source is missing: {source}. Set --source to its COLMAP export.")
    model = colmap_model_dir(source)
    images = args.images or source/"images"
    if not images.is_dir():
        raise FileNotFoundError(f"Orange4F images are missing: {images}. Set --images to the RGB directory.")
    col, calibration, frames = model_cameras(model)
    by_name = {frame.name:frame for frame in frames.values()}
    reference = args.reference_view
    if reference not in by_name:
        matches = [name for name in by_name if Path(name).name == reference]
        if len(matches) != 1:
            raise ValueError(f"Reference frame {reference!r} is missing or ambiguous in {model}. "
                             "Set --reference-view to its full COLMAP image name.")
        reference = matches[0]

    names = sorted(by_name, key=frame_order)
    center_index = names.index(reference)
    window = names[max(0,center_index-args.window):center_index+args.window+1]
    ref_frame = by_name[reference]
    _, ref_rotation = fixed_pose(col, ref_frame)
    ref_center = -ref_rotation.T @ ref_frame.tvec
    ref_camera = calibration[ref_frame.camera_id]
    if ref_camera.model not in {"PINHOLE", "SIMPLE_PINHOLE"}:
        raise ValueError(f"Expected an undistorted camera for {reference}, got {ref_camera.model}.")
    candidates, centers = [], []
    for name in window:
        frame = by_name[name]
        camera = calibration[frame.camera_id]
        if (camera.model not in {"PINHOLE", "SIMPLE_PINHOLE"} or
                camera.height*ref_camera.width != ref_camera.height*camera.width):
            continue
        _, rotation = fixed_pose(col, frame)
        center = -rotation.T @ frame.tvec
        if (rotation[2] @ ref_rotation[2] >= np.cos(np.deg2rad(30)) and
                np.linalg.norm(center-ref_center) <= args.radius and (images/name).is_file()):
            candidates.append(name)
            centers.append(center)
    if reference not in candidates:
        raise FileNotFoundError(f"Could not open the reference RGB image: {images/reference}")
    if len(candidates) < min(8,args.fit_views) + args.edit_views + args.final_views:
        raise ValueError(f"Only {len(candidates)} nearby same-facing views for {reference}. "
                         "Try a wider --window or --radius, or another --reference-view.")
    split = choose_split(candidates, np.array(centers), candidates.index(reference),
                         args.fit_views, args.edit_views, args.final_views)
    out = args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    split_file = out/"split.json"
    save_json(split_file, split)
    save_json(out/"selection.json",dict(input_scene=str(source),image_directory=str(images.resolve()),
              colmap_model=str(model),reference_view=reference,candidates=len(candidates),
              radius_m=args.radius,window_frames_each_side=args.window,
              view_counts={key:len(value) for key,value in split.items()}))
    print(f"Orange4F chart at {reference}: {len(split['fit'])} fit, "
          f"{len(split['edit'])} edit, {len(split['final'])} final views.",flush=True)
    prep = Namespace(model=model,images=images,split=split_file,out=out/"input",width=args.width,
                     reference_view=reference,cell=args.cell,gpu_features=args.gpu_features,
                     pose_provenance=f"Fixed COLMAP BA-export intrinsics and poses from {model}.")
    prepare(prep, (col, calibration, frames))


def write_report(out):
    split = json.loads((out/"split.json").read_text())
    results = json.loads((out/"eval"/"hybrid.json").read_text())
    rows = []
    for name,view in zip(split["final"],results["views"],strict=True):
        plot = f"hybrid/final_{view['view']:02}.png"
        rows.append(f'<tr><td>{escape(name)}</td><td>{view["psnr_db"]:.2f}</td>'
                    f'<td>{view["ssim"]:.3f}</td><td>{view["mesh_hit_fraction"]:.1%}</td>'
                    f'<td>{view["opaque_subsample_fraction"]:.1%}</td>'
                    f'<td><a href="{plot}">RGB, error and mesh views</a></td></tr>')
    (out/"eval"/"index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Orange4F triangle fit</title>'
        '<style>body{font:16px system-ui;margin:2rem}td,th{padding:.5rem;text-align:left}</style>'
        '<h1>Orange4F indoor triangle fit</h1>'
        '<p><a href="paired_results.csv">Metrics CSV</a> · <a href="hybrid/all.ply">Mesh PLY</a> · '
        '<a href="hybrid/textured.obj">Textured OBJ</a></p>'
        '<table><thead><tr><th>Held-out image</th><th>PSNR (dB)</th><th>SSIM</th>'
        '<th>Mesh hit rate</th><th>Opaque coverage</th><th>Preview</th></tr></thead><tbody>'
        + "\n".join(rows) + "</tbody></table></html>\n",encoding="utf-8")
