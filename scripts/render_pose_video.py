#!/usr/bin/env python3
"""Render a synchronized COLMAP camera, calibrated frustum, LiDAR and original-image MP4."""
import argparse
import json
import math
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
from plyfile import PlyData
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_all_frames import read_config
from scripts.keyframe_pipeline.common import project, read_model
from scripts.render_model_comparison import write_video

RUNS = Path("/data/3dgs_outputs/orange4f/3dgs_finetuning")
SUITE = RUNS / "stride3_noaa_v1/suite.json"
BACKGROUND = (16, 22, 30)
PATH = (137, 144, 161)
CAMERA = (255, 177, 67)
FRUSTUM = (57, 222, 219)
TEXT = (227, 233, 240)


def frustum_vertices(view, camera, depth):
    """Apex and TL/TR/BR/BL corners at optical-axis depth, in world coordinates."""
    fx, fy, cx, cy = camera.params
    pixels = np.array([[0, 0], [camera.width, 0], [camera.width, camera.height], [0, camera.height]])
    rays = np.column_stack(((pixels - [cx, cy]) / [fx, fy], np.ones(4)))
    return np.vstack((view.C, view.C + depth * rays @ view.R))


def overview_basis(azimuth, elevation):
    a, e = np.deg2rad([azimuth, elevation])
    toward_eye = [np.cos(a)*np.cos(e), np.sin(a)*np.cos(e), np.sin(e)]
    right = [-np.sin(a), np.cos(a), 0.]
    down = np.cross(right, toward_eye)
    return np.array([right, down, toward_eye])


def coordinates(rows):
    return np.column_stack([rows[k] for k in "xyz"]).astype(np.float64, copy=False)


def chunks(vertices):
    for start in range(0, len(vertices), 262144):
        rows = vertices[start:start+262144]
        xyz = coordinates(rows)
        valid = np.isfinite(xyz).all(axis=1)
        if valid.any():
            yield start + np.flatnonzero(valid), xyz[valid]


class Overview:
    """Rasterize every finite LiDAR return once, retaining the front point at each overview pixel."""

    def __init__(self, vertices, views, cameras, width, height, azimuth, elevation, depth):
        self.basis = overview_basis(azimuth, elevation)
        self.origin = views[0].C
        centers = np.stack([v.C for v in views])
        projected = self.to_view(centers)
        low, high = projected.min(0), projected.max(0)
        finite_count = 0
        for _, xyz in chunks(vertices):
            p = self.to_view(xyz)
            low = np.minimum(low, p.min(0))
            high = np.maximum(high, p.max(0))
            finite_count += len(xyz)
        if finite_count == 0:
            raise ValueError("No finite XYZ coordinates in the input PLY")
        self.depth = depth if depth is not None else .10 * max((high-low).max(), 1e-6)
        corners = np.concatenate([frustum_vertices(v, cameras[v.camera_id], self.depth) for v in views])
        p = self.to_view(corners)
        low, high = np.minimum(low, p.min(0)), np.maximum(high, p.max(0))
        self.scale = .94 * np.min(np.array([width-1, height-1]) / np.maximum((high-low)[:2], 1e-12))
        self.offset = (np.array([width-1, height-1]) - (low+high)[:2] * self.scale) / 2
        self.route = self.pixels(projected)
        self.background = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
        zbuffer = np.full(width*height, -np.inf)
        winners = np.full(width*height, -1, dtype=np.int64)
        for ids, xyz in chunks(vertices):
            p = self.to_view(xyz)
            uv = self.pixels(p)
            idx = uv[:, 1]*width + uv[:, 0]
            np.maximum.at(zbuffer, idx, p[:, 2])
            front = p[:, 2] == zbuffer[idx]
            winners[idx[front]] = ids[front]
        occupied = winners >= 0
        rows = vertices[winners[occupied]]
        self.world = np.full((height, width, 3), np.nan)
        self.world.reshape(-1, 3)[occupied] = coordinates(rows)
        if {"red", "green", "blue"} <= set(rows.dtype.names):
            rgb = np.column_stack([rows[k] for k in ("red", "green", "blue")])
            if np.issubdtype(rgb.dtype, np.floating) and rgb.max() <= 1.:
                rgb *= 255
            elif rgb.dtype.kind == "u" and rgb.dtype.itemsize > 1:
                rgb = rgb * (255. / np.iinfo(rgb.dtype).max)
            colors = 20 + .78 * rgb.clip(0, 255)
        else:
            shade = (zbuffer[occupied] - low[2]) / max(high[2] - low[2], 1e-12)
            colors = (80 + 115 * shade[:, None]) * np.array([.88, .94, 1.])
        self.background.reshape(-1, 3)[occupied] = colors.astype(np.uint8)
        self.highlight = (.38*self.background + .62*np.array(FRUSTUM)).astype(np.uint8)
        cv2.polylines(self.background, [self.route], False, PATH, 1, cv2.LINE_AA)
        print(f"Overview uses {finite_count:,}/{len(vertices):,} points; frustum depth {self.depth:.4g} scene units.",
              flush=True)

    def to_view(self, xyz):
        return np.einsum("nj,ij->ni", xyz-self.origin, self.basis, optimize=False)

    def pixels(self, view_xyz):
        return np.rint(view_xyz[:, :2]*self.scale + self.offset).astype(np.int32)

    def draw(self, index, view, camera):
        panel = self.background.copy()
        uv = self.pixels(self.to_view(frustum_vertices(view, camera, self.depth)))
        # Under orthographic projection the entire pyramid fits in its 2D bounding box.
        x, y = np.maximum(uv.min(0), 0)
        end_x, end_y = np.minimum(uv.max(0) + 1, panel.shape[1::-1])
        area = np.s_[y:end_y, x:end_x]
        xyz = self.world[area]
        pixels, z = project(xyz.reshape(-1, 3), view.R, view.C, camera.params)
        inside = ((z > 0) & (z <= self.depth) & (pixels[:, 0] >= 0) & (pixels[:, 0] < camera.width)
                  & (pixels[:, 1] >= 0) & (pixels[:, 1] < camera.height)).reshape(xyz.shape[:2])
        panel[area][inside] = self.highlight[area][inside]
        cv2.polylines(panel, [self.route[max(0, index-20):index+1]], False, CAMERA, 2, cv2.LINE_AA)
        for corner in uv[1:]:
            cv2.line(panel, uv[0], corner, FRUSTUM, 2, cv2.LINE_AA)
        cv2.polylines(panel, [uv[1:]], True, FRUSTUM, 2, cv2.LINE_AA)
        cv2.line(panel, uv[1], uv[2], CAMERA, 2, cv2.LINE_AA)  # Top of the captured image shows roll.
        cv2.circle(panel, uv[0], 6, BACKGROUND, -1, cv2.LINE_AA)
        cv2.circle(panel, uv[0], 4, CAMERA, -1, cv2.LINE_AA)
        return panel


def video_frames(images, views, cameras, indices, overview, panel_width, panel_height, band):
    scale = band / 54
    def label(canvas, text, x, y, color=TEXT, size=.78):
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size*scale, color,
                    max(1, round(scale)), cv2.LINE_AA)

    for index in tqdm(indices, desc="Encoding pose + RGB", unit="frame", file=sys.stdout):
        view = views[index]
        camera = cameras[view.camera_id]
        path = images / view.name  # read_model validates each COLMAP-relative image name.
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if bgr is None or bgr.shape[:2] != (camera.height, camera.width):
            raise ValueError(f"Cannot read {path} at the COLMAP resolution {camera.width}x{camera.height}")
        ratio = min(panel_width/camera.width, panel_height/camera.height)
        size = tuple(max(1, round(n*ratio)) for n in (camera.width, camera.height))
        resized = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)
        canvas = np.full((panel_height + 2*band, 2*panel_width, 3), BACKGROUND, dtype=np.uint8)
        canvas[band:band+panel_height, :panel_width] = overview.draw(index, view, camera)
        x = panel_width + (panel_width-size[0])//2
        y = band + (panel_height-size[1])//2
        canvas[y:y+size[1], x:x+size[0]] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        margin = max(6, band//3)
        baseline = band*2//3
        label(canvas, "3D trajectory", margin, baseline)
        label(canvas, "Source image", panel_width+margin, baseline)
        footer = band + panel_height + baseline
        label(canvas, "Path", margin, footer, PATH)
        label(canvas, "Camera", int(.16*panel_width), footer, CAMERA)
        label(canvas, f"Frustum  /  depth {overview.depth:.3g}", int(.35*panel_width), footer, FRUSTUM)
        label(canvas, f"{index+1}/{len(views)}   {view.name}", panel_width+margin, footer)
        yield canvas


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, help="Orange4F suite.json supplying registered source and initialization PLY")
    p.add_argument("--source", type=Path, help="Undistorted COLMAP dataset with images/ (use with --point-cloud)")
    p.add_argument("--point-cloud", type=Path, help="Full raw LiDAR PLY in this source's world coordinates")
    p.add_argument("--output", type=Path, default=RUNS/"pose_trajectory.mp4")
    p.add_argument("--width", type=int, default=1920, help="Combined width of both panels")
    p.add_argument("--fps", type=float, default=24)
    p.add_argument("--stride", type=int, default=1, help="Use every Nth registered frame, in natural filename order")
    p.add_argument("--start-frame", help="Start at this exact registered image name")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--frustum-depth", type=float, help="Forward extent in scene units (default 10%% of scene span)")
    p.add_argument("--azimuth", type=float, default=-55, help="Observer angle in the world XY plane, in degrees")
    p.add_argument("--elevation", type=float, default=48, help="Observer elevation above the world XY plane, in degrees")
    args = p.parse_args(argv)
    if ((args.source is None) != (args.point_cloud is None)) or (args.source and args.suite):
        p.error("use --source and --point-cloud together, or use --suite")
    if args.width < 640 or args.width % 2 or not math.isfinite(args.fps) or args.fps <= 0:
        p.error("--width must be even and at least 640; --fps must be finite and positive")
    if args.stride < 1 or (args.max_frames is not None and args.max_frames < 1):
        p.error("--stride and --max-frames must be positive")
    if args.frustum_depth is not None and (args.frustum_depth <= 0 or not math.isfinite(args.frustum_depth)):
        p.error("--frustum-depth must be finite and positive")
    if not math.isfinite(args.azimuth) or not -90 <= args.elevation <= 90:
        p.error("provide a finite azimuth and an elevation in [-90, 90] degrees")
    if args.output.suffix.lower() != ".mp4":
        p.error("--output must end in .mp4")
    folder = "images"
    if args.source:
        source, cloud = args.source.expanduser().resolve(), args.point_cloud.expanduser().resolve()
    elif args.suite is None and not SUITE.is_file():
        model = SUITE.parent / "3dgs"
        cfg = read_config(model / "cfg_args")
        source = Path(cfg["source_path"]).expanduser().resolve()
        cloud = model / "input.ply"
        folder = cfg.get("images", folder)
    else:
        suite_path = (args.suite or SUITE).expanduser()
        suite = json.loads(suite_path.read_text())
        source, cloud = (Path(s).expanduser().resolve() for s in (suite["source"], suite["cloud"]["cached"]))
        print(f"Initialization originally from {suite['cloud'].get('original', cloud)}", flush=True)
    print(f"Reading COLMAP from {source} and point cloud from {cloud}", flush=True)
    _, cameras, views, _ = read_model(source)
    names = [view.name for view in views]
    if args.start_frame is not None and args.start_frame not in names:
        p.error(f"start frame is absent from this reconstruction: {args.start_frame}")
    start = names.index(args.start_frame) if args.start_frame is not None else 0
    indices = range(start, len(views), args.stride)[:args.max_frames]
    vertices = PlyData.read(str(cloud), mmap="r")["vertex"].data
    fields = set(vertices.dtype.names)
    if not set("xyz") <= fields or {"scale_0", "f_dc_0"} & fields:
        raise ValueError(f"Expected a raw point cloud with XYZ in {cloud}")
    width = args.width//2
    height = 2*round(.75*width/2)
    band = round(args.width/40)
    print(f"Preparing {len(indices):,} frames at {args.fps:g} fps and projecting {len(vertices):,} cloud points ...",
          flush=True)
    overview = Overview(vertices, views, cameras, width, height, args.azimuth, args.elevation, args.frustum_depth)
    frames = video_frames(source/folder, views, cameras, indices, overview, width, height, band)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        ready = Path(temporary)/"pose.mp4"
        write_video(ready, frames, args.width, height+2*band, args.fps, 18)
        ready.replace(output)
    print(f"Saved {output} ({len(indices)} frames, {len(indices)/args.fps:.1f} seconds)", flush=True)


if __name__ == "__main__":
    main()
