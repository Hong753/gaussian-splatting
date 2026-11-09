"""Small COLMAP/geometry helpers without importing the CUDA training stack."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import struct

import cv2
import numpy as np
from scipy.ndimage import minimum_filter
from scipy.spatial.transform import Rotation


@dataclass
class Camera:
    id: int
    width: int
    height: int
    params: np.ndarray  # fx, fy, cx, cy; input must already be undistorted.


@dataclass
class Image:
    id: int
    camera_id: int
    name: str
    R: np.ndarray  # world -> camera
    C: np.ndarray  # camera center in the LiDAR/world frame


def natural_key(name):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", name)]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_stamp(path):
    path = Path(path).resolve()
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


def image_path(source, folder, name):
    # COLMAP names must stay relative to the requested image directory.
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or "\n" in name or "\r" in name:
        raise ValueError(f"Invalid COLMAP image name: {name!r}")
    return Path(source) / folder / relative


def read_exact(file, fmt):
    size = struct.calcsize(fmt)
    data = file.read(size)
    if len(data) != size:
        raise ValueError("Truncated COLMAP model")
    return struct.unpack(fmt, data)


def make_camera(cid, model, width, height, params):
    if model == "SIMPLE_PINHOLE":
        params = [params[0], *params]
    elif model != "PINHOLE":
        raise ValueError(f"Camera {cid}: {model} is unsupported. Undistort images/model first.")
    params = np.asarray(params, dtype=float)
    if len(params) != 4 or not np.isfinite(params).all() or min(params[:2]) <= 0 or min(width, height) <= 0:
        raise ValueError(f"Invalid camera {cid}")
    return Camera(cid, int(width), int(height), params)


def make_image(iid, camera_id, name, q, t):
    q, t = np.asarray(q, dtype=float), np.asarray(t, dtype=float)
    if not np.isfinite(q).all() or not np.isfinite(t).all() or np.linalg.norm(q) < 1e-10:
        raise ValueError(f"Invalid pose for {name}")
    R = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    return Image(iid, camera_id, name, R, -R.T @ t)


def read_model(source):
    from utils.colmap_paths import colmap_model_dir
    model = colmap_model_dir(source)
    cameras, images = {}, {}
    binary = all((model / f"{name}.bin").exists() for name in ("cameras", "images"))
    if binary:
        with (model / "cameras.bin").open("rb") as f:
            for _ in range(read_exact(f, "<Q")[0]):
                cid, model_id, width, height = read_exact(f, "<iiQQ")
                if model_id not in (0, 1):
                    raise ValueError(f"Camera {cid}: model {model_id}; undistort first (PINHOLE required).")
                params = read_exact(f, "<" + "d" * (3 if model_id == 0 else 4))
                cameras[cid] = make_camera(cid, ("SIMPLE_PINHOLE", "PINHOLE")[model_id], width, height, params)
        with (model / "images.bin").open("rb") as f:
            for _ in range(read_exact(f, "<Q")[0]):
                vals = read_exact(f, "<i7di")
                name = bytearray()
                while True:
                    byte = f.read(1)
                    if not byte:
                        raise ValueError("Truncated COLMAP image name")
                    if byte == b"\0":
                        break
                    name.extend(byte)
                image = make_image(vals[0], vals[-1], name.decode("utf8"), vals[1:5], vals[5:8])
                n = read_exact(f, "<Q")[0]
                end = f.tell() + n * 24
                if end > (model / "images.bin").stat().st_size:
                    raise ValueError("Truncated COLMAP observations")
                f.seek(end)
                images[image.id] = image
    else:
        for line in (model / "cameras.txt").read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            vals = line.split()
            cid = int(vals[0])
            cameras[cid] = make_camera(cid, vals[1], int(vals[2]), int(vals[3]), list(map(float, vals[4:])))
        with (model / "images.txt").open() as f:
            for line in f:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                vals = line.strip().split(maxsplit=9)
                image = make_image(int(vals[0]), int(vals[8]), vals[9], list(map(float, vals[1:5])), list(map(float, vals[5:8])))
                if f.readline() == "":
                    raise ValueError("Missing COLMAP image observation line")
                images[image.id] = image
    ordered = sorted(images.values(), key=lambda x: natural_key(x.name))
    if not ordered or len({im.name for im in ordered}) != len(ordered):
        raise ValueError("Empty model or duplicate image names")
    for im in ordered:
        if im.camera_id not in cameras:
            raise ValueError(f"Unknown camera {im.camera_id} for {im.name}")
        image_path(source, "images", im.name)
    stamps = [file_stamp(model / f"{name}.{'bin' if binary else 'txt'}") for name in ("cameras", "images")]
    return model, cameras, ordered, stamps


def read_gray(source, folder, image, camera, max_side):
    path = image_path(source, folder, image.name)
    # Match the training loader's stored pixels; do not auto-rotate EXIF images.
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
    if gray is None:
        raise ValueError(f"Cannot read image: {path}")
    if gray.shape != (camera.height, camera.width):
        raise ValueError(f"{path}: image size {gray.shape[::-1]} != COLMAP {(camera.width, camera.height)}")
    scale = min(1., max_side / max(gray.shape))
    if scale < 1:
        gray = cv2.resize(gray, (round(camera.width * scale), round(camera.height * scale)), interpolation=cv2.INTER_AREA)
    # COLMAP uses image coordinates with pixel centers at (0.5, 0.5).
    sx, sy = gray.shape[1] / camera.width, gray.shape[0] / camera.height
    K = camera.params * np.array([sx, sy, sx, sy])
    return gray, K, np.array([sx, sy])


def load_cloud(path, voxel_size=0.08, max_points=300_000):
    from plyfile import PlyData
    vertices = PlyData.read(str(path))["vertex"]
    points = np.column_stack([vertices[axis] for axis in "xyz"]).astype(np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 20:
        raise ValueError("LiDAR PLY has fewer than 20 finite points")
    # Representatives are original returns, not averages across depth edges.
    while True:
        _, indices = np.unique(np.floor(points / voxel_size).astype(np.int64), axis=0, return_index=True)
        if len(indices) <= max_points:
            break
        voxel_size *= max(1.1, (len(indices) / max_points) ** (1/3))
    return points[np.sort(indices)], float(voxel_size)


def project(points, R, C, K):
    xyz = (points - C) @ R.T
    z = xyz[:, 2]
    uv = xyz[:, :2] / np.maximum(z[:, None], 1e-8)
    return uv * K[:2] + K[2:], z


def visible_points(points, image, K, shape, radius=1):
    """Approximate visibility: nearest depth with a small point footprint.

    This cannot resolve occluders absent from a sparse point cloud. Callers
    additionally reject ambiguous associations/large depth differences.
    """
    h, w = shape
    uv, depth = project(points, image.R, image.C, K)
    valid = (depth > .2) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    ids = np.flatnonzero(valid)
    pixels = np.floor(uv[ids]).astype(int)
    zbuffer = np.full((h, w), np.inf)
    np.minimum.at(zbuffer, (pixels[:, 1], pixels[:, 0]), depth[ids])
    if radius:
        zbuffer = minimum_filter(zbuffer, size=2 * radius + 1, mode="constant", cval=np.inf)
    front = zbuffer[pixels[:, 1], pixels[:, 0]]
    ids = ids[depth[ids] <= front + .02 + .005 * depth[ids]]
    return ids, uv[ids], depth[ids]


def prepare_output(output, settings):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    meta = output / "settings.json"
    if meta.exists():
        if json.loads(meta.read_text()) != settings:
            raise ValueError(f"Settings/inputs changed; choose a new output directory: {output}")
    elif any(output.iterdir()):
        raise ValueError(f"Output is nonempty without matching settings: {output}")
    else:
        atomic_json(meta, settings)
    return output
