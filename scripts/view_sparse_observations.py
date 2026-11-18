#!/usr/bin/env python3
"""Browse stored COLMAP image/point links and check their coordinates against a PLY."""

import argparse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import math
import mmap
import os
from pathlib import Path
import socket
import struct
import sys
from urllib.parse import urlsplit

import numpy as np
from PIL import Image
from plyfile import PlyData
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.colmap_paths import colmap_model_dir
from utils.read_write_model import qvec2rotmat, read_cameras_binary

FEATURE = np.dtype([("xy", "<f8", 2), ("id", "<i8")])
HEADER = np.dtype([("id", "<u8"), ("xyz", "<f8", 3), ("rgb", "u1", 3), ("error", "<f8"), ("count", "<u8")])
TRACK = np.dtype([("image", "<i4"), ("feature", "<i4")])
PAIR = np.dtype([("feature", "<i8"), ("id", "<u8")])
REVERSE = np.dtype([("image", "<i4"), ("feature", "<i4"), ("row", "<u8")])


def mapped(path):
    with path.open("rb") as f:
        if path.stat().st_size < 8:
            raise ValueError(f"Truncated binary header in {path}")
        return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)


@dataclass
class Frame:
    id: int
    name: str
    camera: int
    R: np.ndarray
    t: np.ndarray
    offset: int
    count: int
    linked: int


def read_frames(buf):
    offset = 8
    frames = {}
    for _ in tqdm(range(struct.unpack_from("<Q", buf)[0]), desc="Reading images.bin", unit="image"):
        iid, *qt, cid = struct.unpack_from("<i7di", buf, offset)
        end = buf.find(b"\0", offset + 64)
        if end < 0:
            raise ValueError("Unterminated COLMAP image name")
        name = buf[offset + 64 : end].decode("utf-8")
        if Path(name).is_absolute() or ".." in Path(name).parts or not name or iid in frames:
            raise ValueError(f"Invalid or duplicate COLMAP image record: {iid} {name}")
        (count,) = struct.unpack_from("<Q", buf, end + 1)
        offset = end + 9
        records = np.frombuffer(buf, FEATURE, count, offset)
        frames[iid] = Frame(
            iid,
            name,
            cid,
            qvec2rotmat(qt[:4]),
            np.array(qt[4:]),
            offset,
            count,
            int(np.count_nonzero(records["id"] >= 0)),
        )
        offset += count * FEATURE.itemsize
    if not frames or len({f.name for f in frames.values()}) != len(frames) or offset != len(buf):
        raise ValueError("Empty, duplicate, or malformed images.bin")
    return frames


class PointIndex:
    def __init__(self, path):
        self.present = path.is_file()
        self.ids = np.empty(0, dtype=np.uint64)
        self.xyz = np.empty((0, 3))
        self.reverse = np.empty(0, REVERSE)
        self.order = None
        if not self.present:
            return
        buf = mapped(path)
        (n,) = struct.unpack_from("<Q", buf)
        extra = len(buf) - 8 - n * HEADER.itemsize
        if extra < 0 or extra % TRACK.itemsize:
            raise ValueError("Truncated points3D.bin")
        links = extra // TRACK.itemsize
        if links == 0:
            records = np.frombuffer(buf, HEADER, n, 8)
            if np.any(records["count"]):
                raise ValueError("Inconsistent points3D.bin track lengths")
            self.ids, self.xyz = records["id"], records["xyz"]
        else:
            self.ids = np.empty(n, "<u8")
            self.xyz = np.empty((n, 3), "<f8")
            reverse = np.empty(links, REVERSE)
            offset, pos = 8, 0
            header = struct.Struct("<Q3d3x8xQ")
            for row in tqdm(range(n), desc="Reading points3D.bin tracks", unit="point", mininterval=1):
                pid, x, y, z, k = header.unpack_from(buf, offset)
                self.ids[row], self.xyz[row] = pid, (x, y, z)
                offset += HEADER.itemsize
                if k:
                    if pos + k > links:
                        raise ValueError("Track length exceeds points3D.bin size")
                    obs = np.frombuffer(buf, TRACK, k, offset)
                    part = reverse[pos : pos + k]
                    part["image"], part["feature"], part["row"] = obs["image"], obs["feature"], row
                    pos += k
                    offset += k * TRACK.itemsize
            if offset != len(buf) or pos != links:
                raise ValueError("Invalid track counts in points3D.bin")
            self.reverse = reverse[np.argsort(reverse["image"], kind="stable")]
        if len(self.ids) > 1 and not np.all(self.ids[1:] > self.ids[:-1]):
            self.order = np.argsort(self.ids, kind="stable")
            sorted_ids = self.ids[self.order]
            if np.any(sorted_ids[1:] == sorted_ids[:-1]):
                raise ValueError("Duplicate POINT3D_ID in points3D.bin")

    def rows(self, ids):
        result = np.full(len(ids), -1, dtype=np.int64)
        if len(self.ids):
            pos = np.searchsorted(self.ids, ids, sorter=self.order)
            ok = np.flatnonzero(pos < len(self.ids))
            rows = pos[ok] if self.order is None else self.order[pos[ok]]
            hit = self.ids[rows] == ids[ok]
            result[ok[hit]] = rows[hit]
        return result

    def for_image(self, iid):
        images = self.reverse["image"]
        lo, hi = images.searchsorted(iid), images.searchsorted(iid, side="right")
        return self.reverse[lo:hi]


def keys(xyz, dtype):
    xyz = np.array(xyz, dtype=dtype, order="C", copy=True)
    xyz[xyz == 0] = 0  # Canonicalize signed zero before exact comparison.
    return xyz.view(np.dtype((np.void, xyz.dtype.itemsize * 3))).ravel()


class PlyIndex:
    def __init__(self, path, index_coords):
        self.vertices = PlyData.read(str(path), mmap="r")["vertex"].data
        self.dtype = np.result_type(*[self.vertices[k].dtype for k in "xyz"])
        self.sorted_keys = self.order = None
        if index_coords and len(self.vertices):
            print(f"Indexing exact XYZ for {len(self.vertices):,} PLY rows …", flush=True)
            key = np.empty(len(self.vertices), dtype=np.dtype((np.void, self.dtype.itemsize * 3)))
            for start in range(0, len(key), 500000):
                part = slice(start, start + 500000)
                key[part] = keys(self.coords(part), self.dtype)
            self.order = np.argsort(key, kind="stable")
            self.sorted_keys = key[self.order]

    def coords(self, rows):
        return np.column_stack([self.vertices[k][rows] for k in "xyz"])

    def match(self, xyz):
        # -1 absent, -2 duplicate location. An ID is never used as a PLY row.
        result = np.full(len(xyz), -1, dtype=np.int64)
        if self.sorted_keys is not None and len(xyz):
            query = keys(xyz, self.dtype)
            lo = np.searchsorted(self.sorted_keys, query)
            hi = np.searchsorted(self.sorted_keys, query, side="right")
            finite = np.isfinite(xyz).all(axis=1)
            one = (hi - lo == 1) & finite
            result[one] = self.order[lo[one]]
            result[(hi - lo > 1) & finite] = -2
        return result


def project(xyz, frame, camera):
    if camera.model == "SIMPLE_PINHOLE":
        fx, cx, cy = camera.params
        fy = fx
    elif camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params
    else:
        raise ValueError(f"{camera.model} camera: this viewer supports undistorted pinhole input")
    local = xyz @ frame.R.T + frame.t
    uv = np.full((len(local), 2), np.nan)
    ok = (local[:, 2] > 0) & np.isfinite(local).all(axis=1)
    uv[ok] = local[ok, :2] / local[ok, 2, None] * [fx, fy] + [cx, cy]
    return uv, local[:, 2]


def pair_array(features, ids):
    pairs = np.empty(len(ids), dtype=PAIR)
    pairs["feature"], pairs["id"] = features, ids
    return pairs


def finite_xy(xy):
    return xy.round(3).tolist() if np.isfinite(xy).all() else None


class Inspector:
    def __init__(self, source, ply_path=None, images="images", limit=5000):
        self.source, self.images, self.limit = Path(source), images, limit
        model = colmap_model_dir(self.source)
        self.cameras = read_cameras_binary(model / "cameras.bin")
        self.buf = mapped(model / "images.bin")
        self.frames = read_frames(self.buf)
        self.points = PointIndex(model / "points3D.bin")
        ply_path = Path(ply_path) if ply_path else model / "points3D.ply"
        if (
            not ply_path.is_file()
            and (self.source / "input.ply").is_file()
            and ply_path == model / "points3D.ply"
        ):
            ply_path = self.source / "input.ply"
        total = sum(f.linked for f in self.frames.values())
        self.ply = (
            PlyIndex(ply_path, bool(len(self.points.ids) and (total or len(self.points.reverse))))
            if ply_path.is_file()
            else None
        )
        rev_names, rev_counts = np.unique(self.points.reverse["image"], return_counts=True)
        counts = dict(zip(rev_names.tolist(), rev_counts.tolist()))
        listing = [
            dict(id=f.id, name=f.name, forward=f.linked, reverse=counts.get(f.id, 0))
            for f in sorted(self.frames.values(), key=lambda f: f.name)
        ]
        self.summary = dict(
            frames=listing,
            with_forward=sum(f.linked > 0 for f in self.frames.values()),
            with_reverse=sum(i in self.frames for i in counts),
            forward=total,
            reverse=len(self.points.reverse),
            orphan_reverse=sum(c for i, c in counts.items() if i not in self.frames),
            point_file=self.points.present,
            points=len(self.points.ids),
            ply=str(ply_path) if self.ply else None,
            ply_rows=len(self.ply.vertices) if self.ply else 0,
            model=str(model),
            initial=max(listing, key=lambda v: max(v["forward"], v["reverse"]))["id"],
        )
        print(
            f'{len(listing):,} images | {total:,} stored image→point links across {self.summary["with_forward"]:,} images | '
            f'{len(self.points.reverse):,} reverse track entries across {self.summary["with_reverse"]:,} registered images | '
            f'{len(self.points.ids):,} binary points | {self.summary["ply_rows"]:,} PLY rows',
            flush=True,
        )

    def frame(self, iid):
        frame = self.frames[iid]
        cam = self.cameras[frame.camera]
        records = np.frombuffer(self.buf, FEATURE, frame.count, frame.offset)
        features = np.flatnonzero(records["id"] >= 0)
        reverse = self.points.for_image(iid)
        forward = pair_array(features, records["id"][features])
        back = pair_array(reverse["feature"], self.points.ids[reverse["row"]])
        pairs, index = np.unique(np.concatenate([forward, back]), return_inverse=True)
        index = index.ravel()
        flags = np.zeros(len(pairs), dtype=np.uint8)
        flags[index[: len(forward)]] |= 1
        flags[index[len(forward) :]] |= 2
        ids, per_pair = np.unique(pairs["id"], return_inverse=True)
        per_pair = per_pair.ravel()
        rows = self.points.rows(ids)
        xyz = np.full((len(ids), 3), np.nan)
        resolved = rows >= 0
        xyz[resolved] = self.points.xyz[rows[resolved]]
        ply = self.ply.match(xyz) if self.ply else np.full(len(ids), -1)
        can_project = cam.model in ("PINHOLE", "SIMPLE_PINHOLE")
        uv = project(xyz, frame, cam)[0] if can_project else np.full((len(ids), 2), np.nan)
        both = ((flags & 1) != 0) & np.isfinite(uv[per_pair]).all(axis=1)
        residual = np.linalg.norm(records["xy"][pairs["feature"][both]] - uv[per_pair[both]], axis=1)
        residual = residual[np.isfinite(residual)]
        samples = []
        draw = np.linspace(0, len(pairs) - 1, min(self.limit, len(pairs)), dtype=int)
        for j in draw:
            i = per_pair[j]
            feat = int(pairs["feature"][j])
            # A reverse entry can refer to a missing or contradictory feature.
            xy = finite_xy(records["xy"][feat]) if flags[j] & 1 else None
            samples.append(
                [
                    str(ids[i]),
                    feat,
                    int(flags[j]),
                    xy,
                    finite_xy(uv[i]),
                    int(ply[i]),
                    finite_xy(xyz[i]) if resolved[i] else None,
                ]
            )
        return dict(
            id=iid,
            name=frame.name,
            width=cam.width,
            height=cam.height,
            features=frame.count,
            camera_model=cam.model,
            can_project=can_project,
            reciprocal_ply=len(np.unique(per_pair[(flags == 3) & (ply[per_pair] >= 0)])),
            forward=len(forward),
            reverse=len(back),
            reciprocal=int((flags == 3).sum()),
            forward_only=int((flags == 1).sum()),
            reverse_only=int((flags == 2).sum()),
            pairs=len(pairs),
            distinct=len(ids),
            resolved=int(resolved.sum()),
            matched=int((ply >= 0).sum()),
            ambiguous=int((ply == -2).sum()),
            missing_xyz=int((~resolved).sum()),
            median_error_px=float(np.median(residual)) if len(residual) else None,
            samples=samples,
        )

    def context(self, iid):
        if self.ply is None or not len(self.ply.vertices):
            return []
        f = self.frames[iid]
        cam = self.cameras[f.camera]
        if cam.model not in ("PINHOLE", "SIMPLE_PINHOLE"):
            return []
        step = max(1, math.ceil(len(self.ply.vertices) / 250000))
        uv, z = project(self.ply.coords(slice(None, None, step)), f, cam)
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
        uv, z = uv[ok], z[ok]
        # A depth buffer on a subsample is geometric context only.
        cells = (uv[:, 1] // 5) * math.ceil(cam.width / 5) + uv[:, 0] // 5
        order = np.lexsort((z, cells))
        _, first = np.unique(cells[order], return_index=True)
        front = uv[order[first]]
        return (
            front[np.linspace(0, len(front) - 1, min(self.limit, len(front)), dtype=int)].round(2).tolist()
        )

    def photo(self, iid):
        path = self.source / self.images / self.frames[iid].name
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            data = BytesIO()
            rgb.save(data, format="JPEG", quality=93)
            return data.getvalue()


def handler(inspector):
    page = Path(__file__).with_suffix(".html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path).path
            try:
                if url == "/":
                    body, mime = page, "text/html; charset=utf-8"
                elif url == "/summary":
                    body, mime = json.dumps(inspector.summary).encode(), "application/json"
                else:
                    route, key = url.strip("/").split("/")
                    iid = int(key)
                    if iid not in inspector.frames or route not in ("frame", "photo", "context"):
                        raise KeyError(key)
                    if route == "photo":
                        body, mime = inspector.photo(iid), "image/jpeg"
                    else:
                        value = inspector.frame(iid) if route == "frame" else inspector.context(iid)
                        body, mime = (
                            json.dumps(value, allow_nan=False, separators=(",", ":")).encode(),
                            "application/json",
                        )
            except (KeyError, FileNotFoundError, ValueError, OSError) as exc:
                self.send_error(HTTPStatus.NOT_FOUND, str(exc))
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path)
    p.add_argument("--images", default="images")
    p.add_argument("--ply", type=Path)
    p.add_argument("--image", help="Initial frame name, e.g. 09882.jpg")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8094)
    p.add_argument(
        "--max-draw", type=int, default=5000, help="Display cap per layer. Counts use all observations."
    )
    args = p.parse_args()
    if args.max_draw < 1:
        p.error("--max-draw must be positive")
    suite = (
        Path(os.environ.get("ORANGE4F_OUTPUT_ROOT", "/data/3dgs_outputs/orange4f/3dgs_finetuning"))
        / "stride3_noaa_v1/suite.json"
    )
    source = args.source or (
        Path(json.loads(suite.read_text()).get("source", "/workspace/colmap_scenes/orange4f"))
        if suite.is_file()
        else Path("/workspace/colmap_scenes/orange4f")
    )
    inspector = Inspector(source.expanduser().resolve(), args.ply, args.images, args.max_draw)
    if args.image:
        candidates = [f.id for f in inspector.frames.values() if f.name == args.image]
        if not candidates:
            p.error(f"Unknown image {args.image}")
        inspector.summary["initial"] = candidates[0]
    with ThreadingHTTPServer((args.host, args.port), handler(inspector)) as server:
        if args.host == "0.0.0.0":
            print(f"Open http://<training-server-IP>:{server.server_port}/ on your laptop.", flush=True)
        else:
            print(f"Listening at http://{args.host}:{server.server_port}/", flush=True)
        try:
            for ip in sorted(
                {a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}
            ):
                if args.host == "0.0.0.0" and not ip.startswith("127."):
                    print(f"  http://{ip}:{server.server_port}/", flush=True)
        except OSError:
            pass
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
