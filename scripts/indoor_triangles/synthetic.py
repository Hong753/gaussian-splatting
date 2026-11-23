"""Evaluator-only procedural room and independent Möller–Trumbore tracer."""

from pathlib import Path

import numpy as np
from PIL import Image

from .render import Camera, OFFSETS, pixels

BG = np.array([.07, .08, .10])
SPLIT = {"fit": [-.70, -.34, 0., .32, .64], "edit": [-.53, .49], "final": [-.97, .92]}


def camera(x, width=96, height=72):
    return Camera(np.array([x, 0., 0.]), np.eye(3),
                  np.array([.88*width, .88*width, (width-1)/2, (height-1)/2]), width, height)


def room():
    faces, kinds, verts = [], [], []
    def quad(a, b, c, d, kind):
        start = len(verts)
        verts.extend((a, b, c, d))
        faces.extend(((start, start+1, start+2), (start, start+2, start+3)))
        kinds.extend((kind, kind))

    def wall(x, xx, y, yy, z, kind):
        quad((x,y,z), (xx,y,z), (xx,yy,z), (x,yy,z), kind)

    wall(-4, -.42, -2.6, 1.05, 3.3, 1)
    wall(.48, 4, -2.6, 1.05, 3.3, 1)
    wall(-.42, .48, -2.6, -.91, 3.3, 1)
    wall(-2.7, 2.7, -2.6, 1.07, 5.55, 2)
    wall(-.49, -.42, -.98, 1.05, 3.22, 3)
    wall(.48, .55, -.98, 1.05, 3.22, 3)
    wall(-.42, .48, -.98, -.91, 3.22, 3)
    for x, xx in zip([-4,-1.0,.35,1.1], [-1.0,.35,1.1,4]):
        for (z,y), (zz,yy) in zip([(0.4,1.05),(2.2,.96),(3.3,1.05),(4.25,1.16)],
                                   [(2.2,.96),(3.3,1.05),(4.25,1.16),(5.56,1.07)]):
            lift = .055 * np.sin(x*1.3) * max(0, 1-abs(z-2.4)/.9)
            lift_xx = .055 * np.sin(xx*1.3) * max(0, 1-abs(z-2.4)/.9)
            lift_zz = .055 * np.sin(x*1.3) * max(0, 1-abs(zz-2.4)/.9)
            lift_xxzz = .055 * np.sin(xx*1.3) * max(0, 1-abs(zz-2.4)/.9)
            quad((x,y+lift,z),(xx,y+lift_xx,z),(xx,yy+lift_xxzz,zz),(x,yy+lift_zz,zz), 0)
    wall(-1.62,-1.04,-.54,.17,2.18,4)
    return np.array(verts), np.array(faces), np.array(kinds)


def material(points, kind):
    x, y, z = np.moveaxis(points, -1, 0)
    v = np.where(kind == 0, z, y)
    signal = (.60*np.sin(16.9*x+11.3*v) + .33*np.cos(30.1*x-16.2*v)
              + .42*np.sin(25.3*x*v+8.3*v) + .22*np.cos(43.3*x+51.2*v))
    base = np.array([[.49,.43,.35],[.59,.54,.48],[.34,.52,.64],[.27,.18,.11],[.13,.66,.54]])[kind]
    stripe = (kind == 0) & (x > -.85) & (x < -.67)
    plain = ((kind == 0) & (x > .82) & (z > 1.7)) | ((kind == 1) & (x > 1.6) & (y < -.5))
    base = np.where(stripe[..., None], [.70,.20,.16], base)
    detail = np.where(plain | (kind == 4), 0., .15*signal)
    return np.clip(base + detail[..., None], 0, 1)


def scalar_hit(center, direction, tri, near=1e-8):
    a, b, c = tri
    e, g = b-a, c-a
    p = np.cross(direction, g)
    det = np.dot(e, p)
    if abs(det) < 1e-12:
        return None
    rel = center-a
    u = np.dot(rel, p) / det
    q = np.cross(rel, e)
    v = np.dot(direction, q) / det
    t = np.dot(g, q) / det
    if t > near and min(u, v, 1-u-v) >= -1e-9:
        return t, np.array([1-u-v, u, v])
    return None


def oracle(vertices, faces, kinds, cam, coords):
    """Independent intersection path used only to synthesize RGB and score."""
    rays = cam.rays(coords)
    tri = vertices[faces]
    a, b, c = np.moveaxis(tri, 1, 0)
    e, g = b-a, c-a
    p = np.cross(rays[:, None], g)
    det = (e*p).sum(-1)
    usable = np.abs(det) > 1e-12
    inv = np.divide(1, det, out=np.zeros_like(det), where=usable)
    rel = cam.center-a
    u = (rel*p).sum(-1)*inv
    q = np.cross(rel, e)
    vv = (rays[:, None]*q).sum(-1)*inv
    depth = (g*q).sum(-1)*inv
    valid = usable & (u >= -1e-9) & (vv >= -1e-9) & (u+vv <= 1.+1e-9) & (depth > 0)
    front = np.where(valid & (kinds != 4), depth, np.inf)
    index = front.argmin(1)
    r = np.arange(len(rays))
    distance = front[r, index]
    point = cam.center + np.where(np.isfinite(distance), distance, 0)[:, None]*rays
    visible_kind = np.where(np.isfinite(distance), kinds[index], -1)
    rgb = np.where(np.isfinite(distance[:, None]), material(point, kinds[index]), BG)
    pane_depth = np.where(valid & (kinds == 4), depth, np.inf).min(1)
    pane = pane_depth < distance
    rgb[pane] = rgb[pane]*.62 + np.array([.13,.66,.54])*.38
    raw_n = np.cross(e, g)
    n = raw_n[index] / np.linalg.norm(raw_n[index], axis=1, keepdims=True)
    n *= np.where((n*rays).sum(1) > 0, -1, 1)[:, None]
    return rgb, distance, visible_kind, n


def generate(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    vertices, faces, kinds = room()
    manifest = {}
    for name, positions in SPLIT.items():
        cams = [camera(x) for x in positions]
        images = []
        for i, cam in enumerate(cams):
            coords = pixels(cam)
            samples = (coords[..., None, :] + OFFSETS).reshape(-1, 2)
            colors = np.concatenate([oracle(vertices,faces,kinds,cam,block)[0]
                                     for block in np.array_split(samples, 36)])
            image = colors.reshape(cam.height, cam.width, len(OFFSETS), 3).mean(axis=2)
            image = (image*255).round().astype(np.uint8)
            images.append(image)
            Image.fromarray(image).save(folder / f"{name}_{i:02}.png")
        np.savez_compressed(folder / f"{name}.npz", rgb=images,
                            centers=[c.center for c in cams], rotations=[c.rotation for c in cams],
                            intrinsics=[c.intrinsics for c in cams])
        manifest[name] = [{"id": f"{name}_{i:02}", "center_m": c.center.tolist()}
                          for i, c in enumerate(cams)]
    np.savez_compressed(folder / "reference.npz", vertices=vertices, faces=faces, kinds=kinds)
    return manifest
