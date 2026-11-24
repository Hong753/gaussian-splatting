"""Assign training photographs to mesh faces once; render the saved mesh at a supplied pose."""
import ctypes
from functools import lru_cache
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from scripts.orange4f_photo_detail import Camera, sample


@lru_cache(maxsize=1)
def raster_kernel():
    source = Path(__file__).with_name("world_raster.cpp")
    # Compile locally; the portable scene contains geometry and texture data only.
    with TemporaryDirectory() as tmp:
        lib = Path(tmp)/"raster.so"
        subprocess.run(["c++", "-O3", "-std=c++17", "-fPIC", "-shared", str(source), "-o", str(lib)], check=True)
        kernel = ctypes.CDLL(str(lib)).render_faces
    array = np.ctypeslib.ndpointer
    floats = array(dtype=np.float32, flags="C_CONTIGUOUS")
    ints = array(dtype=np.int32, flags="C_CONTIGUOUS")
    kernel.argtypes = [floats, ints, *([ctypes.c_int]*3), *([ctypes.c_double]*4), ints, floats, floats]
    kernel.restype = None
    return kernel


def raster(vertices, faces, camera):
    xyz = np.ascontiguousarray((vertices - camera.center) @ camera.rotation.T, dtype=np.float32)
    shape = (camera.height, camera.width)
    ids = np.full(shape, -1, np.int32)
    depth = np.full(shape, np.inf, np.float32)
    bary = np.empty((*shape, 2), np.float32)
    k = camera.k
    raster_kernel()(xyz, np.ascontiguousarray(faces, dtype=np.int32), len(faces), camera.width,
                    camera.height, k[0, 0], k[1, 1], k[0, 2], k[1, 2], ids, depth, bary)
    return ids, depth, bary


def project(points, cam):
    xyz = (points-cam.center) @ cam.rotation.T
    z = xyz[..., 2]
    return xyz[..., :2]/np.maximum(z[..., None], .01)*cam.k.diagonal()[:2]+cam.k[:2, 2], z


def candidates(vertices, faces, centers, normals, tree, cam, depth, valid, k, voxel, far):
    h, w = depth.shape
    prior = Camera(w, h, k, cam.rotation, cam.center)
    max_xy = np.maximum(np.abs(k[:2, 2]), np.abs(np.array([w-1, h-1])-k[:2, 2])) / k.diagonal()[:2]
    idx = np.asarray(tree.query_ball_point(cam.center, far*np.sqrt(1+max_xy @ max_xy)+4*voxel), int)
    uv, z = project(centers[idx], prior)
    idx = idx[(z > .05) & (z < far) & (uv[:, 0] >= -2) & (uv[:, 0] < w+2)
              & (uv[:, 1] >= -2) & (uv[:, 1] < h+2)]
    if not len(idx):
        return idx, np.empty(0)
    local, inverse = np.unique(faces[idx].ravel(), return_inverse=True)
    _, front, _ = raster(vertices[local], inverse.reshape(-1, 3), prior)
    points = np.concatenate([vertices[faces[idx]], centers[idx, None]], axis=1)
    uv, z = project(points, prior)
    u, v = np.rint(uv).astype(int).transpose(2, 0, 1)
    inside = (z > .05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v = u.clip(0, w-1), v.clip(0, h-1)
    tol = np.maximum(1.5*voxel, .02*z)
    visible = (inside & valid[v, u] & (np.abs(depth[v, u]-z) <= tol)
               & np.isfinite(front[v, u]) & (z <= front[v, u]+tol)).all(axis=1)
    rgb_uv, _ = project(points, cam)
    visible &= ((rgb_uv[..., 0] >= 1) & (rgb_uv[..., 0] <= cam.width-2)
                & (rgb_uv[..., 1] >= 1) & (rgb_uv[..., 1] <= cam.height-2)).all(axis=1)
    delta = cam.center-centers[idx]
    incidence = np.abs(np.sum(normals[idx]*delta, axis=1))/np.maximum(np.linalg.norm(delta, axis=1), 1e-7)
    visible &= incidence > .12
    # Projected area (texture samples per unit surface area).
    scores = cam.k[0, 0]*cam.k[1, 1]*incidence/np.maximum(z[:, -1]**2, 1e-8)
    return idx[visible], scores[visible]


def bake(directory, vertices, faces, inputs, voxel, far):
    directory = Path(directory)
    textures = directory/"textures"
    textures.mkdir()
    points = vertices[faces]
    centers = points.mean(axis=1)
    normal = np.cross(points[:, 1]-points[:, 0], points[:, 2]-points[:, 0])
    area = np.linalg.norm(normal, axis=1)
    normal /= np.maximum(area[:, None], 1e-12)
    del points
    tree = cKDTree(centers)
    axis = np.argmax(np.abs(normal), axis=1)
    side = axis*2 + (np.take_along_axis(normal, axis[:, None], axis=1)[:, 0] < 0)
    cells = np.column_stack([np.floor(centers/.5).astype(int), side])
    _, patch = np.unique(cells, axis=0, return_inverse=True)
    patch = patch.ravel()
    count = int(patch.max())+1
    best_patch = np.zeros(count)
    patch_source = np.full(count, -1, np.int32)
    best_face = np.zeros(len(faces))
    labels = np.full(len(faces), -1, np.int32)
    with TemporaryDirectory(dir=directory) as tmp:
        for j, (cam, depth_file, _) in enumerate(inputs):
            with np.load(depth_file, allow_pickle=False) as data:
                idx, scores = candidates(vertices, faces, centers, normal, tree, cam, data["depth"],
                                         data["valid"], data["intrinsic"], voxel, far)
            np.save(Path(tmp)/f"{j}.npy", idx.astype(np.int32))
            strength = np.bincount(patch[idx], weights=scores*area[idx], minlength=count)
            better = strength > best_patch
            best_patch[better], patch_source[better] = strength[better], j
            improved = scores > best_face[idx]
            best_face[idx[improved]], labels[idx[improved]] = scores[improved], j
            if (j+1) % 50 == 0:
                print(f"Texture visibility {j+1}/{len(inputs)}", flush=True)
        for j in range(len(inputs)):
            idx = np.load(Path(tmp)/f"{j}.npy")
            labels[idx[patch_source[patch[idx]] == j]] = j

    # Keep just the used image rectangles. UVs and page IDs are now immutable.
    uv = np.zeros((len(faces), 3, 2), np.float32)
    pages = {}
    for j in np.unique(labels[labels >= 0]):
        cam, _, image = inputs[j]
        idx = np.flatnonzero(labels == j)
        coords, _ = project(vertices[faces[idx]], cam)
        left, top = np.floor(coords.min(axis=(0, 1))).astype(int)-1
        right, bottom = np.ceil(coords.max(axis=(0, 1))).astype(int)+2
        bounds = (max(0, left), max(0, top), min(cam.width, right), min(cam.height, bottom))
        with Image.open(image) as rgb:
            if rgb.size != (cam.width, cam.height):
                raise ValueError(f"Texture calibration mismatch: {image}")
            tile = rgb.convert("RGB").crop(bounds)
        tile.save(textures/f"{j}.png", compress_level=1)
        uv[idx] = coords-np.array(bounds[:2])
        pages[int(j)] = tile.size
    if not pages:
        raise ValueError("No fused faces passed training-view depth and visibility checks")
    np.savez(directory/"surface.npz", vertices=vertices, faces=faces, page=labels, uv=uv)
    write_obj(directory, vertices, faces, labels, uv, pages)
    return dict(texture_pages=len(pages), textured_area_fraction=float(area[labels >= 0].sum()/area.sum()),
                faces=len(faces), vertices=len(vertices))


def write_obj(directory, vertices, faces, page, uv, sizes):
    with (directory/"surface.mtl").open("w") as mtl:
        mtl.write("newmtl uncovered\nKd 0.5 0.5 0.5\n")
        for j in sizes:
            mtl.write(f"newmtl page_{j}\nKd 1 1 1\nmap_Kd textures/{j}.png\n")
    with (directory/"surface.obj").open("w") as obj:
        obj.write("mtllib surface.mtl\n")
        np.savetxt(obj, vertices, fmt="v %.8g %.8g %.8g")
        tex_index = 1
        for j in np.unique(page):
            selected = np.flatnonzero(page == j)
            if j < 0:
                obj.write("usemtl uncovered\n")
                np.savetxt(obj, faces[selected]+1, fmt="f %d %d %d")
                continue
            obj.write(f"usemtl page_{j}\n")
            for block in range(0, len(selected), 65536):
                idx = selected[block:block+65536]
                coords = (uv[idx].reshape(-1, 2)+.5)/sizes[j]
                coords[:, 1] = 1-coords[:, 1]
                np.savetxt(obj, coords, fmt="vt %.9g %.9g")
                entries = np.empty((len(idx), 6), np.int64)
                entries[:, ::2] = faces[idx]+1
                entries[:, 1::2] = np.arange(tex_index, tex_index+len(idx)*3).reshape(-1, 3)
                np.savetxt(obj, entries, fmt="f %d/%d %d/%d %d/%d")
                tex_index += 3*len(idx)


class WorldMesh:
    def __init__(self, directory):
        self.directory = Path(directory)
        with np.load(self.directory/"surface.npz", allow_pickle=False) as data:
            self.vertices, self.faces, self.page, self.uv = (data[k] for k in ("vertices", "faces", "page", "uv"))
        if (self.vertices.ndim != 2 or self.vertices.shape[1] != 3 or not np.isfinite(self.vertices).all()
                or self.faces.ndim != 2 or self.faces.shape[1] != 3 or not self.faces.size
                or self.faces.dtype != np.int32 or self.faces.min() < 0 or self.faces.max() >= len(self.vertices)
                or self.page.shape != (len(self.faces),) or self.uv.shape != (len(self.faces), 3, 2)
                or not np.isfinite(self.uv).all()):
            raise ValueError("Invalid textured triangle mesh in surface.npz")

    @lru_cache(maxsize=16)
    def texture(self, page):
        with Image.open(self.directory/"textures"/f"{page}.png") as image:
            return np.array(image)

    def render(self, cam):
        ids, _, bary = raster(self.vertices, self.faces, cam)
        hit = ids >= 0
        covered = hit & (self.page[ids.clip(min=0)] >= 0)
        rgb = np.zeros((cam.height, cam.width, 3), np.uint8)
        rgb[hit] = 128
        y, x = np.where(covered)
        face = ids[y, x]
        weights = bary[y, x]
        origin = self.uv[face, 0]
        coords = (origin + weights[:, :1]*(self.uv[face, 1]-origin)
                  + weights[:, 1:]*(self.uv[face, 2]-origin))
        materials = self.page[face]
        order = np.argsort(materials, kind="stable")
        for idx in np.split(order, np.flatnonzero(np.diff(materials[order]))+1):
            if not len(idx):
                continue
            tex = self.texture(int(materials[idx[0]]))
            u, v = coords[idx].T
            for channel in range(3):
                rgb[y[idx], x[idx], channel] = sample(tex[..., channel], v, u)
        return rgb, hit, covered
