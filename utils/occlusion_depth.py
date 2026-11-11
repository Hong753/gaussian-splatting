"""Fixed LiDAR surfel depth, cached as conservative 16-pixel tile maxima.

The cloud supplies opaque local surfaces. Unknown pixels stay +inf. No hole filling,
learned opacity, or gradients through the depth test are used.
"""
from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np
from utils.lidar_occlusion import file_digest, load_geometry, surfel_triangles

TILE = 16
CACHE_VERSION = 2


def camera_signature(camera, edge_pixels):
    view = camera.world_view_transform.detach().cpu().numpy().astype(np.float32)
    payload = dict(version=CACHE_VERSION, tile=TILE, edge_pixels=edge_pixels,
        width=int(camera.image_width), height=int(camera.image_height),
        fovx=float(camera.FoVx), fovy=float(camera.FoVy),
        near=float(camera.znear), far=float(camera.zfar), view=view.tolist())
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def clip_vertices(vertices, camera):
    """Graphdeco camera coordinates (+z forward, +y down) to GL clip coordinates."""
    import torch
    view = camera.world_view_transform.to(device=vertices.device, dtype=vertices.dtype)
    camera_xyz = vertices @ view[:3, :3] + view[3, :3]
    x, y, z = camera_xyz.unbind(-1)
    n, f = camera.znear, camera.zfar
    # Flip camera y into GL y-up; flip the raster image back after rasterization.
    return torch.stack((x / math.tan(camera.FoVx / 2), -y / math.tan(camera.FoVy / 2),
                        (f+n)/(f-n)*z - 2*f*n/(f-n), z), dim=-1)


class LidarDepthRenderer:
    def __init__(self, geometry, device):
        import torch
        try:
            import nvdiffrast.torch as dr
        except ImportError as exc:
            raise ImportError("Install the standalone depth rasterizer in the native environment: "
                              "python -m pip install --no-build-isolation -r requirements-occlusion.txt") from exc
        arrays, _ = load_geometry(geometry)
        self.dr, self.ctx = dr, dr.RasterizeCudaContext(device=device)
        self.centers, self.normals, self.radii = [torch.as_tensor(arrays[k], device=device) for k in ('centers', 'normals', 'radii')]

    def __call__(self, camera):
        import torch
        with torch.no_grad():
            h, w = camera.image_height, camera.image_width
            depth = torch.full((h, w), float("inf"), device=self.centers.device)
            # Bounded scratch: never expand every surfel into triangles at once.
            for start in range(0, len(self.centers), 100_000):
                sl = slice(start, start+100_000)
                vertices, faces = surfel_triangles(self.centers[sl], self.normals[sl], self.radii[sl])
                clip = clip_vertices(vertices, camera)[None].contiguous()
                rast, _ = self.dr.rasterize(self.ctx, clip, faces, resolution=(h, w), grad_db=False)
                hit = rast[0, :, :, 3] > 0
                ndc = rast[0, :, :, 2].clamp(-1, 1)
                n, f = camera.znear, camera.zfar
                linear = 2*n*f / (f+n - ndc*(f-n))
                depth = torch.minimum(depth, torch.where(hit, linear, torch.full_like(linear, float("inf"))))
            return depth.flip(0).contiguous()


def pixel_bounds(depth, edge_pixels=1):
    import torch
    import torch.nn.functional as F
    depth = torch.where(torch.isfinite(depth) & (depth > 0), depth, torch.full_like(depth, float("inf")))
    depth = depth[None, None]
    if edge_pixels:
        depth = F.max_pool2d(F.pad(depth, (edge_pixels,)*4, value=float("inf")), 2*edge_pixels+1, stride=1)
    return depth[0, 0]


def tile_maxima(depth, edge_pixels=1):
    import torch.nn.functional as F
    return F.max_pool2d(pixel_bounds(depth, edge_pixels)[None, None], TILE,
                        stride=TILE, ceil_mode=True)[0, 0]


class LidarDepthCache:
    def __init__(self, geometry, directory="", edge_pixels=1, memory_views=4):
        self.geometry = Path(geometry).expanduser().resolve()
        self.geometry_sha256 = file_digest(self.geometry)
        self.directory = (Path(directory).expanduser().resolve() if directory else
                          self.geometry.parent / "occlusion_depth_cache") / self.geometry_sha256
        self.edge_pixels, self.memory_views = edge_pixels, memory_views
        self.memory, self.renderer = OrderedDict(), None
        self.surface_memory = OrderedDict()
        self.pixel_memory = OrderedDict()

    def pixel_depth(self, camera, device):
        """Full-resolution conservative first-hit bounds, without tile reduction."""
        key = (camera_signature(camera, self.edge_pixels), str(device))
        if key not in self.pixel_memory:
            self.pixel_memory[key] = pixel_bounds(self.tile_depth(self.raw_depth(camera, device)), self.edge_pixels)
            while len(self.pixel_memory) > self.memory_views:
                self.pixel_memory.popitem(last=False)
        self.pixel_memory.move_to_end(key)
        return self.pixel_memory[key]

    def make_renderer(self, device):
        return LidarDepthRenderer(self.geometry, device)

    def tile_depth(self, depth):
        return depth

    def surface(self, camera, device, radius_pixels=3, max_radius=.1):
        """Uninflated first-hit LiDAR depth for local tangent neighborhoods."""
        from utils.view_tangent import camera_intrinsics, surface_frame
        signature = camera_signature(camera, self.edge_pixels)
        key = (signature, str(device), radius_pixels, max_radius)
        if key not in self.surface_memory:
            import torch
            depth = self.raw_depth(camera, device)
            self.surface_memory[key] = surface_frame(depth, torch.isfinite(depth),
                camera_intrinsics(camera, device), radius_pixels, max_radius)
            while len(self.surface_memory) > self.memory_views:
                self.surface_memory.popitem(last=False)
        self.surface_memory.move_to_end(key)
        return self.surface_memory[key]

    def raw_depth(self, camera, device):
        import torch
        path = self.directory / (camera_signature(camera, self.edge_pixels) + '.depth.npz')
        if path.is_file():
            with np.load(path, allow_pickle=False) as data:
                array = data['depth']
            if (array.shape != (camera.image_height, camera.image_width) or
                    np.isnan(array).any() or np.any(array <= 0)):
                raise ValueError(f'Invalid raw LiDAR surface depth: {path}')
            return torch.as_tensor(array, device=device, dtype=torch.float32)
        if self.renderer is None:
            self.renderer = self.make_renderer(device)
        depth = self.renderer(camera)
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.stem + f'.{os.getpid()}.{uuid.uuid4().hex}.tmp.npz')
        try:
            np.savez_compressed(temporary, depth=depth.cpu().numpy())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return depth

    def diagnostic_frame(self, camera, device):
        from utils.view_tangent import camera_intrinsics
        depth = self.raw_depth(camera, device).cpu().numpy()
        return dict(depth=depth, valid=np.isfinite(depth) & (depth > 0),
                    intrinsics=camera_intrinsics(camera, 'cpu').numpy(), report={})

    def tiles(self, camera, device):
        import torch
        # Camera objects are immutable in the training/view pipelines.
        signature = camera_signature(camera, self.edge_pixels)
        key = (signature, str(device))
        if key in self.memory:
            self.memory.move_to_end(key)
            return self.memory[key]
        path = self.directory / (signature + ".npy")
        shape = ((camera.image_height + TILE-1)//TILE, (camera.image_width + TILE-1)//TILE)
        if path.is_file():
            array = np.load(path, allow_pickle=False)
            if array.shape != shape or np.isnan(array).any() or np.any(array <= 0):
                raise ValueError(f"Invalid LiDAR depth cache: {path}")
            tiles = torch.as_tensor(array, dtype=torch.float32, device=device)
        else:
            tiles = tile_maxima(self.tile_depth(self.raw_depth(camera, device)), self.edge_pixels)
            self.directory.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.stem + f".{os.getpid()}.{uuid.uuid4().hex}.tmp.npy")
            try:
                np.save(temp, tiles.cpu().numpy())
                os.replace(temp, path)
            finally:
                temp.unlink(missing_ok=True)
        self.memory[key] = tiles
        while len(self.memory) > self.memory_views:
            self.memory.popitem(last=False)
        return tiles


def record_camera(record, resolution=-1, device="cuda"):
    """Construct the exact centered camera used by loadCam without RGB allocation."""
    import torch
    from PIL import Image
    from types import SimpleNamespace
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix
    with Image.open(record["image_path"]) as image:
        ow, oh = image.size
    if resolution in (1, 2, 4, 8):
        w, h = round(ow / resolution), round(oh / resolution)
    else:
        factor = max(1., ow / 1600) if resolution == -1 else ow / resolution
        w, h = int(ow / factor), int(oh / factor)
    view = torch.tensor(getWorld2View2(np.array(record["R"]), np.array(record["T"])), device=device).T
    projection = getProjectionMatrix(.01, 100., record["fov_x"], record["fov_y"]).to(device).T
    return SimpleNamespace(image_width=w, image_height=h, image_name=record["name"],
        FoVx=record["fov_x"], FoVy=record["fov_y"], znear=.01, zfar=100.,
        world_view_transform=view, full_proj_transform=view @ projection)
