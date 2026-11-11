"""Fixed scene meshes and per-camera LiDAR visibility, independent of Gaussian opacity."""
import json
from pathlib import Path

import numpy as np

from utils.lidar_occlusion import file_digest
from utils.occlusion_depth import LidarDepthCache, clip_vertices

FORMAT = 'scene_occlusion_mesh_v1'
POINT_CLASSES = ('visible', 'occluded', 'unknown', 'in_front')


def load_scene(path, point_cloud=None, source=None):
    path = Path(path).expanduser().resolve()
    report = json.loads(path.read_text())
    if report.get('format') != FORMAT or not report.get('complete'):
        raise ValueError('Incomplete or incompatible scene occlusion mesh')
    if not np.isfinite(report.get('relative_margin', np.nan)) or report['relative_margin'] < 0:
        raise ValueError('Invalid scene depth margin')
    mesh = path.parent / report['mesh']
    if not mesh.is_file() or file_digest(mesh) != report['mesh_sha256']:
        raise ValueError('Scene mesh is missing or changed; rebuild its visibility cache')
    if point_cloud and file_digest(point_cloud) != report['point_cloud_sha256']:
        raise ValueError('Scene mesh and initialization LiDAR cloud differ')
    if source and Path(source).resolve() != Path(report['source']).resolve():
        raise ValueError('Scene mesh and camera dataset differ')
    return report, mesh


def mesh_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        vertices = np.array(data['vertices'], dtype=np.float32)
        faces = np.array(data['faces'], dtype=np.int32)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or
            faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or
            not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('Invalid scene mesh')
    return vertices, faces


class TriangleRaycaster:
    """CPU BVH queries. Unnormalized directions make the hit parameter camera z."""
    def __init__(self, vertices, faces, threads=0):
        import open3d as o3d
        self.o3d = o3d
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=threads)
        self.scene.add_triangles(o3d.core.Tensor(np.ascontiguousarray(vertices, dtype=np.float32)),
            o3d.core.Tensor(np.ascontiguousarray(faces, dtype=np.uint32)))

    def cast(self, origins, directions):
        rays = np.concatenate((np.broadcast_to(origins, directions.shape), directions), axis=-1)
        return self.scene.cast_rays(self.o3d.core.Tensor(np.ascontiguousarray(rays, dtype=np.float32)))['t_hit'].numpy()

    def depth(self, intrinsic, extrinsic, width, height, chunk=131072):
        rotation, translation = extrinsic[:3, :3], extrinsic[:3, 3]
        origin = -rotation.T @ translation
        depth = np.empty(width*height, np.float32)
        for start in range(0, len(depth), chunk):
            ids = np.arange(start, min(start+chunk, len(depth)))
            directions = np.stack(((ids % width-intrinsic[0, 2])/intrinsic[0, 0],
                (ids//width-intrinsic[1, 2])/intrinsic[1, 1], np.ones(len(ids))), axis=1)
            depth[start:start+len(ids)] = self.cast(origin, directions @ rotation)
        return depth.reshape(height, width)


def classify_lidar(points, intrinsic, extrinsic, width, height, raycaster,
                   absolute_margin=.05, relative_margin=.05, device='cpu', chunk=262144):
    """Classify original PLY rows with an exact ray through each in-frustum point.

    Unknown means the mesh has no intersection. In-front means disagreement
    with the first surface, not a verified visible surface sample. Out-of-view
    rows are omitted; stable original row IDs are retained, including after NaNs.
    """
    import torch
    if (not np.isfinite([absolute_margin, relative_margin]).all() or
            min(absolute_margin, relative_margin) < 0 or chunk < 1):
        raise ValueError('Invalid LiDAR visibility margins/chunk')
    e = torch.as_tensor(extrinsic, dtype=torch.float32, device=device)
    origin = -np.asarray(extrinsic[:3, :3]).T @ extrinsic[:3, 3]
    result = {key: [] for key in POINT_CLASSES}
    id_type = np.uint32 if len(points) <= np.iinfo(np.uint32).max else np.uint64
    for start in range(0, len(points), chunk):
        p = torch.as_tensor(points[start:start+chunk], dtype=torch.float32, device=device)
        camera = p @ e[:3, :3].T + e[:3, 3]
        z = camera[:, 2]
        u = intrinsic[0, 0]*camera[:, 0]/z + intrinsic[0, 2]
        v = intrinsic[1, 1]*camera[:, 1]/z + intrinsic[1, 2]
        inside = (torch.isfinite(camera).all(1) & (z > .2) &
            (u >= -.5) & (u < width-.5) & (v >= -.5) & (v < height-.5))
        ids = inside.nonzero().flatten()
        if not len(ids):
            continue
        xyz = p[ids].cpu().numpy()
        z = z[ids].cpu().numpy()
        hit = raycaster.cast(origin, (xyz-origin)/z[:, None])
        known = np.isfinite(hit) & (hit > .2)
        tolerance = absolute_margin + relative_margin*np.where(known, hit, 0)
        delta = np.zeros_like(z)
        delta[known] = z[known]-hit[known]
        masks = dict(visible=known & (np.abs(delta) <= tolerance),
            occluded=known & (delta > tolerance), unknown=~known,
            in_front=known & (delta < -tolerance))
        original_ids = (ids.cpu().numpy()+start).astype(id_type)
        for key, mask in masks.items():
            result[key].append(original_ids[mask])
    return {key+'_ids': np.concatenate(values) if values else np.empty(0, id_type)
            for key, values in result.items()}


class MeshDepthRenderer:
    def __init__(self, mesh, device):
        import torch
        vertices, faces = mesh_arrays(mesh)
        self.device = torch.device(device)
        if self.device.type == 'cpu':
            self.raycaster = TriangleRaycaster(vertices, faces)
        else:
            import nvdiffrast.torch as dr
            self.dr, self.context = dr, dr.RasterizeCudaContext(device=device)
            self.vertices = torch.as_tensor(vertices, device=device)
            self.faces = torch.as_tensor(faces, device=device)

    def __call__(self, camera):
        import torch
        from utils.view_tangent import camera_intrinsics
        if self.device.type == 'cpu':
            k = camera_intrinsics(camera, 'cpu').numpy()
            e = camera.world_view_transform.T.cpu().numpy()
            return torch.from_numpy(self.raycaster.depth(k, e, camera.image_width, camera.image_height))
        with torch.no_grad():
            h, w = camera.image_height, camera.image_width
            clip = clip_vertices(self.vertices, camera)[None].contiguous()
            depth = torch.full((h, w), float('inf'), device=self.device)
            for start in range(0, len(self.faces), 250000):
                raster, _ = self.dr.rasterize(self.context, clip, self.faces[start:start+250000].contiguous(),
                    resolution=(h, w), grad_db=False)
                hit, ndc = raster[0, :, :, 3] > 0, raster[0, :, :, 2].clamp(-1, 1)
                n, f = camera.znear, camera.zfar
                z = 2*n*f / (f+n-ndc*(f-n))
                depth = torch.minimum(depth, torch.where(hit, z, torch.full_like(z, float('inf'))))
            return depth.flip(0).contiguous()


class MeshDepthCache(LidarDepthCache):
    """Same fixed scene mesh for training, held-out cameras and novel poses."""
    def __init__(self, geometry, directory='', edge_pixels=1, memory_views=4, point_cloud=None, source=None):
        self.report, self.mesh = load_scene(geometry, point_cloud, source)
        super().__init__(geometry, directory, edge_pixels, memory_views)

    def make_renderer(self, device):
        return MeshDepthRenderer(self.mesh, device)

    def tile_depth(self, depth):
        return depth*(1+self.report['relative_margin'])

    def diagnostic_frame(self, camera, device):
        result = super().diagnostic_frame(camera, device)
        result['report'] = dict(kind=FORMAT, relative_margin=self.report['relative_margin'],
            reason='fixed scene mesh; missing intersections stay unknown')
        return result
