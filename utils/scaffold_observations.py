"""Exact point-row / scanner-voxel associations for fixed Scaffold parents."""
import hashlib
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData

from utils.checkpoint_snapshots import digest


def discover(source, explicit=None):
    source = Path(source)
    if explicit:
        return Path(explicit).expanduser().resolve(strict=True)
    for name in ('lidar_observations.json', 'lidar_scans.json'):
        if (source/name).is_file():
            return source/name
    raise FileNotFoundError(f'No per-image LiDAR manifest found under {source}. '
                            'Run python scripts/inspect_lidar_observations.py to inspect the capture, '
                            'or set SCAFFOLD_OBSERVATIONS to the associated manifest.')


def read_xyz(path):
    vertex = PlyData.read(path, mmap='r')['vertex'].data
    return np.stack([vertex[k] for k in ('x','y','z')], axis=1)


def anchor_digest(anchors):
    return hashlib.sha256(np.ascontiguousarray(anchors, dtype='<f4')).hexdigest()


def scanner_voxels(keys, world, size):
    # Binary search on sorted integer voxel triples, with no nearest-surface substitution.
    dtype = np.dtype([(dim, '<i8') for dim in 'xyz'])
    query = np.unique(np.round(world/size).astype('<i8'), axis=0)
    query = np.ascontiguousarray(query).view(dtype).ravel()
    pos = np.searchsorted(keys, query)
    inside = pos < len(keys)
    return pos[inside][keys[pos[inside]] == query[inside]]


def indexed_pixels(viewer, frame_id, xyz, limit):
    """Project original indexed rows with the centered training camera at native image resolution."""
    from PIL import Image
    frame = viewer.frames[frame_id]
    cam = viewer.cameras[frame.camera]
    if cam.model not in ('SIMPLE_PINHOLE', 'PINHOLE'):
        raise ValueError(f'Expected an undistorted pinhole camera for {frame.name}')
    fx = cam.params[0]
    fy = fx if cam.model == 'SIMPLE_PINHOLE' else cam.params[1]
    with Image.open(viewer.source/viewer.images/frame.name) as image:
        width, height = image.size
    local = xyz @ frame.R.T + frame.t
    local = local[np.isfinite(local).all(axis=1) & (local[:,2] > .2)]
    # Raster pixel i covers [i,i+1) in corner coordinates (CUDA ndc2Pix adds -0.5).
    uv = (local[:,:2] / local[:,2:] * [fx/cam.width, fy/cam.height] + .5) * [width, height]
    x, y = uv[((uv >= 0) & (uv < [width, height])).all(axis=1)].astype(np.int64).T
    hits = np.unique(y*width + x)
    total = len(hits)
    if total > limit:
        hits = np.sort(np.random.default_rng(frame_id).choice(hits, limit, replace=False))
    return hits, dict(size=[width, height], count=len(hits), total=total)


def prepare_membership(out, manifest, cloud, original, names, size, *, mesh_source=None, indexed_viewer=None,
                       prepare_hits=False):
    out = Path(out)
    manifest = Path(manifest) if manifest is not None else None
    meta = json.loads(manifest.read_text()) if manifest else {}
    points = read_xyz(cloud)
    keys, point_parent = np.unique(np.round(points/size), axis=0, return_inverse=True)
    anchors = (keys*size).astype(np.float32)
    point_parent = point_parent.reshape(-1)
    names = sorted(names)
    view_index = {n:i for i,n in enumerate(names)}
    indexed = indexed_viewer is not None
    kind = 'viewer_indexed_rows_v1' if indexed else meta.get('format')
    mesh = mesh_source is not None
    if mesh != (kind == 'scene_occlusion_mesh_v1'):
        raise ValueError('Measured LiDAR and first-hit mesh visibility require separate experiment variants')
    if indexed:
        # The indexed launcher canonicalizes this exact viewer PLY, keeping vertex order.
        if Path(original).resolve() != indexed_viewer.ply_path.resolve() or len(indexed_viewer.ply.vertices) != len(points):
            raise ValueError('Indexed rows and Scaffold initialization refer to different point clouds')
        for start in range(0, len(points), 262144):
            part = slice(start, start+262144)
            if not np.array_equal(points[part], indexed_viewer.ply.coords(part).astype(points.dtype, copy=False)):
                raise ValueError('Scaffold initialization changed the indexed PLY vertex order or coordinates')
        frames = {f.name:f.id for f in indexed_viewer.frames.values()}
        missing = sorted(set(names)-frames.keys())
        if missing:
            raise ValueError(f'Training images missing from the viewer sparse model: {missing[:4]}')
    elif kind == 'lidar_rows_v1' or mesh:
        # The canonical suite may have added normals to the PLY. Accept the raw
        # PLY digest as well only when its XYZ rows match one-for-one.
        if meta['point_cloud_sha256'] != digest(cloud):
            path = Path(original)
            if (path.suffix.lower() != '.ply' or meta['point_cloud_sha256'] != digest(path)
                    or not np.array_equal(points, read_xyz(path).astype(points.dtype, copy=False))):
                raise ValueError('Per-image LiDAR row IDs refer to a different point cloud or row order')
        if mesh:
            from scripts.view_comparisons import read_colmap_records
            from utils.occlusion_depth import camera_signature, record_camera
            from utils.sparse_lidar import frame_filename
            records = read_colmap_records(mesh_source)
            frames = {n: 'frames/'+frame_filename(n) for n in meta['images']}
        else:
            frames = meta['frames']
        missing = sorted(set(names)-frames.keys())
    elif kind == 'scanner_xyz_v1':
        from scripts.pgsr_depth import scan_records
        records = scan_records(manifest)
        assignments = []
        covered = set()
        for scan in meta['scans']:
            views = scan.get('images', [scan['image']] if 'image' in scan else [])
            if not isinstance(views, list) or not all(isinstance(n,str) and n for n in views):
                raise ValueError('Each scanner entry needs image names in images or a single image')
            assignments.append(sorted(set(views) & view_index.keys()))
            covered.update(views)
        missing = sorted(set(names)-covered)
    else:
        raise ValueError('Expected measured lidar_rows_v1 or scanner_xyz_v1. Mesh visibility and COLMAP tracks are separate inputs.')
    if missing:
        raise ValueError(f'No measured scanner association for {len(missing)}/{len(names)} training images, first {missing[:4]}. '
                         'Use explicit image names or supply a synchronized association from the capture.')
    if prepare_hits and not indexed:
        raise ValueError('Opacity supervision requires the indexed point rows')
    out.mkdir()
    if prepare_hits:
        max_hits = 8192
        hit_info = []
        hit_pixels = np.lib.format.open_memmap(out/'hit_pixels.npy', mode='w+', dtype=np.int32,
                                                shape=(len(names), max_hits))
    packed = np.lib.format.open_memmap(out/'parents.npy', mode='w+', dtype=np.uint8,
                                       shape=(len(names),(len(keys)+7)//8))
    received = np.zeros(len(names), dtype=np.int64)

    def add(name, ids, total):
        row = view_index[name]
        ids = np.unique(ids)
        np.bitwise_or.at(packed[row], ids//8, (1 << (ids%8)).astype(np.uint8))
        received[row] += total

    raw = hashlib.sha256()
    per_view_source = []
    if indexed or kind == 'lidar_rows_v1' or mesh:
        for i,name in enumerate(names):
            if indexed:
                ids, source = indexed_viewer.indexed_rows(frames[name])
                per_view_source.append(source)
            elif mesh:
                with np.load(manifest.parent/frames[name], allow_pickle=False) as frame:
                    signature = camera_signature(record_camera(records[name], meta['resolution'], 'cpu'), 1)
                    if (str(frame['image']) != name or str(frame['camera_signature']) != signature
                            or meta['camera_signatures'][name] != signature):
                        raise ValueError(f'Mesh visibility camera/pose mismatch for {name}')
                    ids = frame['visible_ids']
            else:
                ids = np.load(manifest.parent/frames[name], allow_pickle=False)
            if ids.ndim != 1 or ids.dtype.kind not in 'iu' or (len(ids) and (ids.min()<0 or ids.max()>=len(points))):
                raise ValueError(f'Invalid original LiDAR row IDs for {name}')
            if prepare_hits:
                hits, info = indexed_pixels(indexed_viewer, frames[name], points[ids], max_hits)
                hit_pixels[i, :len(hits)] = hits
                hit_info.append(info)
            raw.update(hashlib.sha256(ids.tobytes()).digest())
            add(name, point_parent[ids], len(ids))
            if i == 0 or (i+1)%100 == 0 or i+1 == len(names):
                print(f'Indexed {i+1}/{len(names)} training views · {name} · {len(ids):,} LiDAR rows', flush=True)
    else:
        dtype = np.dtype([(dim,'<i8') for dim in 'xyz'])
        lookup = np.ascontiguousarray(keys, dtype='<i8').view(dtype).ravel()
        for i,((path,pose),views) in enumerate(zip(records,assignments)):
            if not views:
                continue
            xyz = np.load(path, allow_pickle=False)
            if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.dtype.kind != 'f':
                raise ValueError(f'Expected floating-point scanner XYZ in {path}')
            raw.update(hashlib.sha256(xyz.tobytes()).digest())
            xyz = xyz[np.isfinite(xyz).all(axis=1)]
            world = (xyz @ pose[:3,:3].T + pose[:3,3]).astype(points.dtype)
            ids = scanner_voxels(lookup, world, size)
            for name in views:
                add(name, ids, len(xyz))
            if (i+1)%100 == 0: print(f'Indexed measured scans {i+1}/{len(records)}', flush=True)
    packed.flush()
    counts = np.array([np.unpackbits(row).sum() for row in packed])
    union = np.bitwise_or.reduce(packed, axis=0)
    if not counts.any():
        raise ValueError('No observations matched any training anchor. Check units, frame associations and scan poses.')
    report = dict(format=kind, manifest=str(manifest.resolve()) if manifest else None, manifest_sha256=digest(manifest) if manifest else None,
                  input_data_sha256=raw.hexdigest(), names=names, anchor_count=len(anchors), voxel_size=size,
                  anchor_sha256=anchor_digest(anchors), point_cloud_sha256=digest(cloud),
                  empty_views=int((counts==0).sum()), median_sensed_parents=float(np.median(counts)),
                  ever_sensed_fraction=float(np.unpackbits(union).sum()/len(anchors)),
                  per_view_parents=counts.tolist(), per_view_input_returns=received.tolist())
    if indexed:
        report['viewer_ply'] = str(indexed_viewer.ply_path.resolve())
        report['per_view_source'] = per_view_source
        report['per_view_indexed_rows'] = report.pop('per_view_input_returns')
    if prepare_hits:
        if not any(v['count'] for v in hit_info):
            raise ValueError('No indexed rows project inside any training image')
        hit_pixels.flush()
        report['hit_pixels'] = hit_info
    (out/'metadata.json').write_text(json.dumps(report)+'\n')
    print(f'{kind}: {len(names):,} views, median {report["median_sensed_parents"]:,.0f} eligible parents, '
          f'{report["empty_views"]} empty views, {report["ever_sensed_fraction"]:.1%} of parents eligible at least once.', flush=True)


class Membership:
    def __init__(self, directory, anchors, names, with_hits=False):
        directory = Path(directory)
        meta = json.loads((directory/'metadata.json').read_text())
        if meta['names'] != sorted(names) or meta['anchor_sha256'] != anchor_digest(anchors.detach().cpu().numpy()):
            raise ValueError('Sensed memberships do not match the training cameras or fixed parent positions/order')
        self.kind = meta['format']
        self.index = {name:i for i,name in enumerate(meta['names'])}
        self.bits = np.load(directory/'parents.npy', mmap_mode='r', allow_pickle=False)
        if self.bits.shape != (len(self.index),(len(anchors)+7)//8) or self.bits.dtype != np.uint8:
            raise ValueError('Invalid packed Scaffold parent observations')
        self.count, self.device = len(anchors), anchors.device
        self.identity = (digest(directory/'metadata.json'), digest(directory/'parents.npy'))

        if with_hits:
            self.hit_info = meta['hit_pixels']
            self.hits = np.load(directory/'hit_pixels.npy', mmap_mode='r', allow_pickle=False)
            if (len(self.hit_info) != len(self.index) or self.hits.ndim != 2
                    or self.hits.shape[0] != len(self.index) or self.hits.dtype != np.int32
                    or any(not 0 <= info['count'] <= self.hits.shape[1] for info in self.hit_info)):
                raise ValueError('Invalid projected hit cache')
            self.identity += (digest(directory/'hit_pixels.npy'),)

    def hit_indices(self, name, width, height):
        import torch
        i = self.index[name]
        info = self.hit_info[i]
        if info['size'] != [width, height]:
            raise ValueError(f'Projected hit dimensions differ from training image {name}')
        pixels = self.hits[i, :info['count']].astype(np.int64)
        if len(pixels) and (pixels.min() < 0 or pixels.max() >= width * height):
            raise ValueError(f'Projected hit outside training image {name}')
        return torch.from_numpy(pixels).to(self.device)

    def mask(self, name):
        import torch
        row = np.unpackbits(self.bits[self.index[name]], bitorder='little', count=self.count).view(bool)
        return torch.from_numpy(row).to(self.device)


def gate_children(values, allowed):
    """Keep compositing values, including occluders; stop unassociated child gradients."""
    import torch
    return tuple(torch.where(allowed[:,None], value, value.detach()) for value in values)


def sensed_step(optimizer, active):
    """Only active local rows and shared decoders with eligible contributions step."""
    from types import SimpleNamespace
    from utils.occlusion_culling import masked_adam_step
    local = [group for group in optimizer.param_groups if group['name'] in
             ('anchor','offset','anchor_feat','opacity','scaling','rotation','offset_sh')]
    masked_adam_step(SimpleNamespace(param_groups=local, state=optimizer.state), active)
    for group in local:
        for p in group['params']:
            p.grad = None
    optimizer.step()
