"""Pose-aware, training-only LiDAR mesh cutoffs with bounded depth caching.

Depth is camera-z, as in the splat compositor. Unknown rays remain infinite.
Only primary measured geometry is used; learned opacity never defines occlusion.
"""
from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

DEFAULTS = dict(schema=1, absolute_margin=.05, relative_margin=.005,
                max_margin=.15, edge_pixels=2, memory_views=2, pose_min_pixels=256)


def validate(policy):
    if not isinstance(policy, dict) or set(policy) != set(DEFAULTS) or policy['schema'] != 1:
        raise ValueError('Invalid pose-culling policy')
    if (any(type(policy[k]) is not int or policy[k] < 1 for k in
            ('edge_pixels', 'memory_views', 'pose_min_pixels')) or
            not all(math.isfinite(policy[k]) and policy[k] >= 0 for k in
                    ('absolute_margin', 'relative_margin', 'max_margin')) or
            not 0 < policy['absolute_margin'] <= policy['max_margin']):
        raise ValueError('Invalid pose-culling tolerances or cache limits')
    return dict(policy)


def requested(opt):
    value = getattr(opt, 'pose_culling', '')
    return validate(json.loads(value) if isinstance(value, str) else value) if value else None


def geometry_inputs(path, point_cloud=None, source=None):
    from utils.scene_occlusion import load_scene
    report, mesh = load_scene(path, point_cloud, source)
    if report.get('geometry_source') != 'full_initialization_lidar' or report.get('diagnostic_only'):
        raise ValueError('Pose culling requires the complete measured LiDAR scene mesh')
    return [Path(path).resolve(), mesh.resolve()]


def bind_policy(model, opt, fresh):
    policy = requested(opt)
    state = model.fixed_lidar_state
    if fresh:
        if policy is not None:
            state['pose_culling'] = policy
            state['pose_culling_reference'] = getattr(opt, 'pose_culling_reference', None)
    elif state.get('pose_culling') != policy:
        raise ValueError('Checkpoint pose-culling policy differs from requested policy')
    elif state.get('pose_culling_reference') != getattr(opt, 'pose_culling_reference', None):
        raise ValueError('Checkpoint pose-culling geometry differs from requested geometry')


class PoseMeshDepthCache:
    """Two frames in device memory; one replaceable disk entry per view/size.

    Pose proposals and Gaussian steps while poses change are memory-only. Fixed
    warmup/final cameras cache on disk, with a full pose signature in each entry.
    Rejected proposals never change the original COLMAP data or accumulate files.
    """
    def __init__(self, geometry, directory, policy):
        from utils.scene_occlusion import load_scene
        from utils.lidar_occlusion import file_digest
        self.policy = validate(policy)
        geometry_inputs(geometry)
        self.report, self.mesh = load_scene(geometry)
        self.directory = Path(directory or Path(geometry).parent/'pose_depth_cache') / file_digest(geometry)
        self.memory, self.renderers = OrderedDict(), {}

    def frame(self, camera, device):
        import numpy as np
        import torch
        import torch.nn.functional as F
        from utils.occlusion_depth import camera_signature, pixel_bounds
        from utils.scene_occlusion import MeshDepthRenderer
        signature = camera_signature(camera, self.policy['edge_pixels'])
        key = (signature, str(device))
        if key in self.memory:
            self.memory.move_to_end(key)
            return self.memory[key]
        persistent = not getattr(camera, 'occlusion_transient', False)
        slot = json.dumps([getattr(camera, 'image_name', 'novel'), camera.image_width, camera.image_height])
        path = self.directory / (hashlib.sha256(slot.encode()).hexdigest()+'.npz')
        depth = None
        if persistent and path.is_file():
            with np.load(path, allow_pickle=False) as data:
                if str(data['signature']) == signature:
                    depth = torch.as_tensor(data['depth'], device=device, dtype=torch.float32)
        if depth is None:
            if str(device) not in self.renderers:
                # Intentionally exclude predicted supplementary surfaces.
                self.renderers[str(device)] = MeshDepthRenderer(self.mesh, device)
            with torch.no_grad():
                depth = self.renderers[str(device)](camera).detach()
            if persistent:
                self.directory.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(f'{path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npz')
                try:
                    np.savez_compressed(tmp, signature=np.array(signature), depth=depth.cpu().numpy())
                    os.replace(tmp, path)
                finally:
                    tmp.unlink(missing_ok=True)
        if depth.shape != (camera.image_height, camera.image_width) or bool(torch.isnan(depth).any()) or bool((depth <= 0).any()):
            raise ValueError('Invalid pose-aware mesh depth')
        edge = self.policy['edge_pixels']
        far = pixel_bounds(depth, edge)
        near = -F.max_pool2d(F.pad(-depth[None, None], (edge,)*4, value=-float('inf')),
                            2*edge+1, stride=1)[0, 0]
        margin = (self.policy['absolute_margin'] + self.policy['relative_margin']*far).clamp(max=self.policy['max_margin'])
        # Pixel mask is held fixed across one pose proposal/acceptance pair.
        interior = torch.isfinite(far) & ((far-near) <= margin)
        frame = dict(depth=depth, cutoff=(far+margin).contiguous(), pose_mask=interior, signature=signature)
        self.memory[key] = frame
        while len(self.memory) > self.policy['memory_views']:
            self.memory.popitem(last=False)
        return frame

    def pixel_depth(self, camera, device):
        # This provider already includes its bounded metric margin.
        return self.frame(camera, device)['cutoff']


def provider(pipe):
    if not hasattr(pipe, '_occlusion_provider'):
        pipe._occlusion_provider = PoseMeshDepthCache(pipe.occlusion_geometry, pipe.occlusion_cache, pipe.pose_culling)
    if not isinstance(pipe._occlusion_provider, PoseMeshDepthCache):
        raise ValueError('Pose culling cannot reuse a legacy depth provider')
    return pipe._occlusion_provider


def statistics(package, camera, pc, pipe, iteration):
    import torch
    flags = package['occlusion_fragment_flags']
    from utils.lidar_activation import active_mask
    active=active_mask(pc)
    selected=slice(None) if active is None else active
    allocated=len(flags)
    flags=flags[selected]
    candidates = int((package['raster_radii'][selected] > 0).sum())
    frame = provider(pipe).frame(camera, pc.get_xyz.device)
    return dict(iteration=iteration, image=camera.image_name, gaussians=len(flags),
        allocated_gaussians=allocated,
        image_footprint_candidates=candidates, contributing=int(((flags & 1) != 0).sum()),
        fully_blocked=int((flags == 2).sum()), partially_blocked=int((flags == 3).sum()),
        fully_blocked_candidate_fraction=int((flags == 2).sum())/max(candidates, 1),
        cutoff_pixel_fraction=float(torch.isfinite(frame['cutoff']).float().mean()),
        pose_interior_pixel_fraction=float(frame['pose_mask'].float().mean()),
        depth_signature=frame['signature'], policy=pipe.pose_culling)
