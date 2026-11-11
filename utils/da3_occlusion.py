"""LiDAR-aligned DA3 depths used only for training-view occlusion tests."""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np

from utils.sparse_lidar import frame_filename

VERSION = 2
CALIBRATION_POLICY = dict(name='local_spatial_holdout_v2', grid=8, holdout_grid=32,
                          min_region_size=32, min_region_anchors=24,
                          min_region_holdout=6, min_inlier_fraction=.7)


def camera_matrices(record, width, height):
    """Match the centered camera used by the native Gaussian renderer exactly."""
    fx, fy = width / (2*np.tan(record['fov_x']/2)), height / (2*np.tan(record['fov_y']/2))
    intrinsic = np.array([[fx, 0, (width-1)/2], [0, fy, (height-1)/2], [0, 0, 1]], np.float32)
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = np.asarray(record['R']).T
    extrinsic[:3, 3] = record['T']
    return intrinsic, extrinsic


def reliable_pixels(depth, confidence, percentile=20., edge_relative=.05):
    """Keep confident smooth interiors; expand discontinuities by one model pixel."""
    from scipy.ndimage import maximum_filter, minimum_filter, binary_erosion
    finite = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence)
    if not finite.any():
        return np.zeros_like(finite)
    confident = finite & (confidence >= np.percentile(confidence[finite], percentile))
    lo = minimum_filter(np.where(finite, depth, 0), size=3, mode='constant', cval=0)
    hi = maximum_filter(np.where(finite, depth, np.inf), size=3, mode='constant', cval=np.inf)
    smooth = (hi-lo) <= edge_relative*np.maximum(depth, 1e-6)
    return confident & binary_erosion(smooth, iterations=1, border_value=0)


def calibration_anchors(depth, valid, u, v, z):
    """One nearest finite LiDAR observation per valid prediction pixel."""
    height, width = depth.shape
    u, v, z = np.asarray(u), np.asarray(v), np.asarray(z)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height) & np.isfinite(z) & (z > .2)
    u, v, z = u[inside].astype(int), v[inside].astype(int), z[inside]
    good = valid[v, u] & np.isfinite(depth[v, u]) & (depth[v, u] > 0)
    u, v, z = u[good], v[good], z[good]
    order = np.argsort(z, kind='stable')
    _, first = np.unique(v[order]*width+u[order], return_index=True)
    ids = order[first]
    return u[ids], v[ids], z[ids]


def fit_scale(depth, valid, u, v, z, min_anchors=200, max_relative_error=.10, holdout_grid=16):
    """Fit positive multiplicative camera-z scale with spatially held-out anchors.

    DA3 outputs depth, not inverse disparity. No additive shift is fitted.
    Median log-ratios tolerate outliers, but do not prove visibility of a merged
    cloud; held-out residuals and multiview checks remain necessary.
    """
    height, width = depth.shape
    u, v, z = calibration_anchors(depth, valid, u, v, z)
    # Equalize spatial sampling density for the single shared scale estimate.
    rng = np.random.default_rng(42)
    cell = (v*8//height)*8 + u*8//width
    ids = np.concatenate([rng.permutation(np.flatnonzero(cell == c))[:64] for c in np.unique(cell)]) if len(cell) else np.array([], int)
    u, v, z, cell = u[ids], v[ids], z[ids], cell[ids]
    report = dict(accepted=False, anchors=len(z), spatial_cells=len(np.unique(cell)), reason='insufficient anchors')
    if (len(z) < min_anchors or len(np.unique(cell)) < 8 or
            len(np.unique(cell//8)) < 3 or len(np.unique(cell % 8)) < 3):
        return None, report
    # Hold out complete small image regions, not random samples of the same ray.
    holdout = ((u*holdout_grid//width) + 3*(v*holdout_grid//height)) % 5 == 0
    if holdout.sum() < 30 or (~holdout).sum() < 100:
        return None, {**report, 'reason': 'insufficient spatial holdout'}
    ratios = np.log(z)-np.log(depth[v, u])
    log_scale = np.median(ratios[~holdout])
    for _ in range(3):
        residual = ratios[~holdout]-log_scale
        cutoff = max(.02, 3*1.4826*np.median(np.abs(residual-np.median(residual))))
        inlier = np.abs(residual) <= cutoff
        log_scale = np.median(ratios[~holdout][inlier])
    scale = float(np.exp(log_scale))
    errors = np.abs(scale*depth[v, u]/z-1)
    val = errors[holdout]
    median, p90 = float(np.median(val)), float(np.quantile(val, .9))
    fraction = float(np.mean(errors <= max_relative_error))
    val_inliers = val[val <= max_relative_error]
    inlier_p90 = float(np.quantile(val_inliers, .9)) if len(val_inliers) else None
    accepted = (median <= max_relative_error/2 and fraction >= .7 and
                len(val_inliers) >= max(30, .7*len(val)))
    report.update(accepted=bool(accepted), scale=scale, holdout_anchors=int(holdout.sum()),
                  median_relative_error=median, p90_relative_error=p90,
                  p90_inlier_relative_error=inlier_p90, inlier_fraction=fraction,
                  reason='ok' if accepted else 'LiDAR alignment residuals too large')
    return scale if accepted else None, report


def calibrate_regions(depth, valid, u, v, z, min_anchors=200, max_relative_error=.10,
                      *, scale_override=None, global_fit=None):
    """Fit one scale, then validate bounded regions independently.

    A rejected global inlier fraction is not a reason to erase well-supported
    regions. Regional checks use spatial holdouts excluded from the global fit,
    retain the same 70% inlier / 5% median-error criteria, require distributed
    anchors, and limit extrapolation from inlier anchors. No local scale/shift
    is fitted and no depth is generated in confidence/depth-boundary holes.
    Multiview validation still follows this stage in the preparer.
    """
    from scipy.ndimage import distance_transform_edt
    policy = CALIBRATION_POLICY
    h, w = depth.shape
    if global_fit is None:
        _, global_fit = fit_scale(depth, valid, u, v, z, min_anchors, max_relative_error,
                                  holdout_grid=policy['holdout_grid'])
    scale = global_fit.get('scale') if scale_override is None else scale_override
    mask = np.zeros_like(valid, dtype=bool)
    report = dict(accepted=False, policy=policy['name'], scale=scale, global_fit=global_fit,
                  accepted_regions=0, tested_regions=0, reason='no supported global scale proposal')
    if scale is None or not np.isfinite(scale) or scale <= 0:
        return None, mask, report
    u, v, z = calibration_anchors(depth, valid, u, v, z)
    error = np.abs(scale*depth[v, u]/z-1)
    holdout = ((u*policy['holdout_grid']//w)+3*(v*policy['holdout_grid']//h)) % 5 == 0
    rh, rw = (max(policy['min_region_size'], int(np.ceil(length/policy['grid']))) for length in (h, w))
    rh, rw = min(rh, h), min(rw, w)
    starts = lambda length, size: sorted(set(range(0, length-size+1, max(1, size//2))) | {length-size})
    rejected, accepted_errors = {}, []
    for y0 in starts(h, rh):
        for x0 in starts(w, rw):
            report['tested_regions'] += 1
            ids = np.flatnonzero((u >= x0) & (u < x0+rw) & (v >= y0) & (v < y0+rh))
            train, val = ids[~holdout[ids]], ids[holdout[ids]]
            reason = None
            if (len(ids) < policy['min_region_anchors'] or len(train) < 12 or
                    len(val) < policy['min_region_holdout']):
                reason = 'insufficient_local_anchors_or_holdout'
            else:
                good = ids[error[ids] <= max_relative_error]
                quadrants = ((v[good]-y0)*2//rh)*2 + (u[good]-x0)*2//rw
                if len(np.unique(quadrants)) < 3:
                    reason = 'insufficient_inlier_spatial_support'
                elif any(np.median(error[part]) > max_relative_error/2 or
                         np.mean(error[part] <= max_relative_error) < policy['min_inlier_fraction']
                         for part in (train, val)):
                    reason = 'local_alignment_residuals'
            if reason:
                rejected[reason] = rejected.get(reason, 0)+1
                continue
            mask[y0:y0+rh, x0:x0+rw] = True
            report['accepted_regions'] += 1
            accepted_errors.extend(error[val][error[val] <= max_relative_error].tolist())
    inliers = np.zeros_like(mask)
    good = error <= max_relative_error
    inliers[v[good], u[good]] = True
    if inliers.any():
        mask &= distance_transform_edt(~inliers) <= .5*min(rh, rw)
    else:
        mask[:] = False
    mask &= valid
    report.update(accepted=bool(mask.any()), rejected_regions=rejected,
        locally_calibrated_pixel_fraction=float(mask.mean()),
        p90_inlier_relative_error=float(np.quantile(accepted_errors, .9)) if accepted_errors else None,
        reason='locally validated regions' if mask.any() else 'no locally validated regions')
    return scale, mask, report


def cross_view_support(depth, valid, intrinsic, extrinsic, neighbors, tolerance=.05):
    """Count other calibrated views agreeing with a reprojected camera-z surface."""
    height, width = depth.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    cam = np.stack(((xx-intrinsic[0, 2])/intrinsic[0, 0]*depth,
                    (yy-intrinsic[1, 2])/intrinsic[1, 1]*depth, depth), axis=-1)
    world = (cam-extrinsic[:3, 3]) @ extrinsic[:3, :3]
    count = np.zeros((height, width), np.uint16)
    for other_depth, other_valid, other_k, other_e in neighbors:
        projected = world @ other_e[:3, :3].T + other_e[:3, 3]
        z = projected[..., 2]
        u = other_k[0, 0]*projected[..., 0]/np.maximum(z, 1e-6)+other_k[0, 2]
        v = other_k[1, 1]*projected[..., 1]/np.maximum(z, 1e-6)+other_k[1, 2]
        u = np.floor(np.nan_to_num(u)+.5).astype(np.int64)
        v = np.floor(np.nan_to_num(v)+.5).astype(np.int64)
        oh, ow = other_depth.shape
        inside = valid & (z > .2) & np.isfinite(z) & (u >= 0) & (u < ow) & (v >= 0) & (v < oh)
        uc, vc = u.clip(0, ow-1), v.clip(0, oh-1)
        target = other_depth[vc, uc]
        agreement = np.abs(z-target) <= tolerance*np.maximum(z, target)
        count += (inside & other_valid[vc, uc] & agreement).astype(np.uint16)
    return count


def training_tiles(depth, valid, predicted_k, camera, relative_margin=.05, edge_pixels=1):
    """Reproject to the exact training pixel grid and keep unknown depths infinite."""
    import torch
    from utils.occlusion_depth import tile_maxima
    height, width = camera.image_height, camera.image_width
    yy, xx = np.indices((height, width), dtype=np.float32)
    fx, fy = width/(2*np.tan(camera.FoVx/2)), height/(2*np.tan(camera.FoVy/2))
    u = np.floor(predicted_k[0, 0]*(xx-(width-1)/2)/fx+predicted_k[0, 2]+.5).astype(int)
    v = np.floor(predicted_k[1, 1]*(yy-(height-1)/2)/fy+predicted_k[1, 2]+.5).astype(int)
    ph, pw = depth.shape
    inside = (u >= 0) & (v >= 0) & (u < pw) & (v < ph)
    u, v = u.clip(0, pw-1), v.clip(0, ph-1)
    values = np.where(inside & valid[v, u], depth[v, u]*(1+relative_margin), np.inf).astype(np.float32)
    return tile_maxima(torch.from_numpy(values), edge_pixels).numpy()


class TrainingDepthCache:
    """Explicit training-only input; ordinary render calls never consult this cache."""
    def __init__(self, directory, source, cameras, resolution, edge_pixels=1, memory_views=4, point_cloud=None,
                 allow_partial=False):
        self.root = Path(directory).expanduser().resolve()
        report = json.loads((self.root/'metadata.json').read_text())
        if (report.get('version') not in (1, VERSION) or (not report.get('complete') and not allow_partial) or
                Path(report['source']).resolve() != Path(source).resolve() or
                report['resolution'] != resolution or report['edge_pixels'] != edge_pixels):
            raise ValueError('DA3 cache is incomplete or has incompatible source/resolution/edge padding')
        expected, actual = set(report['images']), {c.image_name for c in cameras}
        if expected != actual:
            raise ValueError('DA3 input images must exactly match the training split; held-out images are prohibited')
        if point_cloud:
            from utils.lidar_occlusion import file_digest
            if file_digest(point_cloud) != report.get('point_cloud_sha256'):
                raise ValueError('DA3 calibration cloud differs from the Gaussian initialization cloud')
        self.report = report
        self.signatures = report['camera_signatures']
        self.edge_pixels, self.memory_views = edge_pixels, memory_views
        self.memory = OrderedDict()
        self.surface_memory = OrderedDict()
        self.available_images = set()
        for camera in cameras:
            if (self.root/'frames'/frame_filename(camera.image_name)).is_file():
                self.available_images.add(camera.image_name)
            elif not allow_partial:
                raise FileNotFoundError(f'Missing DA3 frame: {camera.image_name}')
        if not self.available_images:
            raise ValueError('DA3 cache has no prepared frames; run prepare_da3_occlusion.py first')

    def diagnostic_frame(self, camera, device):
        """Expose raw prediction, trusted mask and calibration anchors separately."""
        self.tiles(camera, device)  # Same identity validation as training.
        with np.load(self.root/'frames'/frame_filename(camera.image_name), allow_pickle=False) as data:
            result = {key: data[key].copy() for key in ('depth', 'valid', 'intrinsics')}
            result['report'] = json.loads(str(data['report'])) if 'report' in data else {}
            result['cache_version'] = self.report['version']
            for key in ('confidence_valid', 'calibration_valid', 'support_views', 'raw_prediction', 'confidence'):
                if key in data and (key != 'confidence_valid' or self.report['version'] >= 2):
                    result[key] = data[key].copy()
        path = self.root/'anchors'/frame_filename(camera.image_name)
        if path.is_file():
            with np.load(path, allow_pickle=False) as data:
                result['anchors'] = {key: data[key].copy() for key in ('u', 'v', 'z')}
        return result

    def tiles(self, camera, device):
        import torch
        from utils.occlusion_depth import camera_signature, TILE
        key = (camera.image_name, str(device))
        if key not in self.memory:
            signature = camera_signature(camera, self.edge_pixels)
            if self.signatures.get(camera.image_name) != signature:
                raise ValueError('DA3 depth pose or intrinsics do not match the training camera')
            with np.load(self.root/'frames'/frame_filename(camera.image_name), allow_pickle=False) as data:
                tiles = data['tiles'].astype(np.float32)
                if str(data['image']) != camera.image_name:
                    raise ValueError('DA3 image identity mismatch')
            shape = ((camera.image_height+TILE-1)//TILE, (camera.image_width+TILE-1)//TILE)
            if tiles.shape != shape or np.isnan(tiles).any() or np.any(tiles <= 0):
                raise ValueError('Invalid DA3 tile depths')
            self.memory[key] = torch.as_tensor(tiles, device=device)
            while len(self.memory) > self.memory_views:
                self.memory.popitem(last=False)
        self.memory.move_to_end(key)
        return self.memory[key]

    def surface(self, camera, device, radius_pixels=3, max_radius=.1):
        """Use calibrated raw z and trusted mask, never inflated culling tiles."""
        import torch
        from utils.view_tangent import surface_frame
        self.tiles(camera, device)  # Enforce the same camera/cache identity check.
        key = (camera.image_name, str(device), radius_pixels, max_radius)
        if key not in self.surface_memory:
            with np.load(self.root/'frames'/frame_filename(camera.image_name), allow_pickle=False) as data:
                if str(data['image']) != camera.image_name:
                    raise ValueError('DA3 surface image identity mismatch')
                depth = torch.as_tensor(data['depth'].astype(np.float32), device=device)
                valid = torch.as_tensor(data['valid'].astype(bool), device=device)
                intrinsic = torch.as_tensor(data['intrinsics'].astype(np.float32), device=device)
            self.surface_memory[key] = surface_frame(depth, valid, intrinsic, radius_pixels, max_radius)
            while len(self.surface_memory) > self.memory_views:
                self.surface_memory.popitem(last=False)
        self.surface_memory.move_to_end(key)
        return self.surface_memory[key]
