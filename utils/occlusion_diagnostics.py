"""Exact mask accounting and bounded diagnostic exports (no CUDA imports)."""
import html
import json
from pathlib import Path

import numpy as np

COLORS = np.array([[130, 140, 155], [40, 220, 120], [255, 65, 80]], dtype=np.uint8)
LABELS = ('other', 'active', 'culled')


def classify(keep, radii_on, radii_off):
    keep = np.asarray(keep, dtype=bool)
    on, off = np.asarray(radii_on), np.asarray(radii_off)
    if keep.ndim != 1 or on.shape != keep.shape or off.shape != keep.shape:
        raise ValueError('Culling mask and radius arrays must match every checkpoint row')
    if np.any(on[~keep] != 0):
        raise RuntimeError('Depth-culled Gaussians still have a raster footprint')
    if not np.array_equal(on[keep], off[keep]):
        raise RuntimeError('Kept Gaussian footprints changed when culling was toggled')
    labels = np.zeros(len(keep), np.uint8)
    labels[keep & (on > 0)] = 1
    labels[~keep] = 2
    counts = {name: int(np.count_nonzero(labels == i)) for i, name in enumerate(LABELS)}
    counts.update(total=len(keep), raster_candidates_without_culling=int(np.count_nonzero(off > 0)),
        culled_with_raster_footprint=int(np.count_nonzero((~keep) & (off > 0))))
    return labels, counts


def display_indices(labels, category, limit=0):
    """Full population unless the user explicitly requests a display-only cap."""
    if limit < 0:
        raise ValueError('Display limit cannot be negative')
    indices = np.flatnonzero(labels == category)
    if limit and len(indices) > limit:
        indices = indices[np.linspace(0, len(indices)-1, limit, dtype=np.int64)]
    return indices


def depth_points(depth, intrinsic, camera_to_world, stride=3):
    if stride < 1:
        raise ValueError('Depth stride must be positive')
    depth, k = np.asarray(depth), np.asarray(intrinsic)
    y, x = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[y, x]
    valid = np.isfinite(z) & (z > 0)
    x, y, z = x[valid], y[valid], z[valid]
    points = np.stack(((x-k[0, 2])*z/k[0, 0], (y-k[1, 2])*z/k[1, 1], z), 1)
    return (points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]).astype(np.float32)


def depth_rgb(tiles, width, height, pixel_size=16):
    """Color actual conservative tile bounds; unknown coverage is black."""
    values = np.asarray(tiles)
    valid = np.isfinite(values) & (values > 0)
    color = np.zeros((*values.shape, 3), np.uint8)
    if valid.any():
        lo, hi = np.percentile(values[valid], [2, 98])
        t = np.clip((values[valid]-lo)/max(float(hi-lo), 1e-6), 0, 1)
        color[valid] = np.stack((255*t, 220*(1-np.abs(2*t-1)), 255*(1-t)), 1).astype(np.uint8)
    else:
        lo, hi = None, None
    return np.repeat(np.repeat(color, pixel_size, axis=0), pixel_size, axis=1)[:height, :width], dict(low=lo, high=hi)


def reference_on_image(frame, intrinsic, height, width):
    """Map reference depth to native rays using both K matrices, not an image resize."""
    depth, valid, k = frame['depth'], frame['valid'], frame['intrinsics']
    y, x = np.indices((height, width), dtype=np.float32)
    u = np.floor(k[0, 0]*(x-intrinsic[0, 2])/intrinsic[0, 0]+k[0, 2]+.5).astype(int)
    v = np.floor(k[1, 1]*(y-intrinsic[1, 2])/intrinsic[1, 1]+k[1, 2]+.5).astype(int)
    h, w = depth.shape
    inside = (u >= 0) & (v >= 0) & (u < w) & (v < h)
    u, v = u.clip(0, w-1), v.clip(0, h-1)
    raw = np.where(inside, depth[v, u], 0)
    finite = np.isfinite(raw) & (raw > 0)
    trusted = inside & valid[v, u].astype(bool) & finite
    return raw, trusted


def reference_panels(frame, intrinsic, rgb, tiles):
    """Separate reference prediction, accepted surface, and usable culling area."""
    from PIL import Image, ImageFilter
    h, w = rgb.shape[:2]
    raw, trusted = reference_on_image(frame, intrinsic, h, w)
    finite = np.isfinite(raw) & (raw > 0)
    full_tiles = np.repeat(np.repeat(np.isfinite(tiles), 16, axis=0), 16, axis=1)[:h, :w]
    raw_rgb, _ = depth_rgb(raw, w, h, pixel_size=1)
    coverage = np.zeros_like(rgb)
    coverage[trusted] = (85, 155, 255)
    coverage[full_tiles] = (40, 220, 120)
    overlay = (rgb*.25).astype(np.uint8)
    overlay[trusted] = (.45*rgb[trusted]+.55*np.array([85, 155, 255])).astype(np.uint8)
    summary = dict(raw_depth_pixel_fraction=float(finite.mean()), trusted_pixel_fraction=float(trusted.mean()),
                   usable_tile_pixel_fraction=float(full_tiles.mean()), calibration=frame.get('report', {}))
    images = dict(reference_raw=('Raw reference depth; may include rejected predictions', raw_rgb),
        reference_trusted=('Trusted reference over GT; blue accepted, dark rejected/unknown', overlay),
        coverage=('Coverage: green usable tiles, blue trusted pixels lost to tile tests, black unknown', coverage))
    if 'anchors' in frame:
        anchors = frame['anchors']
        u, v, z = (np.asarray(anchors[key]) for key in ('u', 'v', 'z'))
        inside = (u >= 0) & (v >= 0) & (u < w) & (v < h) & np.isfinite(z) & (z > 0)
        u, v, z = u[inside].astype(int), v[inside].astype(int), z[inside]
        known = trusted[v, u]
        agree = known & (np.abs(raw[v, u]-z) <= .1*z)
        mismatch = known & ~agree
        anchor_rgb = rgb.copy()
        for mask, color in ((~known, (255, 195, 65)), (agree, (40, 220, 120)), (mismatch, (255, 65, 80))):
            dots = np.zeros((h, w), np.uint8)
            dots[v[mask], u[mask]] = 255
            dots = np.asarray(Image.fromarray(dots).filter(ImageFilter.MaxFilter(3))) > 0
            anchor_rgb[dots] = color
        images['anchors'] = ('Projected nearest LiDAR anchors: green within 10% of trusted depth, red disagree, gold untrusted', anchor_rgb)
        summary['lidar_anchors'] = dict(projected=len(z), agree=int(agree.sum()), disagree=int(mismatch.sum()),
                                        untrusted=int((~known).sum()))
    return images, summary


def reason_counts(keep, reliable, surface_bound, radii_off):
    """Disjoint reasons over OFF raster candidates; no out-of-view denominator."""
    candidate = np.asarray(radii_off) > 0
    keep, reliable = np.asarray(keep, bool), np.asarray(reliable, bool)
    finite = np.isfinite(np.asarray(surface_bound))
    masks = dict(culled=~keep, outside_conservative_test=keep & ~reliable,
                 missing_depth_coverage=keep & reliable & ~finite,
                 overlaps_depth_or_margin=keep & reliable & finite)
    result = {name: int(np.count_nonzero(candidate & mask)) for name, mask in masks.items()}
    if sum(result.values()) != int(candidate.sum()):
        raise RuntimeError('Culling reason partition does not match raster candidates')
    return result


def projected_overlay(xyz, labels, world_to_camera, intrinsic, rgb, chunk=131072):
    """Center markers, deliberately without z-buffering so hidden points show."""
    from PIL import Image, ImageFilter
    h, w = rgb.shape[:2]
    result = rgb.copy()
    for category in (1, 2):
        mask = np.zeros((h, w), np.uint8)
        for start in range(0, len(xyz), chunk):
            indices = np.flatnonzero(labels[start:start+chunk] == category)+start
            p = xyz[indices]@world_to_camera[:3, :3].T + world_to_camera[:3, 3]
            z = p[:, 2]
            u = np.floor(intrinsic[0, 0]*p[:, 0]/np.maximum(z, 1e-8)+intrinsic[0, 2]+.5)
            v = np.floor(intrinsic[1, 1]*p[:, 1]/np.maximum(z, 1e-8)+intrinsic[1, 2]+.5)
            good = (z > 0) & (u >= 0) & (v >= 0) & (u < w) & (v < h)
            mask[v[good].astype(int), u[good].astype(int)] = 255
        mask = np.array(Image.fromarray(mask).filter(ImageFilter.MaxFilter(3))) > 0
        result[mask] = (.25*result[mask]+.75*COLORS[category]).astype(np.uint8)
    return result


def write_colored_ply(path, xyz, labels, include_other=False, chunk=131072):
    """Full-resolution center export with original checkpoint row IDs."""
    path = Path(path)
    total = len(labels) if include_other else int(np.count_nonzero(labels))
    dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'),
             ('blue', 'u1'), ('label', 'u1'), ('gaussian_id', '<u4')]
    header = ('ply\nformat binary_little_endian 1.0\n'
        'comment Gaussian centers; labels 0 other, 1 active, 2 depth-culled\n'
        f'element vertex {total}\nproperty float x\nproperty float y\nproperty float z\n'
        'property uchar red\nproperty uchar green\nproperty uchar blue\n'
        'property uchar label\nproperty uint gaussian_id\nend_header\n')
    with path.open('wb') as stream:
        stream.write(header.encode('ascii'))
        for start in range(0, len(labels), chunk):
            ids = np.arange(start, min(start+chunk, len(labels)))
            if not include_other:
                ids = ids[labels[ids] != 0]
            rows = np.empty(len(ids), dtype=dtype)
            for j, axis in enumerate(('x', 'y', 'z')): rows[axis] = xyz[ids, j]
            for j, channel in enumerate(('red', 'green', 'blue')): rows[channel] = COLORS[labels[ids], j]
            rows['label'], rows['gaussian_id'] = labels[ids], ids
            stream.write(rows.tobytes())
    return total


def nearest_ray_point(xyz, indices, origin, direction, chunk=131072):
    """Pick the displayed center nearest the click ray in angular distance."""
    origin, direction = np.asarray(origin), np.asarray(direction)
    direction = direction / max(np.linalg.norm(direction), 1e-12)
    best, best_id = float('inf'), None
    for start in range(0, len(indices), chunk):
        ids = indices[start:start+chunk]
        delta = xyz[ids]-origin
        z = delta@direction
        score = np.maximum(np.einsum('ij,ij->i', delta, delta)-z*z, 0)/np.maximum(z*z, 1e-12)
        score[z <= 0] = np.inf
        if len(score):
            idx = score.argmin()
            if score[idx] < best:
                best, best_id = score[idx], int(ids[idx])
    return best_id


def write_report(directory, summary, images):
    from PIL import Image
    from scripts.view_comparisons import atomic_json
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    panels = []
    for key, (title, image) in images.items():
        Image.fromarray(image).save(directory/(key+'.png'))
        panels.append(f'<figure><figcaption>{html.escape(title)}</figcaption><img src="{key}.png"></figure>')
    atomic_json(directory/'summary.json', summary)
    report = ('<!doctype html><meta charset="utf-8"><title>Occlusion diagnostic</title>'
        '<style>body{background:#111827;color:#e5e7eb;font:15px system-ui;margin:30px}'
        'section{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}'
        'figure{margin:0}img{width:100%}figcaption{padding:8px}pre{white-space:pre-wrap}</style>'
        f'<h1>{html.escape(summary["image"])}</h1><p>Green: kept with raster footprint; '
        'red: depth-culled; gray: other. Raster footprint does not prove pixel contribution. '
        'Center overlay is not z-buffered. Classification and masks cover every Gaussian.</p>'
        '<section>'+''.join(panels)+'</section><pre>'+html.escape(json.dumps(summary, indent=2))+'</pre>')
    (directory/'index.html').write_text(report)
