#!/usr/bin/env python3
"""Infer pose-conditioned DA3, calibrate with LiDAR, and cache training occluders.

Only selected training images enter the network. The cloud and COLMAP poses must
already share a coordinate frame. Frontmost cloud projection is an approximation
to visibility; scale fitting, held-out residuals, and multiview checks filter it.
"""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.da3_occlusion import (VERSION, CALIBRATION_POLICY, camera_matrices, reliable_pixels, calibrate_regions,
                                 cross_view_support, training_tiles)
from utils.lidar_occlusion import file_digest
from utils.sparse_lidar import frame_filename
from scripts.view_comparisons import read_colmap_records, atomic_json


def windows(names, core=4, context=2):
    for start in range(0, len(names), core):
        end = min(len(names), start+core)
        yield names[max(0, start-context):min(len(names), end+context)], names[start:end]


def inspection_targets(names, count=0, image=None):
    """Select a few training views while retaining their normal inference windows."""
    if image:
        if image not in names:
            raise ValueError('--inspect-image must be in the training keyframe manifest')
        return {image}
    if count:
        indices = ([len(names)//2] if count == 1 else
                   (np.linspace(0, len(names)-1, min(count, len(names)))+.5).astype(int))
        return {names[i] for i in indices}
    return set(names)


def save_npz(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.' + uuid.uuid4().hex + '.tmp.npz')
    try:
        np.savez_compressed(tmp, **values)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def front_points(points, intrinsic, extrinsic, width, height, cell_size=4):
    import torch
    k = torch.as_tensor(intrinsic, device=points.device)
    e = torch.as_tensor(extrinsic, device=points.device)
    cam = points @ e[:3, :3].T + e[:3, 3]
    z = cam[:, 2]
    u = k[0, 0]*cam[:, 0]/z+k[0, 2]
    v = k[1, 1]*cam[:, 1]/z+k[1, 2]
    inside = (z > .2) & torch.isfinite(z) & (u >= -.5) & (u < width-.5) & (v >= -.5) & (v < height-.5)
    u, v, z = torch.floor(u[inside]+.5).long(), torch.floor(v[inside]+.5).long(), z[inside]
    cells_w, cells_h = (width+cell_size-1)//cell_size, (height+cell_size-1)//cell_size
    cell = (v//cell_size)*cells_w+u//cell_size
    nearest = torch.full((cells_w*cells_h,), float('inf'), device=points.device)
    nearest.scatter_reduce_(0, cell, z, reduce='amin', include_self=True)
    front = z <= nearest[cell]+1e-6
    return tuple(a[front].cpu().numpy() for a in (u, v, z))


def preview(path, record, depth, trusted, report):
    from PIL import Image, ImageDraw
    h, w = depth.shape
    with Image.open(record['image_path']) as image:
        rgb = np.array(image.convert('RGB').resize((w, h)))
    finite = np.isfinite(depth) & (depth > 0)
    quantiles = np.quantile(depth[finite], [.02, .98]) if finite.any() else [0, 1]
    t = np.nan_to_num((depth-quantiles[0])/max(quantiles[1]-quantiles[0], 1e-6)).clip(0, 1)
    colors = np.stack((255*(1-t), 200*(1-np.abs(2*t-1)), 255*t), axis=-1).astype(np.uint8)
    colors[~finite] = 0
    overlay = rgb.copy()
    overlay[trusted] = (.45*rgb[trusted]+.55*np.array([50, 220, 90])).astype(np.uint8)
    canvas = Image.new('RGB', (3*w, h+40), '#171b24')
    for i, pixels in enumerate((rgb, colors, overlay)):
        canvas.paste(Image.fromarray(pixels), (i*w, 40))
    draw = ImageDraw.Draw(canvas)
    for i, label in enumerate(('RGB', 'LiDAR-aligned DA3 depth', 'Green: trusted occlusion pixels')):
        draw.text((i*w+8, 12), label, fill='white')
    canvas.save(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--keyframes', type=Path, required=True)
    p.add_argument('--point-cloud', type=Path, required=True, help='The exact registered initialization PLY')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', default='3')
    p.add_argument('--resolution', type=int, default=-1, help='Native GS training resolution')
    p.add_argument('--model-id', default='depth-anything/DA3-LARGE')
    p.add_argument('--model-revision', default='main')
    p.add_argument('--process-res', type=int, default=756)
    p.add_argument('--core-views', type=int, default=4)
    p.add_argument('--context-views', type=int, default=2)
    p.add_argument('--confidence-percentile', type=float, default=20.)
    p.add_argument('--min-anchors', type=int, default=200)
    p.add_argument('--max-relative-error', type=float, default=.10)
    p.add_argument('--multiview-tolerance', type=float, default=.05)
    p.add_argument('--min-support-views', type=int, default=2)
    p.add_argument('--relative-margin', type=float, default=.05)
    p.add_argument('--edge-pixels', type=int, default=1)
    p.add_argument('--preview-count', type=int, default=16)
    p.add_argument('--inspect-views', type=int, default=0,
                   help='Prepare only this many distributed training views for inspection; 0 prepares all')
    p.add_argument('--inspect-image', help='Prepare this exact training image with its usual context views')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    numeric = [args.confidence_percentile, args.max_relative_error, args.multiview_tolerance, args.relative_margin]
    if (not np.isfinite(numeric).all() or not 0 <= args.confidence_percentile < 100 or
            min(args.core_views, args.context_views, args.min_support_views, args.preview_count) < 1 or
            args.min_anchors < 100 or args.process_res < 140 or args.edge_pixels < 0 or
            min(args.max_relative_error, args.multiview_tolerance) <= 0 or args.relative_margin < 0 or
            args.resolution == 0 or args.resolution < -1 or args.inspect_views < 0):
        raise ValueError('Invalid DA3 processing/calibration settings')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    from PIL import Image
    from tqdm import tqdm
    from utils.occlusion_depth import record_camera, camera_signature
    source, output, cloud = args.source.resolve(), args.output.resolve(), args.point_cloud.resolve()
    records = read_colmap_records(source)
    names = sorted(json.loads(args.keyframes.read_text())['selected_images'])
    if len(names) != len(set(names)) or len(names) < 3 or set(names)-records.keys():
        raise ValueError('Require at least three unique registered training frames')
    targets = inspection_targets(names, args.inspect_views, args.inspect_image)
    inspect_only = bool(args.inspect_views or args.inspect_image)
    cameras = {name: record_camera(records[name], args.resolution, device='cpu') for name in names}
    signatures = {n: camera_signature(c, args.edge_pixels) for n, c in cameras.items()}
    images_identity = [(n, Path(records[n]['image_path']).stat().st_size,
                       Path(records[n]['image_path']).stat().st_mtime_ns) for n in names]
    config = dict(version=VERSION, calibration_policy=CALIBRATION_POLICY, source=str(source),
        point_cloud=str(cloud), point_cloud_sha256=file_digest(cloud),
        images=names, image_identity_sha256=hashlib.sha256(json.dumps(images_identity).encode()).hexdigest(),
        camera_signatures=signatures, **{k: v for k, v in vars(args).items()
            if k not in ('source', 'output', 'point_cloud', 'keyframes', 'gpu', 'preview_count',
                         'inspect_views', 'inspect_image')})
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output/'metadata.json'
    existing = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    # Scene copies input.ply into each run. Byte-identical copies have the same
    # geometric identity; their filesystem location must not invalidate a cache.
    if existing and any(existing.get(k) != v for k, v in config.items() if k != 'point_cloud'):
        raise ValueError(f'Existing DA3 cache has different inputs/settings or an older calibration policy; '
                         f'use a new output directory such as {output}_v{VERSION}')
    if existing.get('complete'):
        if not all((output/'frames'/frame_filename(n)).is_file() for n in names):
            raise ValueError('Completed DA3 cache is missing frames')
        print(f'Reusing complete aligned DA3 cache: {output}', flush=True)
        return 0
    if inspect_only and all((output/'frames'/frame_filename(n)).is_file() for n in targets):
        print(f'Reusing {len(targets)} diagnostic DA3 views: {output} (partial cache is not training-ready)', flush=True)
        return 0
    from scripts.check_da3_environment import require_cuda
    require_cuda(torch, args.gpu)
    from huggingface_hub import model_info
    revision = existing.get('resolved_model_revision') or model_info(args.model_id, revision=args.model_revision).sha
    metadata = dict(config, resolved_model_revision=revision, complete=False,
                    policy='training images only; evaluation and novel views render without DA3 culling')
    atomic_json(metadata_path, metadata)
    from depth_anything_3.api import DepthAnything3
    from plyfile import PlyData
    torch.manual_seed(42)
    model = DepthAnything3.from_pretrained(args.model_id, revision=revision).to('cuda').eval()
    vertex = PlyData.read(str(cloud))['vertex']
    xyz = np.stack([vertex[k] for k in ('x', 'y', 'z')], axis=-1).astype(np.float32)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if not len(xyz):
        raise ValueError('Initialization cloud has no finite points')
    points = torch.from_numpy(xyz).to('cuda')
    del xyz, vertex
    previews = set(names[i] for i in np.linspace(0, len(names)-1, min(args.preview_count, len(names)), dtype=int))
    if inspect_only:
        previews = targets
    (output/'previews').mkdir(exist_ok=True)
    rows = {}
    work = [(batch, [n for n in core if n in targets])
            for batch, core in windows(names, args.core_views, args.context_views) if targets.intersection(core)]
    for batch, core in tqdm(work, desc='Aligning DA3 windows'):
        if all((output/'frames'/frame_filename(n)).exists() for n in core):
            for n in core:
                with np.load(output/'frames'/frame_filename(n), allow_pickle=False) as data:
                    rows[n] = json.loads(str(data['report']))
            continue
        images, intrinsics, extrinsics = [], [], []
        # Same native training image dimensions across a batch prevents DA3's
        # automatic crop of mixed-size images from changing camera coverage.
        shapes = {(cameras[n].image_width, cameras[n].image_height) for n in batch}
        if len(shapes) != 1:
            raise ValueError('DA3 windows currently require a common training image size')
        for n in batch:
            camera = cameras[n]
            with Image.open(records[n]['image_path']) as image:
                images.append(image.convert('RGB').resize((camera.image_width, camera.image_height), Image.Resampling.LANCZOS))
            k, e = camera_matrices(records[n], camera.image_width, camera.image_height)
            intrinsics.append(k)
            extrinsics.append(e)
        with torch.inference_mode():
            prediction = model.inference(images, extrinsics=np.stack(extrinsics), intrinsics=np.stack(intrinsics),
                align_to_input_ext_scale=True, infer_gs=False, process_res=args.process_res,
                process_res_method='upper_bound_resize', ref_view_strategy='middle')
        if prediction.conf is None or prediction.intrinsics is None:
            raise ValueError('Use DA3 main-series weights with confidence and pose conditioning')
        calibrated = []
        for i, n in enumerate(batch):
            raw = np.asarray(prediction.depth[i], np.float32)
            conf = np.asarray(prediction.conf[i], np.float32)
            k = np.asarray(prediction.intrinsics[i], np.float32)
            if raw.ndim != 2 or conf.shape != raw.shape or k.shape != (3, 3) or not np.isfinite(k).all():
                raise ValueError('Unexpected DA3 output dimensions/intrinsics')
            camera, full_k = cameras[n], intrinsics[i]
            anchor_path = output/'anchors'/frame_filename(n)
            if anchor_path.exists():
                with np.load(anchor_path, allow_pickle=False) as data:
                    u, v, z = data['u'], data['v'], data['z']
            else:
                with torch.inference_mode():
                    u, v, z = front_points(points, full_k, extrinsics[i], camera.image_width, camera.image_height)
                save_npz(anchor_path, u=u, v=v, z=z)
            pu = np.floor(k[0, 0]*(u-full_k[0, 2])/full_k[0, 0]+k[0, 2]+.5).astype(int)
            pv = np.floor(k[1, 1]*(v-full_k[1, 2])/full_k[1, 1]+k[1, 2]+.5).astype(int)
            confident = reliable_pixels(raw, conf, args.confidence_percentile)
            scale, valid, report = calibrate_regions(raw, confident, pu, pv, z, args.min_anchors, args.max_relative_error)
            report.update(projected_lidar_anchors=len(z), confident_pixel_fraction=float(confident.mean()))
            depth = np.where(np.isfinite(raw), raw*(scale or 1.), 0).astype(np.float32)
            calibrated.append((depth, valid, k, extrinsics[i], report, confident, raw, conf))
        for i, n in enumerate(batch):
            if n not in core:
                continue
            depth, valid, k, e, report, confident, raw, conf = calibrated[i]
            neighbors = [entry[:4] for j, entry in enumerate(calibrated) if j != i]
            support = cross_view_support(depth, valid, k, e, neighbors, args.multiview_tolerance)
            trusted = valid & (support >= args.min_support_views)
            uncertainty = max(args.relative_margin, report.get('p90_inlier_relative_error') or 0.)
            tiles = training_tiles(depth, trusted, k, cameras[n], uncertainty, args.edge_pixels)
            report = dict(report, image=n, trusted_pixel_fraction=float(trusted.mean()),
                          reliable_tile_fraction=float(np.isfinite(tiles).mean()), relative_margin=uncertainty)
            report['empty_reference_reason'] = (None if np.isfinite(tiles).any() else
                'no locally validated LiDAR alignment' if not valid.any() else
                'no pixels survived multiview validation' if not trusted.any() else
                'trusted pixels do not cover any complete culling tile')
            save_npz(output/'frames'/frame_filename(n), image=n, depth=depth, valid=trusted,
                     intrinsics=k, extrinsics=e, tiles=tiles, report=json.dumps(report),
                     confidence_valid=confident, calibration_valid=valid, support_views=support,
                     raw_prediction=raw, confidence=conf)
            rows[n] = report
            print(f"{n}: confident {confident.mean():.1%}, locally calibrated {valid.mean():.1%}, "
                  f"multiview trusted {trusted.mean():.1%}, usable tiles {np.isfinite(tiles).mean():.1%}", flush=True)
            if report['empty_reference_reason']:
                print(f"EMPTY OCCLUSION REFERENCE: {n}: {report['empty_reference_reason']}", flush=True)
            if n in previews:
                preview(output/'previews'/(Path(frame_filename(n)).stem+'.jpg'), records[n], depth, trusted, report)
        atomic_json(output/'calibration.json', dict(frames=[rows[n] for n in names if n in rows]))
    # Include prior diagnostic windows when resuming; never promote a partial
    # cache to training-ready just because the sampled views calibrated well.
    for n in names:
        path = output/'frames'/frame_filename(n)
        if n not in rows and path.is_file():
            with np.load(path, allow_pickle=False) as data:
                rows[n] = json.loads(str(data['report']))
    atomic_json(output/'calibration.json', dict(frames=[rows[n] for n in names if n in rows]))
    summary = dict(accepted_frames=sum(r['accepted'] for r in rows.values()), frames=len(rows), expected_frames=len(names),
                   empty_reference_frames=sum(r['reliable_tile_fraction'] == 0 for r in rows.values()),
                   mean_reliable_tile_fraction=float(np.mean([r['reliable_tile_fraction'] for r in rows.values()])))
    metadata.update(summary=summary)
    # Reject unusable preparations instead of silently training with empty masks.
    usable = (len(rows) == len(names) and summary['accepted_frames'] >= .5*len(names)
              and summary['mean_reliable_tile_fraction'] > .01)
    atomic_json(metadata_path, dict(metadata, complete=usable))
    items = ''.join('<h3>'+html.escape(n)+'</h3><pre>'+html.escape(json.dumps(rows[n], indent=2))+
        '</pre><img style="max-width:100%" src="previews/'+Path(frame_filename(n)).stem+'.jpg">'
        for n in names if n in rows and (output/'previews'/(Path(frame_filename(n)).stem+'.jpg')).is_file())
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>DA3 LiDAR calibration</title>'
        '<body style="background:#171b24;color:#eee;font:16px sans-serif;margin:24px"><h1>DA3 / LiDAR calibration</h1>'
        '<p>Training views only. Green regions have confidence, boundary, and cross-view checks. '
        'These checks reduce errors; they do not establish ground-truth visibility.</p><pre>'+html.escape(json.dumps(summary, indent=2))+'</pre>'+items)
    print(json.dumps(summary, indent=2))
    print(f'Calibration previews: {output}/index.html')
    if inspect_only:
        print('Diagnostic preparation finished. Inspect coverage before preparing all views or training.', flush=True)
        return 0
    if not usable:
        raise RuntimeError('DA3 calibration/coverage is insufficient; inspect index.html and calibration.json before training')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
