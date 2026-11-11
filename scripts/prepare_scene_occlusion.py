#!/usr/bin/env python3
"""Fuse a fixed scene surface, then precompute first-hit depth and LiDAR visibility."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_da3_occlusion import save_npz
from scripts.view_comparisons import atomic_json, read_colmap_records
from utils.lidar_occlusion import file_digest
from utils.scene_occlusion import FORMAT, POINT_CLASSES, MeshDepthCache, classify_lidar, load_scene, scene_raycaster
from utils.sparse_lidar import frame_filename


def read_points(path):
    from plyfile import PlyData
    vertices = PlyData.read(str(path))['vertex']
    # Original PLY row order is the stable identity for all visibility outputs.
    return np.stack([vertices[key] for key in ('x', 'y', 'z')], axis=1).astype(np.float32)


def fuse(args):
    import open3d as o3d
    from scipy.spatial import cKDTree
    from tqdm import tqdm
    depth_dir, output, cloud = args.depth_dir.resolve(), args.output.resolve(), args.point_cloud.resolve()
    metadata = json.loads((depth_dir/'metadata.json').read_text())
    diagnostic = bool(getattr(args, 'allow_partial', False))
    if metadata.get('version') != 2 or (not metadata.get('complete') and not diagnostic):
        raise ValueError('Prepare the COMPLETE v2 DA3 training cache before scene fusion; a diagnostic view is insufficient')
    cloud_hash = file_digest(cloud)
    if metadata['point_cloud_sha256'] != cloud_hash:
        raise ValueError('LiDAR cloud differs from the depth calibration cloud')
    names = metadata['images']
    if diagnostic:
        names = [n for n in names if (depth_dir/'frames'/frame_filename(n)).is_file()]
        if len(names) < 3:
            raise ValueError('A multi-frame surface preview requires at least three prepared views')
    frame_paths = [depth_dir/'frames'/frame_filename(name) for name in names]
    identity = [(name, file_digest(path)) for name, path in zip(names, frame_paths)]
    config = dict(format=FORMAT, source=metadata['source'], training_images=names,
        camera_signatures=metadata.get('camera_signatures', {}), resolution=metadata.get('resolution', -1),
        point_cloud=str(cloud), point_cloud_sha256=cloud_hash, depth_directory=str(depth_dir),
        depth_frames_sha256=hashlib.sha256(json.dumps(identity).encode()).hexdigest(),
        voxel_size=args.voxel_size, truncation=args.truncation, depth_max=args.depth_max,
        lidar_support_distance=args.lidar_support_distance, relative_margin=args.relative_margin)
    if diagnostic:
        config['diagnostic_only'] = True
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/'scene.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if any(old.get(key) != value for key, value in config.items() if key != 'point_cloud'):
            raise ValueError('Scene output has different inputs/settings; choose a new output directory')
        if old.get('complete'):
            load_scene(manifest, cloud)
            print(f'Reusing fixed scene mesh: {manifest}', flush=True)
            return
    atomic_json(manifest, dict(config, complete=False))
    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=args.voxel_size,
        sdf_trunc=args.truncation, color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        depth_sampling_stride=1)
    used, skipped = 0, []
    for name, path in tqdm(zip(names, frame_paths), total=len(names), desc='Fusing scene surfaces (CPU)'):
        with np.load(path, allow_pickle=False) as frame:
            depth, valid = frame['depth'], frame['valid']
            k, e = frame['intrinsics'], frame['extrinsics']
            valid = valid & np.isfinite(depth) & (depth > .2) & (depth < args.depth_max)
            if not valid.any():
                skipped.append(name)
                continue
            depth = np.ascontiguousarray(np.where(valid, depth, 0), dtype=np.float32)
        h, w = depth.shape
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.zeros((h, w, 3), np.uint8)), o3d.geometry.Image(depth),
            depth_scale=1., depth_trunc=args.depth_max, convert_rgb_to_intensity=False)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, k[0, 0], k[1, 1], k[0, 2], k[1, 2])
        volume.integrate(rgbd, intrinsic, e.astype(np.float64))
        used += 1
    mesh = volume.extract_triangle_mesh()
    del volume
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if not len(faces):
        raise RuntimeError('Fusion produced no surfaces; inspect the calibrated depths before training')
    print(f'Checking {len(vertices):,} mesh vertices against the full LiDAR cloud...', flush=True)
    points = read_points(cloud)
    finite = points[np.isfinite(points).all(1)]
    if not len(finite):
        raise ValueError('No finite initialization points')
    tree = cKDTree(finite)
    supported = np.zeros(len(vertices), bool)
    for start in range(0, len(vertices), 262144):
        distance, _ = tree.query(vertices[start:start+262144], workers=args.threads,
                                 distance_upper_bound=args.lidar_support_distance)
        supported[start:start+len(distance)] = np.isfinite(distance)
    before = len(faces)
    mesh.remove_triangles_by_mask(~supported[faces].all(1))
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    if not len(mesh.triangles):
        raise RuntimeError('No fused triangles have LiDAR support; check calibration and coordinate frames')
    # No Poisson closure, convex hull, or hole filling: openings remain openings.
    save_npz(output/'mesh.npz', vertices=np.asarray(mesh.vertices, np.float32), faces=np.asarray(mesh.triangles, np.int32))
    mesh.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(str(output/'mesh.ply'), mesh):
        raise RuntimeError('Failed to write mesh.ply')
    result = dict(config, complete=True, mesh='mesh.npz', mesh_sha256=file_digest(output/'mesh.npz'),
        input_frames=len(names), integrated_frames=used, skipped_empty_frames=skipped,
        triangles_before_support_filter=before, triangles=len(mesh.triangles), vertices=len(mesh.vertices),
        policy='training depths only; local LiDAR alignment and multiview checks; no hole filling')
    atomic_json(manifest, result)
    print(f'Fixed scene: {len(mesh.triangles):,} triangles from {used}/{len(names)} training views: {manifest}', flush=True)


def precompute_views(args):
    import torch
    from tqdm import tqdm
    from utils.occlusion_depth import record_camera, camera_signature
    from utils.view_tangent import camera_intrinsics
    from utils.occlusion_diagnostics import reference_panels, write_report
    from PIL import Image
    scene, mesh = load_scene(args.scene, args.point_cloud)
    visibility_backend = getattr(args, 'visibility_backend', 'auto')
    if visibility_backend == 'auto':
        visibility_backend = 'cuda' if args.device == 'cuda' else 'exact'
    if visibility_backend == 'cuda' and args.device != 'cuda':
        raise ValueError('CUDA visibility requires --device cuda')
    source = Path(scene['source'])
    output = args.output.resolve() if args.output else args.scene.resolve().parent/'visibility'
    records = read_colmap_records(source)
    all_names = sorted(records) if args.split == 'all' else sorted(scene['training_images'])
    if args.image and args.image not in all_names:
        raise ValueError('Requested image is not registered in the selected split')
    names = [args.image] if args.image else all_names
    cameras = {name: record_camera(records[name], args.resolution, device='cpu') for name in all_names}
    config = dict(format=FORMAT, scene_manifest=str(args.scene.resolve()),
        scene_sha256=file_digest(args.scene), point_cloud_sha256=scene['point_cloud_sha256'],
        source=str(source), resolution=args.resolution, split=args.split, images=all_names,
        camera_signatures={name: camera_signature(camera, 1) for name, camera in cameras.items()},
        absolute_margin=args.margin, relative_margin=scene['relative_margin'], edge_pixels=1,
        point_ids='original input PLY vertex row indices',
        classification=('CUDA first-hit pixel depth lookup per point; outside/invalid rows are omitted'
                        if visibility_backend == 'cuda' else
                        'exact first-hit mesh ray per point; outside/invalid rows are omitted'))
    output.mkdir(parents=True, exist_ok=True)
    status_path = output/'metadata.json'
    if status_path.exists():
        old = json.loads(status_path.read_text())
        if any(old.get(key) != value for key, value in config.items()):
            raise ValueError('Visibility output has different inputs/settings; choose a new directory')
    atomic_json(status_path, dict(config, complete=False))
    points = read_points(args.point_cloud)
    raycaster = scene_raycaster(scene, mesh, threads=args.threads) if visibility_backend == 'exact' else None
    provider = MeshDepthCache(args.scene)
    preview_names = {all_names[i] for i in np.linspace(0, len(all_names)-1, min(16, len(all_names)), dtype=int)}
    if args.image:
        preview_names.add(args.image)
    if '07519.jpg' in names:
        preview_names.add('07519.jpg')
    rows = {}
    for name in tqdm(names, desc='Caching camera depth and LiDAR visibility'):
        camera = cameras[name]
        signature = camera_signature(camera, 1)
        path = output/'frames'/frame_filename(name)
        if path.exists():
            with np.load(path, allow_pickle=False) as previous:
                if str(previous['camera_signature']) != signature:
                    raise ValueError(f'Camera changed: {name}; choose a fresh visibility output')
                rows[name] = json.loads(str(previous['report']))
            continue
        k, e = camera_intrinsics(camera, 'cpu').numpy(), camera.world_view_transform.T.numpy()
        depth = provider.raw_depth(camera, args.device)
        if visibility_backend == 'cuda':
            from utils.ray_occlusion import classify_points_depth
            labels = classify_points_depth(points, k, e, depth, args.margin, scene['relative_margin'])
            mapping = dict(visible='on_surface', occluded='culled', unknown='unknown', in_front='in_front')
            id_type = np.uint32 if len(points) <= np.iinfo(np.uint32).max else np.uint64
            ids = {name+'_ids': labels[value+'_ids'].astype(id_type) for name,value in mapping.items()}
        else:
            ids = classify_lidar(points, k, e, camera.image_width, camera.image_height, raycaster,
                args.margin, scene['relative_margin'], args.device)
        tiles = provider.tiles(camera, args.device)
        counts = {key: len(ids[key+'_ids']) for key in POINT_CLASSES}
        in_view = sum(counts.values())
        counts.update(in_view=in_view, outside_or_invalid=len(points)-in_view, total=len(points))
        row = dict(image=name, counts=counts, depth_coverage=float(torch.isfinite(depth).float().mean()),
            usable_tile_fraction=float(torch.isfinite(tiles).float().mean()),
            occluded_fraction_in_view=counts['occluded']/max(in_view, 1))
        save_npz(path, **ids, image=name, intrinsic=k, extrinsic=e, width=camera.image_width,
            height=camera.image_height, camera_signature=signature, report=json.dumps(row))
        rows[name] = row
        print(f"{name}: depth {row['depth_coverage']:.1%}, usable tiles {row['usable_tile_fraction']:.1%}; "
              f"LiDAR visible {counts['visible']:,}, occluded {counts['occluded']:,}, "
              f"unknown {counts['unknown']:,}, in front {counts['in_front']:,}", flush=True)
        if name in preview_names:
            with Image.open(records[name]['image_path']) as im:
                rgb = np.asarray(im.convert('RGB').resize((camera.image_width, camera.image_height)))
            panels, coverage = reference_panels(provider.diagnostic_frame(camera, args.device), k, rgb, tiles.cpu().numpy())
            panels['gt'] = ('Actual image', rgb)
            write_report(output/'previews'/Path(frame_filename(name)).stem, {**row, 'coverage': coverage}, panels)
    # Recover completed rows across interrupted/one-view runs; never treat a subset as complete.
    for name in all_names:
        path = output/'frames'/frame_filename(name)
        if name not in rows and path.exists():
            with np.load(path, allow_pickle=False) as data:
                rows[name] = json.loads(str(data['report']))
    complete = len(rows) == len(all_names)
    atomic_json(status_path, dict(config, complete=complete, prepared=len(rows), total_points=len(points)))
    atomic_json(output/'summary.json', dict(complete=complete, frames=[rows[n] for n in all_names if n in rows]))
    lines = []
    for name in all_names:
        if name not in rows:
            continue
        row = rows[name]
        cells = [html.escape(name), f"{row['depth_coverage']:.1%}", f"{row['usable_tile_fraction']:.1%}"]
        cells += [f"{row['counts'][key]:,}" for key in POINT_CLASSES]
        lines.append('<tr>'+''.join('<td>'+cell+'</td>' for cell in cells)+'</tr>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Scene visibility audit</title>'
        '<style>body{font:16px sans-serif;margin:30px}td,th{padding:8px;text-align:right;border-bottom:1px solid #ccc}</style>'
        '<h1>Fixed scene surface / original LiDAR visibility</h1><p>Unknown intersections stay unknown. '
        'These point labels are not Gaussian visibility masks after movement or densification.</p><table><tr>'
        '<th>Image</th><th>Depth coverage</th><th>Usable tiles</th><th>Visible</th><th>Occluded</th>'
        '<th>Unknown</th><th>In front</th></tr>'+''.join(lines)+'</table>')
    print(f'Visibility audit: {output}/index.html ({len(rows)}/{len(all_names)} frames)', flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='stage', required=True)
    f = sub.add_parser('fuse', help='Fuse all trusted training depths into one fixed scene mesh (CPU)')
    f.add_argument('--depth-dir', type=Path, required=True)
    f.add_argument('--point-cloud', type=Path, required=True)
    f.add_argument('--output', type=Path, required=True)
    f.add_argument('--voxel-size', type=float, default=.05)
    f.add_argument('--truncation', type=float, default=.15)
    f.add_argument('--depth-max', type=float, default=60.)
    f.add_argument('--lidar-support-distance', type=float, default=.20)
    f.add_argument('--relative-margin', type=float, default=.10)
    f.add_argument('--threads', type=int, default=8)
    f.add_argument('--allow-partial', action='store_true', help='Diagnostic multi-frame mesh; cannot be used for training')
    v = sub.add_parser('views', help='Precompute all registered camera views, using the fixed training mesh')
    v.add_argument('--scene', type=Path, required=True)
    v.add_argument('--point-cloud', type=Path, required=True)
    v.add_argument('--output', type=Path)
    v.add_argument('--resolution', type=int, default=-1)
    v.add_argument('--gpu', default='0')
    v.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    v.add_argument('--visibility-backend', choices=('auto', 'cuda', 'exact'), default='auto',
                   help='auto uses CUDA pixel-depth classification on GPU; exact uses CPU triangle BVH rays')
    v.add_argument('--split', choices=('all', 'train'), default='all')
    v.add_argument('--image', help='Inspect one view first; omit to finish all views')
    v.add_argument('--margin', type=float, default=.05)
    v.add_argument('--threads', type=int, default=8)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.threads < 1:
        raise ValueError('threads must be positive')
    if args.stage == 'fuse':
        values = [args.voxel_size, args.truncation, args.depth_max, args.lidar_support_distance, args.relative_margin]
        if (not np.isfinite(values).all() or min(values[:4]) <= 0 or args.relative_margin < 0 or
                args.truncation < 2*args.voxel_size):
            raise ValueError('Use positive finite fusion settings, truncation >= 2 voxels, and nonnegative margin')
        fuse(args)
    else:
        if args.resolution == 0 or args.resolution < -1 or not np.isfinite(args.margin) or args.margin < 0:
            raise ValueError('Invalid visibility resolution/margin')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        precompute_views(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
