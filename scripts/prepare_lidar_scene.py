#!/usr/bin/env python3
"""Build the primary occluding surface directly from the full initialization LiDAR."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_scene_occlusion import read_points
from scripts.prepare_da3_occlusion import save_npz
from scripts.view_comparisons import atomic_json, read_colmap_records
from utils.lidar_occlusion import file_digest
from utils.scene_occlusion import FORMAT, load_scene


def prepare(args):
    from utils.lidar_surface import build_lidar_mesh
    from utils.occlusion_depth import camera_signature, record_camera
    source, cloud, output = args.source.resolve(), args.point_cloud.resolve(), args.output.resolve()
    names = sorted(json.loads(args.keyframes.read_text())['selected_images'])
    records = read_colmap_records(source)
    if not names or len(set(names)) != len(names) or not set(names).issubset(records):
        raise ValueError('Keyframes must be unique registered camera names')
    cloud_hash = file_digest(cloud)
    config = dict(format=FORMAT, builder_version=1, geometry_source='full_initialization_lidar',
        source=str(source), training_images=names, resolution=args.resolution,
        point_cloud=str(cloud), point_cloud_sha256=cloud_hash,
        voxel_size=args.voxel_size, max_edge=args.max_edge, support_distance=args.support_distance,
        normal_radius=args.normal_radius, block_size=args.block_size,
        max_block_points=args.max_block_points, relative_margin=args.relative_margin)
    supplement = None
    if args.supplement_scene:
        extra, supplement = load_scene(args.supplement_scene, cloud, source)
        if extra.get('diagnostic_only') or set(extra['training_images']) != set(names):
            raise ValueError('A DA3 supplement must be a complete validated mesh for these training views')
        if extra.get('geometry_source') == 'full_initialization_lidar' or extra.get('supplement_mesh'):
            raise ValueError('Supply the validated DA3-only scene as the supplement, not another combined scene')
        if not extra.get('depth_frames_sha256'):
            raise ValueError('Supplement is missing validated depth-frame provenance')
        config.update(supplement_mesh='da3_supplement.npz', supplement_mesh_sha256=extra['mesh_sha256'],
            supplement_scene_sha256=file_digest(args.supplement_scene),
            supplement_policy='missing_lidar_rays_only')
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/'scene.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if any(old.get(k) != v for k, v in config.items() if k != 'point_cloud') or (
                bool(old.get('supplement_mesh')) != bool(supplement)):
            raise ValueError('Inputs/settings changed; use a new LiDAR scene output directory')
        if old.get('complete'):
            load_scene(manifest, cloud, source)
            print(f'Reusing full LiDAR surface: {manifest}', flush=True)
            return manifest
    config['camera_signatures'] = {name: camera_signature(record_camera(records[name], args.resolution, 'cpu'), 1)
                                    for name in names}
    atomic_json(manifest, dict(config, complete=False))
    points = read_points(cloud)
    print(f'Meshing ALL {len(points):,} initialization LiDAR points (CPU, bounded spatial blocks). '
          'DA3 does not gate measured surfaces.', flush=True)
    def progress(index, total, count, faces):
        print(f'LiDAR surface block {index}/{total}: {count:,} points, {faces:,} candidate triangles', flush=True)
    vertices, faces, row_ids, unmeshed_ids, report = build_lidar_mesh(points,
        voxel_size=args.voxel_size, max_edge=args.max_edge, support_distance=args.support_distance,
        normal_radius=args.normal_radius, block_size=args.block_size,
        max_block_points=args.max_block_points, threads=args.threads, progress=progress)
    save_npz(output/'mesh.npz', vertices=vertices, faces=faces, source_row_ids=row_ids)
    save_npz(output/'unmeshed_lidar.npz', original_point_ids=unmeshed_ids)
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
    mesh.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(str(output/'mesh.ply'), mesh):
        raise RuntimeError('Failed to save the LiDAR surface mesh')
    if supplement:
        shutil.copyfile(supplement, output/config['supplement_mesh'])
    result = dict(config, complete=True, diagnostic_only=False, mesh='mesh.npz',
        mesh_sha256=file_digest(output/'mesh.npz'), input_frames=0, integrated_frames=0,
        vertices=len(vertices), triangles=len(faces), lidar_surface=report,
        policy='LiDAR mesh first; optional validated DA3 only where the LiDAR ray has no hit')
    atomic_json(manifest, result)
    print(f"LiDAR mesh: {len(faces):,} triangles; {report['meshed_representative_fraction']:.1%} of "
          f"measured representatives connected; {len(unmeshed_ids):,} unmeshed.\n{manifest}", flush=True)
    print('Connected-point fraction is not image coverage. Check first-hit depth against foreground LiDAR per view.', flush=True)
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--keyframes', type=Path, required=True)
    p.add_argument('--point-cloud', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resolution', type=int, default=-1)
    p.add_argument('--voxel-size', type=float, default=.05)
    p.add_argument('--max-edge', type=float, default=.20)
    p.add_argument('--support-distance', type=float, default=.075)
    p.add_argument('--normal-radius', type=float, default=.15)
    p.add_argument('--block-size', type=float, default=4.)
    p.add_argument('--max-block-points', type=int, default=100000)
    p.add_argument('--relative-margin', type=float, default=0., help='LiDAR uses a metric margin, not a 10%% depth band')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--supplement-scene', type=Path, help='Optional validated DA3 scene; used only for rays missing LiDAR')
    args = p.parse_args(argv)
    numeric = [args.voxel_size, args.max_edge, args.support_distance, args.normal_radius, args.block_size]
    if (not np.isfinite(numeric).all() or min(numeric) <= 0 or args.threads < 1 or
            args.max_block_points < 32 or args.resolution == 0 or args.resolution < -1 or
            not np.isfinite(args.relative_margin) or args.relative_margin < 0):
        raise ValueError('Invalid meshing settings')
    os.environ.setdefault('OMP_NUM_THREADS', str(args.threads))
    prepare(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
