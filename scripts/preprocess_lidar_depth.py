#!/usr/bin/env python3
"""Project registered LiDAR points to sparse training samples with coarse z-buffering."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils.colmap_paths import colmap_model_dir
from utils.sparse_lidar import frame_filename


def image_resolution(width, height, resolution):
    if resolution in (1, 2, 4, 8):
        return round(width / resolution), round(height / resolution)
    down = (width / 1600 if width > 1600 else 1) if resolution == -1 else width / resolution
    return int(width / down), int(height / down)


def project(points, rotation, translation, fx, fy, width, height, cell_size):
    import torch
    xyz = points @ rotation.T + translation
    z = xyz[:, 2]
    # Graphdeco's getProjectionMatrix is centered; ndc2Pix uses (size-1)/2.
    x = fx * xyz[:, 0] / z + (width - 1) * 0.5
    y = fy * xyz[:, 1] / z + (height - 1) * 0.5
    good = ((z > 0.2) & torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(z)
            & (x >= -0.5) & (x < width - 0.5) & (y >= -0.5) & (y < height - 0.5))
    x = torch.floor(x[good] + 0.5).long()
    y = torch.floor(y[good] + 0.5).long()
    z = z[good]
    cells_w = (width + cell_size - 1) // cell_size
    cells_h = (height + cell_size - 1) // cell_size
    cell = (y // cell_size) * cells_w + x // cell_size
    front = torch.full((cells_w * cells_h,), float('inf'), device=points.device)
    front.scatter_reduce_(0, cell, z, reduce='amin', include_self=True)
    keep = z <= front[cell] + 1e-6
    return x[keep], y[keep], z[keep]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--keyframes', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--point-cloud', type=Path)
    parser.add_argument('--gpu', default='6')
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--cell-size', type=int, default=4)
    args = parser.parse_args()
    if args.cell_size < 1 or args.resolution == 0 or args.resolution < -1:
        parser.error('cell size must be positive and resolution must be -1 or positive')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    from plyfile import PlyData
    from PIL import Image
    from tqdm import tqdm
    from scene.colmap_loader import (read_extrinsics_binary, read_intrinsics_binary,
                                     read_extrinsics_text, read_intrinsics_text, qvec2rotmat)
    if not torch.cuda.is_available():
        parser.error(f'GPU {args.gpu} not available')
    source = args.source.expanduser().resolve()
    model = colmap_model_dir(source)
    cloud = args.point_cloud.expanduser().resolve() if args.point_cloud else model / 'points3D.ply'
    selected = json.loads(args.keyframes.read_text())['selected_images']
    if not selected or len(selected) != len(set(selected)):
        parser.error('Keyframe manifest is empty or contains duplicates')
    if (model / 'images.bin').is_file() and (model / 'cameras.bin').is_file():
        cams = read_extrinsics_binary(str(model / 'images.bin'))
        intrinsics = read_intrinsics_binary(str(model / 'cameras.bin'))
    else:
        cams = read_extrinsics_text(str(model / 'images.txt'))
        intrinsics = read_intrinsics_text(str(model / 'cameras.txt'))
    images = {c.name: c for c in cams.values()}
    if set(selected) - images.keys():
        parser.error('Some keyframes are absent from COLMAP')
    config = {'version': 1, 'source': str(source), 'point_cloud': str(cloud),
              'point_cloud_size': cloud.stat().st_size, 'point_cloud_mtime_ns': cloud.stat().st_mtime_ns,
              'keyframes_sha256': hashlib.sha256(args.keyframes.read_bytes()).hexdigest(),
              'resolution': args.resolution, 'cell_size': args.cell_size, 'images': selected,
              'projection': 'centered Graphdeco camera; frontmost point per coarse cell'}
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    meta = out / 'metadata.json'
    if meta.exists():
        existing = json.loads(meta.read_text())
        if {k: existing.get(k) for k in config} != config:
            parser.error('Existing LiDAR depth directory has different inputs or settings')
    elif any(out.iterdir()):
        parser.error('Output directory contains unrecognized files')
    meta.write_text(json.dumps({**config, 'complete': False}, indent=2) + '\n')
    vertex = PlyData.read(str(cloud))['vertex'].data
    p = np.stack([vertex[k] for k in ('x', 'y', 'z')], axis=1).astype(np.float32)
    p = p[np.isfinite(p).all(axis=1)]
    if not len(p):
        parser.error('Point cloud has no finite points')
    points = torch.from_numpy(p).to('cuda')
    print(f'Projecting {len(p):,} LiDAR points onto {len(selected):,} keyframes', flush=True)
    sample_counts = []
    with torch.inference_mode():
        for name in tqdm(selected, desc='Projecting LiDAR', unit='frame'):
            destination = out / frame_filename(name)
            if destination.exists():
                with np.load(destination, allow_pickle=False) as previous:
                    if str(previous['image']) != name or not len(previous['z']):
                        raise ValueError(f'Invalid existing sparse depth frame: {destination}')
                    sample_counts.append(len(previous['z']))
                continue
            cam = images[name]
            intr = intrinsics[cam.camera_id]
            if intr.model == 'PINHOLE':
                fx, fy = intr.params[:2]
            elif intr.model == 'SIMPLE_PINHOLE':
                fx = fy = intr.params[0]
            else:
                raise ValueError(f'Unsupported distorted COLMAP camera: {intr.model}')
            with Image.open(source / 'images' / name) as image:
                orig_w, orig_h = image.size
            if abs(orig_w / orig_h - intr.width / intr.height) > 1e-3:
                raise ValueError(f'Image aspect ratio differs from COLMAP: {name}')
            w, h = image_resolution(orig_w, orig_h, args.resolution)
            rotation = torch.as_tensor(qvec2rotmat(cam.qvec), device='cuda', dtype=torch.float32)
            translation = torch.as_tensor(cam.tvec, device='cuda', dtype=torch.float32)
            u, v, z = project(points, rotation, translation, fx*w/intr.width, fy*h/intr.height, w, h, args.cell_size)
            if z.numel() == 0:
                raise ValueError(f'No LiDAR depth samples for {name}')
            sample_counts.append(z.numel())
            temporary = destination.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, image=name, shape=np.array([h, w]),
                                u=u.cpu().numpy().astype(np.int32), v=v.cpu().numpy().astype(np.int32),
                                z=z.cpu().numpy().astype(np.float32))
            temporary.replace(destination)
    summary = {'min': min(sample_counts), 'median': float(np.median(sample_counts)), 'max': max(sample_counts)}
    meta.write_text(json.dumps({**config, 'complete': True, 'points_used': len(p), 'samples_per_frame': summary}, indent=2) + '\n')
    print(f'Samples per frame: {summary}', flush=True)
    print(f'Sparse LiDAR depth ready: {out}')


if __name__ == '__main__':
    main()
