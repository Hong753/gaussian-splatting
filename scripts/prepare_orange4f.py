#!/usr/bin/env python3
"""One immutable sharp-stride3 selection and initialization for four no-AA runs."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.keyframes import select_keyframes
from scripts.view_comparisons import read_colmap_records
from scripts import run_native_lidar as native, run_soft_surface_long as periodic
from utils.colmap_paths import colmap_model_dir
from utils.checkpoint_snapshots import atomic_json, digest

RUNS = ('3dgs', '3dgs_fixed_lidar', 'gsprior', 'gsprior_fixed_lidar')
DEFAULT_OUTPUT_ROOT = '/data/3dgs_outputs/orange4f/3dgs_finetuning'


def point_records(path):
    """Stream COLMAP points, skipping tracks instead of allocating observations."""
    if path.suffix == '.txt':
        with path.open() as stream:
            for line in stream:
                if not line.strip() or line.startswith('#'): continue
                row = line.split(maxsplit=8)
                yield tuple(map(float, row[1:4])) + tuple(map(int, row[4:7]))
    else:
        with path.open('rb') as stream:
            count = struct.unpack('<Q', stream.read(8))[0]
            for _ in range(count):
                row = struct.unpack('<QdddBBBd', stream.read(43))
                n = struct.unpack('<Q', stream.read(8))[0]
                stream.seek(8*n, 1)
                yield row[1:7]


def canonical_cloud(source, output, explicit=None):
    import numpy as np
    from plyfile import PlyData, PlyElement
    sparse = colmap_model_dir(source)
    candidates = [explicit] if explicit else [source/'input.ply', sparse/'points3D.ply', sparse/'points3D.bin', sparse/'points3D.txt']
    path = next((p.resolve() for p in candidates if p and p.is_file()), None)
    if path is None: raise FileNotFoundError('No initialization cloud; provide --point-cloud /path/to/LiDAR.ply')
    dtype = np.dtype([(k, '<f4') for k in ('x','y','z','nx','ny','nz')] + [(k, 'u1') for k in ('red','green','blue')])
    if path.suffix == '.ply':
        vertices = PlyData.read(path, mmap='r')['vertex'].data
        fields = set(vertices.dtype.names)
        if not {'x','y','z','red','green','blue'} <= fields or 'scale_0' in fields or 'f_dc_0' in fields:
            raise ValueError('Initialization must contain raw XYZ/RGB, not learned Gaussians')
        if set(dtype.names) <= fields:
            output.symlink_to(path)
        else:
            data = np.zeros(len(vertices), dtype=dtype)
            for name in dtype.names:
                if name in fields: data[name] = vertices[name]
            PlyData([PlyElement.describe(data, 'vertex')], byte_order='<').write(output)
    else:
        if path.suffix == '.txt':
            count = sum(1 for _ in point_records(path))
        else:
            with path.open('rb') as stream: count = struct.unpack('<Q', stream.read(8))[0]
        data = np.zeros(count, dtype=dtype)
        print(f'Converting {count:,} points from {path} into {output}', flush=True)
        for i, row in enumerate(point_records(path)):
            for key, value in zip(('x','y','z','red','green','blue'), row): data[key][i] = value
        PlyData([PlyElement.describe(data, 'vertex')], byte_order='<').write(output)
    vertices = PlyData.read(output, mmap='r')['vertex'].data
    if not len(vertices): raise ValueError('Empty initialization')
    lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
    for start in range(0, len(vertices), 262144):
        xyz = np.column_stack([vertices[k][start:start+262144] for k in ('x','y','z')])
        if not np.isfinite(xyz).all(): raise ValueError('Nonfinite initialization positions')
        lower = np.minimum(lower, xyz.min(0)); upper = np.maximum(upper, xyz.max(0))
    return dict(original=str(path), cached=str(output), points=len(vertices),
                sha256=digest(output), bounds=[lower.tolist(), upper.tolist()])


def grid_geometry(bounds, cameras, requested_voxel=.02, max_voxels=64000000):
    import numpy as np
    low, high = np.array(bounds, dtype=float)
    centers = np.array([-np.array(c['R'])@np.array(c['T']) for c in cameras.values()])
    low = np.minimum(low, centers.min(0)); high = np.maximum(high, centers.max(0))
    voxel = requested_voxel
    while True:
        origin = low-2*voxel
        dims = np.maximum(2, np.ceil((high-low)/voxel).astype(np.int64)+5)
        if math.prod(map(int, dims)) <= max_voxels: break
        voxel *= max(1.01, (math.prod(map(int, dims))/max_voxels)**(1/3))
    return dict(origin=origin.tolist(), dimensions=dims.tolist(), voxel_size=float(voxel),
                bounds=np.column_stack((origin, origin+dims*voxel)).tolist(),
                voxels=math.prod(map(int, dims)), max_voxels=max_voxels,
                requested_voxel_size=requested_voxel,
                truncation=[float(voxel*k) for k in (96,48,24)])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('/workspace/colmap_scenes/orange4f'))
    p.add_argument('--output', type=Path)
    p.add_argument('--output-root', type=Path,
                   default=Path(os.environ.get('ORANGE4F_OUTPUT_ROOT', DEFAULT_OUTPUT_ROOT)),
                   help='Experiment parent directory, independent of the read-only dataset')
    p.add_argument('--point-cloud', type=Path)
    p.add_argument('--version', default='v1')
    p.add_argument('--prior-voxel-size', type=float, default=.02)
    p.add_argument('--prior-max-voxels', type=int, default=64000000)
    args = p.parse_args(argv)
    source = args.source.expanduser().resolve()
    output = (args.output or args.output_root/f'stride3_noaa_{args.version}').expanduser().resolve()
    if args.prior_voxel_size <= 0 or args.prior_max_voxels < 1000: p.error('Invalid TSDF limits')
    if (output/'suite.json').exists():
        old = json.loads((output/'suite.json').read_text())
        if old['source'] != str(source): raise ValueError('Suite belongs to another dataset')
        if args.point_cloud and str(args.point_cloud.expanduser().resolve()) != old['cloud']['original']:
            raise ValueError('Initialization changed; use a new suite version')
        if (old['grid']['requested_voxel_size'], old['grid']['max_voxels']) != (args.prior_voxel_size,args.prior_max_voxels):
            raise ValueError('TSDF configuration changed; use a new suite version')
        for name, expected in old['input_fingerprints'].items():
            if digest(Path(name)) != expected: raise ValueError(f'Prepared input changed: {name}')
        print(f'Reusing prepared suite: {output}', flush=True); return 0
    if output.exists(): raise FileExistsError(f'Incomplete/existing output: {output}; use a fresh --version')
    records = read_colmap_records(source)
    if len(records)<4: raise ValueError('Need at least four registered images')
    missing = [r['image_path'] for r in records.values() if not Path(r['image_path']).is_file()]
    if missing: raise FileNotFoundError(missing[:5])
    shared = output/'shared'; shared.mkdir(parents=True)
    manifest = select_keyframes(source, 'images', shared, 3, False)
    selected = json.loads(manifest.read_text())['selected_images']
    cloud = canonical_cloud(source, shared/'input.ply', args.point_cloud)
    grid = grid_geometry(cloud['bounds'], records, args.prior_voxel_size, args.prior_max_voxels)
    # A separate read-only view of the dataset supports upstream GSPrior's layout.
    data = shared/'gsprior_data'; (data/'sparse').mkdir(parents=True)
    (data/'images').symlink_to(source/'images', target_is_directory=True)
    sparse = colmap_model_dir(source)
    fingerprints = {str(manifest):digest(manifest), str(shared/'input.ply'):cloud['sha256']}
    for kind in ('cameras','images'):
        ext = 'bin' if (sparse/(kind+'.bin')).exists() else 'txt'
        fingerprints[str(sparse/(kind+'.'+ext))] = digest(sparse/(kind+'.'+ext))
        (data/'sparse'/(kind+'.'+ext)).symlink_to(sparse/(kind+'.'+ext))
    (data/'sparse/points3D.ply').symlink_to(shared/'input.ply')
    atomic_json(data/'split.json', dict(train=selected, test=sorted(set(records)-set(selected))))
    entries = []
    for gpu, name in enumerate(RUNS):
        model = output/name
        fixed = gpu in (1,3)
        entry = dict(name=name, gpu=str(gpu), model=str(model), fixed_positions=fixed,
                     antialiasing=False, iterations=30000 if gpu<2 else 40000)
        if gpu<2:
            model.mkdir()
            shutil.copyfile(manifest, model/'keyframes.json')
            command = ['--source',str(source),'--model',str(model),'--keyframes',str(model/'keyframes.json'),
                '--point-cloud',str(shared/'input.ply'),'--gpu',str(gpu),'--iterations','30000',
                '--expected-frames',str(len(records)),'--resolution','1','--no-antialiasing',
                '--position-lr-divisor','1','--pruning','--opacity-resets','--feature-lr','.0025',
                '--opacity-lr','.05','--scaling-lr','.005','--rotation-lr','.001',
                '--percent-dense','.01', *(['--freeze-lidar-positions'] if fixed else [])]
            options = native.parser().parse_args(command)
            plan = native.build_plan(options)
            periodic.prepare_experiment(options, plan, 'native_lidar',
                [model/'keyframes.json', shared/'input.ply', *[x for x in sparse.iterdir() if x.name in ('images.bin','images.txt','cameras.bin','cameras.txt')]])
        else:
            model.mkdir()
            shutil.copyfile(manifest, model/'keyframes.json')
            atomic_json(model/'gsprior_run.json', dict(**entry, source=str(data), original_source=str(source),
                cloud=cloud, grid=grid, tsdf_iterations=[20000,30000,40000], checkpoints=[10000,20000,30000,40000],
                preview_every=500, all_frames=len(records), train_frames=len(selected),
                max_all_points=max(6000000, math.ceil(1.5*cloud['points'])),
                input_fingerprints=fingerprints,
                upstream='takeshie/GSPrior', upstream_commit='dcb7c89fb6b60f068b440de45d064ecc7fbcba55'))
        entries.append(entry)
    suite = dict(source=str(source), output=str(output), antialiasing=False, cloud=cloud,
                 input_fingerprints=fingerprints,
                 frames=len(records), train_frames=len(selected), heldout_frames=len(records)-len(selected),
                 selection=str(manifest), grid=grid, runs=entries)
    atomic_json(output/'suite.json', suite)
    links = ''.join(f'<li>GPU {r["gpu"]}: <a href="{r["name"]}/debug_previews/index.html">{r["name"]}</a></li>' for r in entries)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>orange4f · stride 3 · AA off</h1><ul>'+links+'</ul>')
    print(json.dumps(suite, indent=2), flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
