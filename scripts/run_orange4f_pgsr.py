#!/usr/bin/env python3
"""Train and compare PGSR with a fixed LiDAR warmup and optional tangent guidance."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_all_frames import read_config
from scripts.orange4f_detail import experiment_lock, select_gpu
from scripts.prepare_orange4f import DEFAULT_OUTPUT_ROOT, main as prepare_suite
from scripts.view_comparisons import read_colmap_records, split_names
from utils.checkpoint_snapshots import atomic_json, digest, read_snapshot
from utils.colmap_paths import colmap_model_dir
from scripts.pgsr_lidar import WARMUP, RAMP, STEPS, NORMAL_STEP

PIN = 'de24f1a38b350387e8d8fe381b2cd70c1ae946e7'
PATCH = ROOT/'scripts/pgsr_orange4f.patch'
DEPTH_PATCH = ROOT/'scripts/pgsr_depth.patch'


def inputs(suite, fresh_source):
    fixed = suite/'3dgs_fixed_lidar'
    if (fixed/'cfg_args').is_file():
        cfg = read_config(fixed/'cfg_args')
        return Path(cfg['source_path']), cfg.get('images') or 'images', fixed/'keyframes.json', fixed/'input.ply'
    suite.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(suite)+'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (suite/'suite.json').is_file():
            argv = ['--source', str(fresh_source), '--output', str(suite)]
            if os.environ.get('ORANGE4F_POINT_CLOUD'):
                argv += ['--point-cloud', os.environ['ORANGE4F_POINT_CLOUD']]
            prepare_suite(argv)
    info = json.loads((suite/'suite.json').read_text())
    return Path(info['source']), 'images', Path(info['selection']), Path(info['cloud']['cached'])


def protocol(args):
    from plyfile import PlyData
    suite = args.suite.expanduser().resolve()
    source, images, manifest, cloud = inputs(suite, args.source)
    records = read_colmap_records(source, images)
    selected = split_names(records, {}, json.loads(manifest.read_text()))['train']
    missing = next((r['image_path'] for r in records.values() if not Path(r['image_path']).is_file()), None)
    if missing:
        raise FileNotFoundError(missing)
    sparse = colmap_model_dir(source)
    ext = 'bin' if all((sparse/(k+'.bin')).is_file() for k in ('images', 'cameras')) else 'txt'
    depth = None
    suffix = ''
    if getattr(args, 'depth_supervision', False) or getattr(args, 'scan_manifest', None):
        from scripts.pgsr_depth import depth_config
        depth = depth_config(source, getattr(args, 'scan_manifest', None))
        suffix = '_' + depth['mode'] + '_depth'
    model = (args.output or suite.parent/f'pgsr_lidar_{args.variant}{suffix}_50k_v1').expanduser().resolve()
    source_dir = model/'dataset'
    links = {source_dir/'images': (source/images).resolve(), source_dir/'sparse/points3D.ply': cloud.resolve()}
    for k in ('images', 'cameras'):
        links[source_dir/'sparse'/(k+'.'+ext)] = sparse/(k+'.'+ext)
    vertices = PlyData.read(cloud, mmap='r')['vertex']
    if vertices.count < 24 or not {'x','y','z','red','green','blue'} <= set(vertices.data.dtype.names):
        raise ValueError(f'Expected a colored raw LiDAR cloud with at least 24 points at {cloud}')
    fingerprints = {str(p): digest(p) for p in [manifest, *[p for p in links.values() if p.is_file()]]}
    implementation = {name: digest(ROOT/'scripts'/name)
                      for name in ('pgsr_entry.py', 'pgsr_lidar.py')}
    cfg = dict(implementation=implementation, variant=args.variant, model=str(model), source=str(source_dir), original_source=str(source.resolve()),
               initial_points=vertices.count, all_frames=len(records), train_frames=len(selected),
               max_all_points=max(6000000, math.ceil(1.5*vertices.count)), upstream_commit=PIN,
               fingerprints=fingerprints, patch_sha256=digest(PATCH), warmup=WARMUP, ramp=RAMP,
               position_lr_endpoints=[3.2e-6, 3.2e-8], normal_step=NORMAL_STEP if args.variant=='guided' else 1.,
               scale_cap_over_initial_sigma=2., checkpoints=STEPS, exposure=False, pose_refinement=False)
    if depth:
        cfg['depth_supervision'] = depth
        implementation['pgsr_depth.py'] = digest(ROOT/'scripts/pgsr_depth.py')
        implementation['pgsr_depth.patch'] = digest(DEPTH_PATCH)
    # JSON round-trip gives identical checkpoint schedule types on restart.
    return json.loads(json.dumps(cfg)), links, manifest, dict(train=sorted(selected), test=sorted(set(records)-selected))


def setup_runtime():
    import torch
    import simple_knn._C as knn
    if not torch.cuda.is_available() or not callable(getattr(knn, 'distCUDA2', None)):
        raise RuntimeError('Activate the CUDA 3dgs environment with its existing simple_knn')
    architecture = torch.cuda.get_device_capability()
    runtime = (PIN, digest(PATCH), sys.executable, torch.__version__, torch.version.cuda,
               architecture, os.environ.get('TORCH_CUDA_ARCH_LIST'))
    key = hashlib.sha256(repr(runtime).encode()).hexdigest()[:16]
    cache = Path.home()/'.cache/3dgs'
    cache.mkdir(parents=True, exist_ok=True)
    checkout = cache/f'pgsr-{key}'
    print(f'Reusing simple_knn from {knn.__file__}', flush=True)
    with checkout.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not checkout.is_dir():
            with tempfile.TemporaryDirectory(dir=cache) as temp:
                build = Path(temp)/'build'
                src = build/'src'
                subprocess.run(['git', 'clone', 'https://github.com/zju3dv/PGSR.git', str(src)], check=True)
                subprocess.run(['git', 'checkout', '--detach', PIN], cwd=src, check=True)
                subprocess.run(['git', 'apply', '--unidiff-zero', str(PATCH)], cwd=src, check=True)
                subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-build-isolation', '--no-deps',
                                '--target', str(build/'extension'), str(src/'submodules/diff-plane-rasterization')],
                               env={**os.environ, 'MAX_JOBS':os.environ.get('MAX_JOBS', '8')}, check=True)
                build.rename(checkout)
    return checkout


def depth_runtime(base):
    # Reuse the installed rasterizer and the frozen upstream checkout. Only two
    # Python files need a private copy for the depth experiment.
    checkout = base.with_name(base.name+'-depth-'+digest(DEPTH_PATCH)[:12])
    with checkout.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not checkout.exists():
            with tempfile.TemporaryDirectory(dir=base.parent) as temp:
                root = Path(temp)/'build'
                src = root/'src'
                src.mkdir(parents=True)
                for item in (base/'src').iterdir():
                    if item.name == 'train.py':
                        shutil.copyfile(item, src/item.name)
                    elif item.name == 'gaussian_renderer':
                        shutil.copytree(item, src/item.name, ignore=shutil.ignore_patterns('__pycache__'))
                    elif item.name not in ('.git', '__pycache__'):
                        (src/item.name).symlink_to(item)
                subprocess.run(['git', 'apply', '--unidiff-zero', str(DEPTH_PATCH)], cwd=src, check=True)
                shutil.copytree(base/'extension', root/'extension', copy_function=os.link)
                root.rename(checkout)
    return checkout


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=('guided', 'covariance'), default='guided')
    p.add_argument('--suite', type=Path, default=Path(os.environ.get('ORANGE4F_OUTPUT_ROOT', DEFAULT_OUTPUT_ROOT))/'stride3_noaa_v1')
    p.add_argument('--source', type=Path, default=Path('/workspace/colmap_scenes/orange4f'), help='Source for a fresh suite')
    p.add_argument('--output', type=Path)
    p.add_argument('--gpu', help='Physical GPU. Default honors CUDA_VISIBLE_DEVICES or selects a free GPU')
    p.add_argument('--depth-supervision', action='store_true',
                   help='Add scanner depth if source/lidar_scans.json exists, otherwise use front-surface fused-cloud targets')
    p.add_argument('--scan-manifest', type=Path,
                   help='Explicit scanner_xyz_v1 manifest. Enables measured scanner depth; see scripts/pgsr_depth.md')
    p.add_argument('--plan-only', action='store_true')
    args = p.parse_args(argv)
    cfg, links, manifest, split = protocol(args)
    print(json.dumps(cfg, indent=2), flush=True)
    if args.plan_only:
        return
    model = Path(cfg['model'])
    with experiment_lock(model) as lock:
        plan = model/'pgsr_run.json'
        if model.exists() and (not plan.is_file() or json.loads(plan.read_text()) != cfg):
            raise FileExistsError(f'Output belongs to another protocol: {model}')
        atomic_json(plan, cfg)
        for target, original in links.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink() and target.resolve() == original.resolve():
                continue
            target.symlink_to(original)
        atomic_json(Path(cfg['source'])/'split.json', split)
        shutil.copyfile(manifest, model/'keyframes.json')
        gpu = select_gpu(args.gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        print(f'Using CUDA device {gpu}.', flush=True)
        checkout = setup_runtime()
        if 'depth_supervision' in cfg:
            checkout = depth_runtime(checkout)
        done = [i for i in STEPS if read_snapshot(model, i)]
        entry = [sys.executable, '-u', str(ROOT/'scripts/pgsr_entry.py')]
        def run(mode, step):
            subprocess.run([*entry, mode, str(checkout), str(plan), str(step)], cwd=checkout/'src',
                           check=True, pass_fds=(lock.fileno(),))
        if STEPS[-1] not in done:
            run('train', max(done, default=0))
        for step in STEPS[1:]:
            if not (model/'psnr_all_frames'/f'iteration_{step}'/'summary.json').is_file():
                run('eval', step)
            if 'depth_supervision' in cfg:
                from scripts.pgsr_depth_eval import compare
                baseline = args.suite.expanduser().resolve().parent/f'pgsr_lidar_{args.variant}_50k_v1'
                compare(model, baseline, step)
        print(model/'view_comparisons'/f'iteration_{STEPS[-1]}'/'index.html', flush=True)


if __name__ == '__main__':
    main()
