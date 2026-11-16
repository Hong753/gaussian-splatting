#!/usr/bin/env python3
"""Prepare a fresh copy of the exact active culling experiment, then stop only it."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_local_scale, run_soft_surface_long as periodic, stop_reconstruction_run as stopper
from utils.checkpoint_snapshots import atomic_json, digest

PATH_ARGS = ('source','model','baseline','keyframes','point_cloud','local_scale_cache',
             'reuse_cache','lidar_activation_cache','scene_geometry')


def validate_source(data, gpu):
    command = data['plan']['stages'][0]['argv']
    if (str(data['gpu']) != str(gpu) or data['experiment'] != 'local_scale' or
        '--occlusion_mode' not in command or command[command.index('--occlusion_mode')+1] != 'mesh'):
        raise ValueError('Expected the local-scale mesh-culling experiment on the requested GPU; no process stopped')


def choose_source(root, gpu, model=None):
    if model is not None:
        data = periodic.load_plan(model)
        validate_source(data, gpu)
        return data
    matches = []
    for path in sorted(Path(root).glob('*/periodic_evaluation/plan.json')):
        try:
            data = periodic.load_plan(path.parent.parent)
            validate_source(data, gpu)
        except (ValueError, KeyError, OSError):
            continue
        if stopper.candidates(data['model']):
            matches.append(data)
    if len(matches) != 1:
        raise ValueError(f'Expected one active GPU {gpu} culling run, found {len(matches)}. '
                         'Pass --model with its exact output directory. No process stopped.')
    return matches[0]


def prepare_restart(data, output, every=500):
    # Verify immutable source inputs BEFORE preparing a replacement or stopping.
    for path, expected in data['input_fingerprints'].items():
        if digest(path) != expected:
            raise ValueError(f'Original input changed: {path}; no process stopped')
    args = SimpleNamespace(**data['training_args'])
    for name in PATH_ARGS:
        if getattr(args, name, None) is not None:
            setattr(args, name, Path(getattr(args, name)))
    args.model = Path(output).expanduser().resolve()
    if args.model == Path(data['model']).resolve() or args.model.exists():
        raise ValueError('Restart requires a fresh output directory; original outputs are retained')
    args.dry_run = False
    args.debug_preview_every = every
    args.debug_preview_train = getattr(args, 'debug_preview_train', 2)
    args.debug_preview_test = getattr(args, 'debug_preview_test', 2)
    args.debug_preview_seed = getattr(args, 'debug_preview_seed', 0)
    # Cache paths are deliberately retained: identical fixed geometry / KNN /
    # neighbor preprocessing, with new learned weights and optimizer state.
    plan = run_local_scale.build_plan(args)
    periodic.prepare_experiment(args, plan, 'local_scale',
        [Path(path) for path in data['input_fingerprints']] + [periodic.directory(data['model'])/'plan.json'])
    receipt = dict(original=data['model'], model=str(args.model), gpu=str(args.gpu),
                   restart='fresh initialization; old weights and outputs retained', preview_every=every)
    atomic_json(args.model/'restart_from.json', receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/3dgs_outputs/orange_4F/1_full'))
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--every', type=int, default=500)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--stop', action='store_true')
    args = parser.parse_args(argv)
    if args.every <= 0:
        parser.error('Preview interval must be positive')
    data = choose_source(args.root, args.gpu, args.model)
    output = args.output or Path(data['model'] + f'_debug{args.every}_{time.time_ns()}')
    receipt = prepare_restart(data, output, args.every)
    if args.stop:
        stopper.main(['--model', data['model'], '--gpu', args.gpu, '--stop'])
    atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
