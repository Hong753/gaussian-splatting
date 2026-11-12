#!/usr/bin/env python3
"""Repeat an existing fixed-LiDAR clamp experiment with distance-weighted Adam."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_fixed_lidar, run_soft_surface_long as periodic
from utils.checkpoint_snapshots import digest


def build_plan(args):
    previous = Path(args.previous_model).expanduser().resolve()
    old = periodic.load_plan(previous)
    if (old.get('experiment') != 'fixed_lidar' or old['plan'].get('scale_control','clamp') != 'clamp' or
            old['plan'].get('distance_updates') or old['gpu'] != args.gpu):
        raise ValueError('Previous experiment must be the unweighted fixed-LiDAR clamp run on the requested GPU')
    plan = run_fixed_lidar.build_plan(args)
    # Verify all original settings, including a user-supplied cap/resolution,
    # survived inheritance. Only model destinations and distance policy change.
    old_command = old['plan']['stages'][0]['argv']
    relocated = list(old_command)
    relocated[relocated.index('-m')+1] = plan['model']
    relocated[relocated.index('--keyframe_manifest')+1] = str(Path(plan['model'])/'keyframes.json')
    new_command = plan['stages'][0]['argv']
    distance_at = new_command.index('--distance_weighting')
    if new_command[:distance_at] != relocated:
        raise ValueError('Inherited training command changed beyond distance weighting; inspect previous plan')
    plan.update(previous_model=str(previous), previous_plan_sha256=digest(previous/'periodic_evaluation/plan.json'))
    return plan


def prepare(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous-model',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--gpu',default='0')
    p.add_argument('--reference',type=float,default=2.)
    p.add_argument('--power',type=float,default=2.)
    p.add_argument('--min-weight',type=float,default=.05)
    p.add_argument('--chunk-size',type=int,default=131072)
    p.add_argument('--dry-run',action='store_true')
    requested = p.parse_args(argv)
    old = periodic.load_plan(requested.previous_model)
    args = run_fixed_lidar.parser().parse_args(['--source',old['source'],'--model',str(requested.model)])
    for key,value in old['training_args'].items():
        if hasattr(args,key):
            setattr(args,key,value)
    args.previous_model = requested.previous_model
    args.model, args.gpu, args.dry_run = requested.model, requested.gpu, requested.dry_run
    args.distance_weighting = True
    args.distance_reference, args.distance_power = requested.reference, requested.power
    args.distance_min_weight, args.distance_chunk_size = requested.min_weight, requested.chunk_size
    for key in ('source','model','baseline','keyframes','point_cloud','local_scale_cache'):
        if getattr(args,key,None) is not None:
            setattr(args,key,Path(getattr(args,key)))
    inputs = [args.keyframes,args.point_cloud,requested.previous_model/'periodic_evaluation/plan.json']
    # Reuse exactly the same immutable source inputs as the old experiment.
    for path,expected in old['input_fingerprints'].items():
        if digest(path) != expected:
            raise ValueError(f'Original experiment input changed: {path}')
    plan = build_plan(args)
    folder = periodic.directory(args.model)
    if not args.dry_run and (folder/'plan.json').is_file():
        # A failed preflight leaves a prepared directory, not a trained model.
        # Permit an exact retry while refusing concurrent or changed experiments.
        with periodic.process_lock(folder/'trainer.lock'),periodic.process_lock(folder/'evaluator.lock'):
            existing = periodic.load_plan(args.model)
            serialized = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
            prior=dict(existing['training_args'])
            prior.setdefault('distance_metric','range')
            prior.setdefault('distance_decay','inverse_power')
            prior.setdefault('distance_half_life',2.)
            prior.setdefault('distance_normalization','absolute')
            prior.setdefault('distance_near_quantile',.05)
            prior.setdefault('distance_far_quantile',.95)
            prior.setdefault('distance_rank_samples',8192)
            if existing['plan'] != plan or prior != serialized:
                raise ValueError('Prepared replacement has different settings; use a fresh output directory')
            print(f'Reusing identical prepared replacement: {folder/"plan.json"}',flush=True)
            return 0
    return periodic.prepare_experiment(args,plan,'distance_weighted',inputs)


def execute_plan(args,plan):
    return run_fixed_lidar.execute_plan(args,plan)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'prepare':
        return prepare(argv[1:])
    return periodic.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
