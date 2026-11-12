#!/usr/bin/env python3
"""Matched local-scale cap/target ablations from the fourth run's saved plan."""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_local_scale,run_soft_surface_long as periodic
from utils.checkpoint_snapshots import digest


def read_rows(path):
    result=[]
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return result


def inspect(model):
    model=Path(model).expanduser().resolve()
    audits=read_rows(model/'fixed_lidar_training.jsonl')
    updates=read_rows(model/'scale_regularization_training.jsonl')[-100:]
    result=dict(model=str(model),latest=audit_subset(audits[-1]) if audits else None)
    for key in ('active_at_cap_fraction','active_growth_at_cap_fraction','active_net_growth_at_cap_fraction',
                'active_above_target_fraction','regularizer_to_rgb_scale_grad_ratio','active_regularizer_to_rgb_scale_grad_ratio'):
        values=[row[key] for row in updates if row.get(key) is not None]
        if values:
            result[key+'_recent_median']=statistics.median(values)
    result['interpretation']='Growth at cap measures supervised points whose RGB gradient requests larger scales. Global at-cap fraction alone also includes untouched initialization points.'
    print(json.dumps(result,indent=2),flush=True)
    return result


def audit_subset(row):
    return {key:row[key] for key in ('iteration','gaussians','at_cap_fraction','above_target_fraction','largest_sigma',
        'ceiling_multiplier','target_multiplier','effective_max_scale_world') if key in row}


def prepare(argv):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-model',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--gpu',default='0')
    p.add_argument('--ceiling-multiplier',type=float,default=2.)
    p.add_argument('--target-multiplier',type=float,default=1.)
    p.add_argument('--iterations',type=int,default=30000)
    p.add_argument('--dry-run',action='store_true')
    request=p.parse_args(argv)
    reference=request.reference_model.expanduser().resolve()
    old=periodic.load_plan(reference)
    if (old.get('experiment') != 'local_scale' or old['plan'].get('ceiling_multiplier',1.) != 1. or
            old['plan'].get('target_multiplier',1.) != 1.):
        raise ValueError('Use the original 1x fourth-run local-scale experiment as the reference')
    args=run_local_scale.parser().parse_args(['--source',old['source'],'--model',str(request.model)])
    for key,value in old['training_args'].items():
        if hasattr(args,key):
            setattr(args,key,value)
    args.model=request.model
    args.gpu=request.gpu
    args.iterations=request.iterations
    args.dry_run=request.dry_run
    args.ceiling_multiplier=request.ceiling_multiplier
    args.target_multiplier=request.target_multiplier
    args.reuse_cache=Path(old['plan']['local_scale_cache'])
    args.local_scale_cache=request.model/'local_scale_cache'
    for name in ('source','model','baseline','keyframes','point_cloud','reuse_cache','local_scale_cache'):
        if getattr(args,name,None) is not None:
            setattr(args,name,Path(getattr(args,name)))
    for path,expected in old['input_fingerprints'].items():
        if digest(path) != expected:
            raise ValueError(f'Reference experiment input changed: {path}')
    inputs=[args.keyframes,args.point_cloud,reference/'periodic_evaluation/plan.json']
    plan=run_local_scale.build_plan(args)
    folder=periodic.directory(args.model)
    if not args.dry_run and (folder/'plan.json').is_file():
        with periodic.process_lock(folder/'trainer.lock'),periodic.process_lock(folder/'evaluator.lock'):
            existing=periodic.load_plan(args.model)
            serialized={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
            if existing['plan'] != plan or existing['training_args'] != serialized:
                raise ValueError('Prepared tuning run differs; use a fresh model directory')
            print(f'Reusing identical prepared tuning run: {folder/"plan.json"}',flush=True)
            return 0
    return periodic.prepare_experiment(args,plan,'local_scale',inputs)


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0]=='prepare':
        return prepare(argv[1:])
    p=argparse.ArgumentParser(description='Inspect scale constraints on the latest logged checkpoint/views')
    p.add_argument('--model',type=Path,required=True)
    args=p.parse_args(argv)
    inspect(args.model)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
