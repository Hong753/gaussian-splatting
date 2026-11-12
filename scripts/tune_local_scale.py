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
from utils.local_scale_schedule import GROUP_LRS, validate as validate_schedule


def schedule_arguments(parser):
    parser.add_argument('--schedule-mode',choices=('off','cap','lr','joint'),default='off')
    parser.add_argument('--scale-start-multiplier',type=float,default=1.)
    parser.add_argument('--scale-ramp-start',type=int,default=7000)
    parser.add_argument('--scale-ramp-end',type=int,default=20000)
    parser.add_argument('--lr-decay-start',type=int,default=7000)
    parser.add_argument('--lr-decay-end',type=int,help='Defaults to the final training step')
    for name, value in GROUP_LRS.items():
        parser.add_argument('--lr-'+name.replace('_','-'),type=float,default=value)
        parser.add_argument('--lr-'+name.replace('_','-')+'-final',type=float,default=value*.1)
    parser.add_argument('--pose-refinement',action='store_true')
    parser.add_argument('--pose-start',type=int,default=7000)
    parser.add_argument('--pose-end',type=int,default=20000)
    parser.add_argument('--pose-every',type=int,default=1)
    parser.add_argument('--pose-max-translation',type=float,default=.03)
    parser.add_argument('--pose-max-rotation-degrees',type=float,default=.5)
    parser.add_argument('--pose-translation-lr',type=float,default=.001)
    parser.add_argument('--pose-rotation-lr',type=float,default=.0002)
    parser.add_argument('--pose-prior-weight',type=float,default=.001)
    parser.add_argument('--pose-max-width',type=int,default=960)


def requested_schedule(args):
    if args.schedule_mode == 'off':
        return None
    cap = (dict(initial_multiplier=args.scale_start_multiplier,final_multiplier=args.ceiling_multiplier,
                start=args.scale_ramp_start,end=args.scale_ramp_end) if args.schedule_mode in ('cap','joint') else None)
    rates = (dict(start=args.lr_decay_start,end=args.lr_decay_end if args.lr_decay_end is not None else args.iterations,
                  groups={name:[getattr(args,'lr_'+name),getattr(args,'lr_'+name+'_final')] for name in GROUP_LRS})
             if args.schedule_mode in ('lr','joint') else None)
    return validate_schedule(dict(schema=1,cap=cap,learning_rates=rates),args.ceiling_multiplier,args.iterations)


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
    if updates:
        result['latest_schedule'] = {key:value for key,value in updates[-1].items()
            if key.startswith('lr_') or key in ('schedule_iteration','current_ceiling_multiplier')}
    result['interpretation']='Growth at cap measures supervised points whose RGB gradient requests larger scales. Global at-cap fraction alone also includes untouched initialization points.'
    print(json.dumps(result,indent=2),flush=True)
    return result


def audit_subset(row):
    return {key:row[key] for key in ('iteration','gaussians','at_cap_fraction','above_target_fraction','largest_sigma',
        'ceiling_multiplier','current_ceiling_multiplier','target_multiplier','effective_max_scale_world') if key in row}


def prepare(argv):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-model',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--gpu',default='0')
    p.add_argument('--ceiling-multiplier',type=float,default=2.)
    p.add_argument('--target-multiplier',type=float,default=1.)
    p.add_argument('--iterations',type=int,default=30000)
    p.add_argument('--dry-run',action='store_true')
    schedule_arguments(p)
    request=p.parse_args(argv)
    schedule=requested_schedule(request)
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
    args.training_schedule=schedule
    args.pose_policy=None
    if request.pose_refinement:
        from utils.pose_refinement import DEFAULTS, validate as validate_pose
        args.pose_policy=validate_pose({**DEFAULTS,**{key:getattr(request,'pose_'+key) for key in
            ('start','end','every','max_translation','max_rotation_degrees','translation_lr','rotation_lr','prior_weight','max_width')}},request.iterations)
    args.reuse_cache=Path(old['plan']['local_scale_cache'])
    args.local_scale_cache=request.model/'local_scale_cache'
    for name in ('source','model','baseline','keyframes','point_cloud','reuse_cache','local_scale_cache'):
        if getattr(args,name,None) is not None:
            setattr(args,name,Path(getattr(args,name)))
    for path,expected in old['input_fingerprints'].items():
        if digest(path) != expected:
            raise ValueError(f'Reference experiment input changed: {path}')
    inputs=[args.keyframes,args.point_cloud,reference/'periodic_evaluation/plan.json']
    if args.pose_policy is not None:
        from utils.pose_refinement import source_fingerprints
        inputs.extend(Path(path) for path in source_fingerprints(args.source))
    plan=run_local_scale.build_plan(args)
    folder=periodic.directory(args.model)
    if not args.dry_run and (folder/'plan.json').is_file():
        with periodic.process_lock(folder/'trainer.lock'),periodic.process_lock(folder/'evaluator.lock'):
            existing=periodic.load_plan(args.model)
            serialized={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
            prior_args=dict(existing['training_args'])
            prior_args.setdefault('training_schedule',None)
            prior_args.setdefault('pose_policy',None)
            if existing['plan'] != plan or prior_args != serialized:
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
