#!/usr/bin/env python3
"""Fresh, matched Orange4F scale-control runs, independent of earlier checkpoints."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

SOURCE = Path('/data/colmap_scenes/orange4f')
OUTPUT_ROOT = Path('/data/3dgs_outputs/orange4f/scale_control_v1')
ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('baseline', 'split3cm_noreset', 'size_driven_split', 'size_driven_split_regularized')
STEPS = [5000,6000,10000,15000,20000,30000,60000,100000,120000]


def command_options(variant,extent):
    opt = dict(iterations=120000, position_warmup_steps=6000, position_ramp_steps=3000,
               position_lr_init=.000016, position_lr_final=.00000016, position_lr_max_steps=114000,
               densification_phase_offset=6000, densify_from_iter=0, densify_until_iter=94000,
               densification_interval=250, densify_grad_threshold=.0002,
               scaling_lr=.001, opacity_reset_interval=9000, percent_dense=.01,
               # All four runs use the same opacity-only deletion policy.
               # Size-driven subdivision is not mixed with destructive size pruning.
               prune_screen_size=0., prune_world_scale_ratio=0.)
    if variant != 'baseline':
        opt.update(percent_dense=.03/extent,disable_opacity_resets=True)
    if variant in ('size_driven_split','size_driven_split_regularized'):
        opt.update(native_forced_split_sigma=.1,native_forced_split_fraction=.01,
                   native_forced_split_max_parents=50000,native_forced_split_population=12000000,
                   native_forced_split_min_observations=1,native_forced_split_min_opacity=.05,
                   native_longest_axis_split=True,native_split_offset_fraction=.8)
    if variant == 'size_driven_split_regularized':
        opt.update(native_scale_soft_sigma=.05,native_scale_loss_weight=.01)
    return opt


def flags(values):
    result = []
    for key,value in values.items():
        if isinstance(value,bool):
            if value: result.append('--'+key)
        else:
            result.extend(('--'+key,str(value)))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--gpu',type=int,required=True,help='Physical GPU index')
    p.add_argument('--initial-point-cloud',type=Path,help='Optional shared PLY already aligned with the source')
    p.add_argument('--render-max-views',type=int,default=0,help='0: all registered training views')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args(argv)
    if args.gpu<0 or args.render_max_views<0:
        p.error('GPU and render view count must be nonnegative')
    out=OUTPUT_ROOT/args.variant
    if out.exists():
        p.error(f'Refusing to overwrite existing run: {out}')
    from scripts.run_orange4f_blur_ablation import alternate_source,convert_initial_cloud
    from utils.native_scale_control import self_test
    self_test()
    cfg,extent,cloud,conversion,count=alternate_source({},SOURCE,args.initial_point_cloud,out)
    opt=command_options(args.variant,extent)
    command=[sys.executable,'-u','train.py','-s',cfg['source_path'],'-m',str(out),
             '-r','1600','--data_device','cpu','--colmap_principal_point',
             '--initial_point_cloud',str(cloud),'--training_scene_extent',str(extent),
             *flags(opt),
             '--test_iterations',*map(str,STEPS),'--save_iterations',*map(str,STEPS),
             '--checkpoint_iterations',*map(str,STEPS),
             '--debug_preview_every','500','--debug_preview_train','2','--debug_preview_test','2',
             '--debug_preview_seed','0','--disable_viewer']
    render=[sys.executable,'-u','scripts/render.py','--model',str(out),'--split','both',
            '--iteration','120000','--gpu',str(args.gpu),'--max-views',str(args.render_max_views)]
    audit=[sys.executable,'-u','-m','scripts.audit_splat_blur','--model',str(out),
           '--iteration','120000','--views','12','--history']
    plan=dict(source=cfg['source_path'],training_views=count,scene_extent=extent,
              initial_point_cloud=str(cloud),converted_from=str(conversion) if conversion else None,
              variant=args.variant,gpu=args.gpu,split_sigma=opt['percent_dense']*extent,
              options=opt,commands=[command,render,audit],
              note='All registered views, fixed source calibration, no pose refinement; scene units assumed metric. '
                   'Baseline uses native gradient densification and opacity pruning with size deletion disabled. '
                   'Forced split cap bounds extra growth only; ordinary densification can exceed it.')
    print(json.dumps(plan,indent=2),flush=True)
    if args.dry_run:
        return 0
    out.mkdir(parents=True,exist_ok=False)
    if conversion is not None:
        convert_initial_cloud(conversion,cloud)
    (out/'scale_control_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu))
    for cmd in (command,render,audit):
        print('CUDA_VISIBLE_DEVICES='+str(args.gpu)+' '+shlex.join(cmd),flush=True)
        subprocess.run(cmd,cwd=ROOT,env=env,check=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
