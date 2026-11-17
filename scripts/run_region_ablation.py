#!/usr/bin/env python3
"""Prepare fixed-position regional reconstruction or its full-cloud control."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_native_lidar as native
from scripts import run_soft_surface_long as periodic
from scripts.run_fixed_lidar import mask_inputs
from utils.region_loss import mask_path


def prepare_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--region',type=Path,required=True)
    p.add_argument('--context',choices=('crop','full'),default='crop')
    p.add_argument('--gpu',default='0')
    p.add_argument('--model',type=Path)
    p.add_argument('--iterations',type=int,default=30000)
    p.add_argument('--reference-steps',type=int,default=100000,
                   help='Full-scene step count for the matched expected updates/image checkpoint')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args(argv)
    region=args.region.expanduser().resolve()
    meta=json.loads((region/'selection.json').read_text())
    if meta.get('schema') != 1 or Path(meta['region']) != region:
        raise ValueError('Selection export was moved or has an incompatible schema; re-export with select_region.py')
    train=[r['name'] for r in meta['selection'] if r['split']=='train']
    if len(train)<2 or args.iterations<1 or args.reference_steps<1:
        raise ValueError('Need two training images and positive iteration counts')
    if set(train) != set(json.loads((region/'keyframes.json').read_text())['selected_images']):
        raise ValueError('Regional keyframe manifest changed after export')
    for name, expected in meta['mask_sha256'].items():
        if hashlib.sha256(mask_path(region/'masks',name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Supervision mask changed after export: {name}')
    matched=max(1,round(args.reference_steps*len(train)/meta['original_train_count']))
    # Always include the requested exposure-matched checkpoint, even for a large box.
    iterations=args.iterations if matched<=args.iterations else 10000*math.ceil(matched/10000)
    model=(args.model or region/'runs'/f'{args.context}_v1').expanduser().resolve()
    point_cloud=region/'input.ply' if args.context=='crop' else Path(meta['original_point_cloud'])
    command=['--source',str(region),'--model',str(model),'--keyframes',str(region/'keyframes.json'),
             '--point-cloud',str(point_cloud),'--gpu',args.gpu,'--iterations',str(iterations),
             '--checkpoint-every','10000','--extra-checkpoints',str(matched),
             '--expected-frames',str(len(meta['selection'])),'--resolution','1',
             '--loss-mask-dir',str(region/'masks'),'--freeze-lidar-positions','--scaling-lr','.001',
             '--densify-from-iter','0','--densify-until-iter','1','--densification-interval','250']
    if args.dry_run: command.append('--dry-run')
    training=native.parser().parse_args(command)
    protocol=dict(region=str(region),context=args.context,model=str(model),
                  selected_images=len(train),heldout_images=len(meta['selection'])-len(train),
                  full_scene_train_images=meta['original_train_count'],reference_full_scene_steps=args.reference_steps,
                  matched_expected_updates_step=matched,iterations=iterations,
                  expected_updates_per_image=dict(full=args.reference_steps/meta['original_train_count'],regional=matched/len(train)),
                  loss='ROI-normalized L1 + SSIM; outside-ROI pixels unsupervised',
                  visibility='native 3DGS + AA, no external culling',
                  geometry='fixed XYZ; learned color/opacity/rotation/scale; no growth, pruning, scale limits or opacity resets',
                  evaluation='ROI PSNR on every exported frame; original training membership; full-frame images dimmed outside ROI',
                  caveats=['Matched expected image draws, not equal per-surface gradient exposure.',
                           'SH degree follows the native global-step schedule; very early matched checkpoints may have lower SH degree.',
                           'Crop and full controls recompute KNN initialization; the halo reduces boundary differences.',
                           'A temporal-window selection tests view consistency but also changes coverage.'])
    return training, protocol


def main(argv=None):
    args, protocol=prepare_args(argv)
    plan=native.build_plan(args)
    print(json.dumps(protocol,indent=2),flush=True)
    result=periodic.prepare_experiment(args,plan,'native_lidar',
                [args.keyframes,args.point_cloud,Path(protocol['region'])/'selection.json',*mask_inputs(args)])
    if not args.dry_run:
        (args.model/'region_ablation.json').write_text(json.dumps(protocol,indent=2)+'\n')
    return result


if __name__=='__main__':
    raise SystemExit(main())
