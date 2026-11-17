#!/usr/bin/env python3
"""Compare shared regional images/masks; explicitly opt in to different poses."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.view_comparisons import model_info, read_colmap_records, write_gallery
from scripts.eval_all_frames import choose_iteration
from scripts.occlusion_options import apply_overrides


def prepare_specs(region, models, iterations, different_poses=False):
    import numpy as np
    region=Path(region).resolve()
    selection=json.loads((region/'selection.json').read_text())
    records=read_colmap_records(region)
    specs=[]
    for model, step in zip(models,iterations):
        spec=model_info(model)
        if spec['type']!='3dgs':
            raise ValueError('Regional diagnostic currently supports native 3DGS checkpoints')
        for name, reference in records.items():
            if name not in spec['records']:
                raise ValueError(f'{model}: missing regional camera {name}')
            saved=spec['records'][name]
            keys=('width','height','fx','fy','cx','cy') if different_poses else ('R','T','width','height','fx','fy','cx','cy')
            for key in keys:
                if not np.allclose(saved[key],reference[key],rtol=0,atol=1e-8):
                    raise ValueError(f'{model}: camera {name} has different {key}; this is not the same-pose ablation')
            if Path(saved['image_path']).resolve()!=Path(reference['image_path']).resolve():
                raise ValueError(f'{model}: image source differs for {name}')
        selected_step,_=choose_iteration(Path(model),step,'3dgs')
        spec=apply_overrides(spec,SimpleNamespace(occlusion='off'))
        own_records={name:spec['records'][name] for name in records} if different_poses else records
        spec.update(records=own_records,camera_poses='original',comparison_iteration=selected_step)
        spec['cfg']={**spec['cfg'],'loss_mask_dir':str(region/'masks'),'resolution':1}
        specs.append(spec)
    return specs,selection


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--region',type=Path,required=True)
    p.add_argument('--models',type=Path,nargs='+',required=True)
    p.add_argument('--iterations',type=int,nargs='+',default=[30000])
    p.add_argument('--different-steps',action='store_true',help='Explicitly compare different iterations, e.g. expected image exposure matching')
    p.add_argument('--different-poses',action='store_true',help='Render each model using its own dataset poses; still require identical RGB/calibration/masks')
    p.add_argument('--gpu',default='0')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    steps=args.iterations*len(args.models) if len(args.iterations)==1 else args.iterations
    if len(steps)!=len(args.models): p.error('Pass one shared iteration or one iteration per model')
    specs,selection=prepare_specs(args.region,args.models,steps,args.different_poses)
    if len({s['comparison_iteration'] for s in specs})>1 and not args.different_steps:
        p.error('Different checkpoints require --different-steps; otherwise choose a shared saved iteration')
    if args.output.exists(): raise FileExistsError(f'Choose a fresh comparison output: {args.output}')
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    from scripts.render_views_worker import render_views
    import torch
    names=sorted(specs[0]['records'])
    rendered=[]
    for i,spec in enumerate(specs):
        folder=args.output/f'model_{i}'
        render_views(spec,names,spec['comparison_iteration'],folder,True)
        rendered.append(json.loads((folder/'frames.json').read_text()))
        torch.cuda.empty_cache()
    rows=[]
    for name in names:
        if len({r[name]['target_sha256'] for r in rendered})!=1:
            raise ValueError(f'Ground truth differs across runs: {name}')
        rows.append(dict(name=name,gt=f'model_0/{rendered[0][name]["gt"]}',renders=[
            {**{k:f'model_{i}/{r[name][k]}' for k in ('image','error','comparison')},'psnr_db':r[name]['psnr_db']}
            for i,r in enumerate(rendered)]))
    training={r['name'] for r in selection['selection'] if r['split']=='train'}
    def average(r, subset):
        if not subset: return None
        value=sum(float(r[n]['psnr_db']) for n in subset)/len(subset)
        return value if math.isfinite(value) else 'inf'
    summaries=[dict(label=f'{s["path"].name} · {s["comparison_iteration"]:,} · ROI',path=str(s['path']),
        iteration=s['comparison_iteration'],mean_psnr_db=average(r,names),
        region_train_psnr=average(r,training),region_heldout_psnr=average(r,set(names)-training))
        for s,r in zip(specs,rendered)]
    write_gallery(args.output,'Regional GT / rendering comparison — ROI PSNR',rows,summaries,
        dict(metric='ROI-only mean per-channel PSNR, then mean over views; outside ROI dimmed in images',
             camera_poses=('Per-model dataset poses (may differ across runs)' if args.different_poses else 'Identical original COLMAP poses')+'; full resolution; AA; no culling; no exposure',
             region=str(args.region.resolve()),regional_training_names=sorted(training),
             different_steps=args.different_steps,different_poses=args.different_poses,
             counts=dict(all=len(names),train=len(training),heldout=len(names)-len(training))))
    print(json.dumps(summaries,indent=2))
    print(f'Open {args.output / "index.html"}')


if __name__=='__main__':
    raise SystemExit(main())
