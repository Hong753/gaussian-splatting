#!/usr/bin/env python3
"""Compare full and indexed-only renders of the same trained Scaffold on the same cameras."""
import argparse
import html
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.checkpoint_snapshots import atomic_json, atomic_write, read_snapshot


def checkpoint(model, iteration, wait):
    while True:
        steps = []
        for marker in (model/'point_cloud').glob('iteration_*/ready.json'):
            match = re.fullmatch(r'iteration_(\d+)', marker.parent.name)
            if match:
                steps.append(int(match[1]))
        step = iteration if iteration is not None else max(steps, default=None)
        if step in steps:
            snapshot = read_snapshot(model, step)
            if snapshot.get('model_type') != 'scaffold':
                raise ValueError('This comparison requires a published Scaffold checkpoint')
            return step
        state_path = model/'periodic_evaluation/training.json'
        state = json.loads(state_path.read_text()).get('state') if state_path.is_file() else None
        history = model/'debug_previews/history.json'
        previews = json.loads(history.read_text())['snapshots'] if history.is_file() else []
        last = max((p['iteration'] for p in previews), default=0)
        msg = f'Waiting for a published checkpoint. Latest RGB preview step {last:,}.'
        if not wait or state in ('failed', 'trained'):
            raise SystemExit(f'No matching saved checkpoint in {model}. Training state {state}. Latest preview step {last:,}.')
        print(msg, flush=True)
        time.sleep(30)


def frame_names(model, training, requested, count):
    if requested:
        missing = set(requested) - set(training)
        if missing:
            raise ValueError(f'These frames have no training membership: {sorted(missing)}')
        return list(dict.fromkeys(requested))
    selection = model/'debug_previews/selection.json'
    preferred = []
    if selection.is_file():
        preferred = [f['name'] for f in json.loads(selection.read_text())['frames']
                     if f['split'] == 'train' and f['name'] in training]
    # Include the train previews the user saw, then sample across the trajectory.
    import numpy as np
    remaining = sorted(set(training)-set(preferred))
    extra = min(max(0,count-len(preferred)),len(remaining))
    return (preferred + [remaining[i] for i in np.linspace(0,len(remaining)-1,extra,dtype=int)])[:count]


def compare(camera, gaussians, pipe, bg, indexed):
    import torch
    from models.scaffold_renderer import prefilter_voxel, render
    from utils.region_loss import score
    visible = prefilter_voxel(camera, gaussians, pipe, bg)
    selected = visible & indexed
    target = camera.original_image[:3].to(bg.device)
    def draw(mask, weights=False):
        result = render(camera, gaussians, pipe, bg, visible_mask=mask,
                        contribution_anchors=indexed if weights else None)
        return result['render'].clamp(0,1), int((result['radii'] > 0).sum())
    full, n_full = draw(visible)
    only, n_only = draw(selected)
    weights, _ = draw(visible, True)
    selected_weights, _ = draw(selected, True)
    if not all(torch.isfinite(x).all() for x in (target,full,only,weights,selected_weights)):
        raise ValueError(f'Nonfinite render for {camera.image_name}')
    full_indexed = weights[1].mean().item()
    alone_indexed = selected_weights[1].mean().item()
    blocked = max(0., min(1., 1. - full_indexed / alone_indexed)) if alone_indexed > 0 else None
    stats = dict(full_psnr_db=score(full,target).item(), indexed_psnr_db=score(only,target).item(),
                 full_children=n_full, indexed_children=n_only,
                 mean_unindexed_weight=weights[0].mean().item(),
                 mean_indexed_weight=full_indexed, mean_background_weight=weights[2].mean().item(),
                 indexed_weight_alone=alone_indexed, fraction_indexed_weight_blocked=blocked,
                 indexed_coverage=(selected_weights[1]>.5).float().mean().item())
    return (target, full, only, weights), stats


def blocked_label(row):
    value = row['fraction_indexed_weight_blocked']
    return 'n/a (zero indexed weight)' if value is None else f'{value:.1%}'


def write_report(directory, step, rows):
    cards = []
    for name, row in rows.items():
        labels = ['Camera image',f'Full · {row["full_psnr_db"]:.2f} dB',
                  f'Indexed only · {row["indexed_psnr_db"]:.2f} dB','Composited weight']
        figures = ''.join(f'<figure><figcaption>{label}</figcaption><a href="{url}"><img src="{url}"></a></figure>'
                          for label,url in zip(labels,row['images']))
        cards.append(f'<section><h2>{html.escape(name)}</h2><p>Mean unindexed weight {row["mean_unindexed_weight"]:.1%}'
                     f' · Indexed weight blocked {blocked_label(row)} · Indexed-only coverage {row["indexed_coverage"]:.1%}</p><div>{figures}</div></section>')
    page = f'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scaffold parent comparison</title><style>
body{{font:16px/1.5 system-ui;background:#15191e;color:#f3f4f6;margin:2vw}}h1,h2{{margin-bottom:0}}
section{{margin:1.5em 0}}div{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:4px}}
figure{{margin:0}}img{{display:block;width:100%}}p,figcaption{{color:#c4cad3}}
@media(max-width:800px){{div{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}</style>
<h1>Scaffold parent comparison · step {step:,}</h1>
<p>The weight image uses red for unindexed parents, green for indexed parents and blue for background.
Blocked weight measures the fraction of indexed contribution lost when other parents are included.
Indexed-only coverage is the fraction of image pixels where indexed children alone reach opacity above 0.5.</p>
{''.join(cards)}'''
    atomic_write(directory/'index.html',lambda path:path.write_text(page))


def run(args):
    import torch
    from PIL import Image
    from scripts.view_comparisons import model_info, image_id
    from scripts.render_views_worker import load_record_camera
    from models.scaffold_gaussian import ScaffoldGaussianModel
    from utils.scaffold_observations import Membership

    model = args.model.expanduser().resolve()
    if not (model/'observations/metadata.json').is_file():
        raise ValueError(f'No prepared per-camera membership in {model}')
    meta = json.loads((model/'observations/metadata.json').read_text())
    print(f'{len(meta["names"]):,} training images · {meta["empty_views"]:,} empty selections · '
          f'median {meta["median_sensed_parents"]:,.0f} eligible parents out of {meta["anchor_count"]:,}', flush=True)
    log = model/'scaffold_training.jsonl'
    if log.is_file():
        for line in reversed(log.read_text().splitlines()):
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue  # The active trainer may be writing its last line.
            if 'eligible_rendered_children' in last:
                print(f'Logged step {last["iteration"]:,} · {last["image"]} · '
                      f'{last["eligible_rendered_children"]:,} eligible children / {last["active_gaussians"]:,} '
                      'with positive screen radius', flush=True)
            break
    step = checkpoint(model,args.iteration,args.wait)
    if not torch.cuda.is_available():
        raise RuntimeError('The saved-model comparison needs the CUDA-enabled 3dgs environment')
    spec = model_info(model)
    cfg = spec['cfg']
    names = frame_names(model, spec['splits']['train'],args.frame,args.count)
    gs = ScaffoldGaussianModel(feat_dim=cfg['scaffold_feat_dim'], n_offsets=cfg['scaffold_n_offsets'],
                               voxel_size=cfg['scaffold_voxel_size'],update_init_factor=16,appearance_dim=0)
    gs.load_ply(str(model/f'point_cloud/iteration_{step}/point_cloud.ply'))
    gs.eval()
    membership = Membership(model/'observations',gs.get_anchor,spec['splits']['train'])
    pipe = SimpleNamespace(debug=False,antialiasing=cfg.get('antialiasing',False))
    bg = torch.full((3,),float(cfg.get('white_background',False)),device='cuda')
    cam_args = SimpleNamespace(resolution=cfg['resolution'],data_device='cpu',train_test_exp=False)
    folder = model/'debug_previews/parent_audit'
    folder.mkdir(parents=True,exist_ok=True)
    rows = {}
    with torch.inference_mode():
        for i,name in enumerate(names):
            cam = load_record_camera(cam_args,spec['records'][name],i)
            images, stats = compare(cam,gs,pipe,bg,membership.mask(name))
            links = []
            for tensor, label in zip(images,('gt','full','indexed','weights')):
                file = f'{step}_{image_id(name)}_{label}.png'
                image = Image.fromarray(tensor.mul(255).round().clamp(0,255).byte().permute(1,2,0).cpu().numpy())
                atomic_write(folder/file,lambda path:image.save(path,format='PNG'))
                links.append(file)
            rows[name] = dict(**stats,images=links)
            print(f'{name} · full {stats["full_psnr_db"]:.2f} dB · indexed only {stats["indexed_psnr_db"]:.2f} dB'
                  f' · unindexed weight {stats["mean_unindexed_weight"]:.1%} · indexed weight blocked {blocked_label(stats)}'
                  f' · indexed coverage {stats["indexed_coverage"]:.1%}',flush=True)
    metrics = {name:{k:('inf' if isinstance(v,float) and math.isinf(v) else v) for k,v in row.items()}
               for name,row in rows.items()}
    atomic_json(folder/f'metrics_{step}.json',dict(iteration=step,frames=metrics))
    write_report(folder,step,rows)
    print(f'Comparison saved to {folder}/index.html',flush=True)


def main():
    base = Path(os.environ.get('ORANGE4F_OUTPUT_ROOT','/data/3dgs_outputs/orange4f/3dgs_finetuning'))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,default=base/'stride3_noaa_v1/scaffold_indexed_anchors_v1')
    p.add_argument('--iteration',type=int,help='Defaults to the latest published checkpoint')
    p.add_argument('--count',type=int,default=6)
    p.add_argument('--frame',action='append',help='Training image name. Repeat to select multiple frames.')
    p.add_argument('--wait',action='store_true',help='Wait for a checkpoint if the run has only RGB previews so far')
    args = p.parse_args()
    if args.count < 1 or (args.iteration is not None and args.iteration < 1):
        p.error('Count and iteration must be positive')
    run(args)


if __name__ == '__main__':
    main()
