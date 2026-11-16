"""Small, repeatable RGB diagnostics from the live model, without checkpoints.

Only selected held-out images are loaded. No model copy, optimizer step, depth
cache, or Gaussian population change is made by the observer.
"""
from copy import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
from types import SimpleNamespace

from utils.checkpoint_snapshots import atomic_json, atomic_write


DEFAULTS = dict(every=500, train=2, test=2, seed=0)


def add_arguments(parser):
    for name, default in DEFAULTS.items():
        parser.add_argument('--debug_preview_' + name, '--debug-preview-' + name,
                            type=int, default=default,
                            help={'every': 'Live GT/render/error interval (0 disables; also renders step 1 and final)',
                                  'train': 'Number of fixed random training preview frames',
                                  'test': 'Number of fixed random held-out preview frames',
                                  'seed': 'Private preview selection seed; does not change training RNG'}[name])


def options(args):
    result = {name: getattr(args, 'debug_preview_' + name, default) for name, default in DEFAULTS.items()}
    if any(not isinstance(value, int) or value < 0 for value in result.values()):
        raise ValueError('Debug preview interval, frame counts and seed must be nonnegative integers')
    if result['every'] and not (result['train'] or result['test']):
        raise ValueError('Enabled debug previews require at least one requested frame')
    return result


def command_arguments(args):
    # Older prepared plans have none of these fields. Keep their command stable;
    # train.py supplies its defaults when those plans are executed after upgrade.
    if not hasattr(args, 'debug_preview_every'):
        return []
    return [part for name, value in options(args).items()
            for part in ('--debug_preview_' + name, str(value))]


def select_names(train_names, test_names, config):
    train_names = sorted(set(train_names))
    test_names = sorted(set(test_names) - set(train_names))
    return {split: sorted(random.Random(config['seed'] + salt).sample(names, min(count, len(names))))
            for split, names, count, salt in (
                ('train', train_names, config['train'], 0),
                ('held-out', test_names, config['test'], 1))}


def load_heldout_camera(record, dataset):
    import numpy as np
    from utils.camera_utils import loadCam
    info = SimpleNamespace(uid=0, image_name=record['name'], image_path=record['image_path'],
        R=np.array(record['R']), T=np.array(record['T']), FovX=record['fov_x'], FovY=record['fov_y'],
        depth_path='', depth_params=None, is_test=True)
    # Preserve native image resolution, but keep the few extra RGB tensors on CPU.
    args = SimpleNamespace(resolution=dataset.resolution, data_device='cpu', train_test_exp=False)
    return loadCam(args, 0, info, 1., False, True)


def create_previewer(scene, dataset, config, *, start_iteration=0):
    if not config['every']:
        return None
    train = {c.image_name: c for c in scene.getTrainCameras()}
    test = {c.image_name: c for c in scene.getTestCameras() if c.image_name not in train}
    records = {}
    protocol = 'Dataset test cameras excluding all training image names'
    if config['test'] and getattr(dataset, 'keyframe_manifest', ''):
        from scripts.view_comparisons import read_colmap_records
        records = read_colmap_records(dataset.source_path, dataset.images or 'images')
        test_names = set(records) - set(train)
        protocol = 'All registered COLMAP frames not selected for training (keyframe complement)'
    else:
        test_names = set(test)
    selected = select_names(train, test_names, config)
    frames = []
    for split, names in selected.items():
        for name in names:
            camera = train[name] if split == 'train' else test.get(name)
            if camera is None:
                camera = load_heldout_camera(records[name], dataset)
            frames.append((split, camera))
    if config['test'] and not selected['held-out']:
        print('Debug previews: no disjoint held-out frames available; showing training frames only.', flush=True)
    if not frames:
        print('Debug previews: no requested frames available.', flush=True)
        return None
    root = Path(scene.model_path) / 'debug_previews'
    history = root / 'history.json'
    if history.exists() and any(s['iteration'] > start_iteration for s in json.loads(history.read_text())['snapshots']):
        # A normal checkpoint resume can rewind past previews saved after that
        # checkpoint. Preserve that old trajectory, rather than reusing its
        # images or crashing training at the first duplicated preview step.
        archive = root.with_name(f'debug_previews_before_restart_{time.time_ns()}')
        root.rename(archive)
        print(f'Debug previews: archived the later trajectory to {archive}', flush=True)
    previewer = TrainingPreviewer(root, frames, config,
        dict(source=str(Path(dataset.source_path).resolve()), heldout_protocol=protocol,
             eligible_train=len(train), eligible_heldout=len(test_names)))
    print(f'Debug previews: {len(selected["train"])} train + {len(selected["held-out"])} held-out, '
          f'every {config["every"]} steps; {previewer.root / "index.html"}', flush=True)
    return previewer


class TrainingPreviewer:
    def __init__(self, root, frames, config, metadata):
        self.root, self.frames, self.config = Path(root), frames, config
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = dict(schema=1, seed=config['seed'], frames=[dict(split=s, name=c.image_name)
                            for s, c in frames], **metadata)
        manifest_path = self.root / 'selection.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != self.manifest:
            raise ValueError('Debug preview selection changed; use a fresh model directory to avoid mixing views')
        atomic_json(manifest_path, self.manifest)
        if not (self.root / 'history.json').exists():
            atomic_json(self.root / 'history.json', dict(selection=self.manifest, snapshots=[]))
        atomic_write(self.root / 'index.html', lambda path: path.write_text(HTML))

    def due(self, iteration, final_iteration):
        return bool(self.config['every'] and
                    (iteration == 1 or iteration == final_iteration or iteration % self.config['every'] == 0))

    def capture(self, iteration, model, render_fn, pipeline, background, *, pose_refiner=None,
                separate_sh=False, tb_writer=None):
        import numpy as np
        import torch
        from PIL import Image

        started = time.monotonic()
        folder = self.root / f'iteration_{iteration:06d}'
        if folder.exists():
            raise FileExistsError(f'Debug preview already exists: {folder}; do not mix restarted trajectories')
        temporary = Path(tempfile.mkdtemp(prefix='.preview-', dir=self.root))
        rows = []
        # Use the inference pipeline supplied by training. Copy the settings so
        # the observer cannot modify flags used by the next training iteration.
        pipe = copy(pipeline)
        devices = [background.device.index] if background.is_cuda else []
        py_state, np_state = random.getstate(), np.random.get_state()

        def rgb(tensor):
            array = tensor.detach().cpu().permute(1, 2, 0).numpy()
            return Image.fromarray((array * 255).round().clip(0, 255).astype(np.uint8))

        try:
            with torch.no_grad(), torch.random.fork_rng(devices=devices):
                for split, original_camera in self.frames:
                    refined = (split == 'train' and pose_refiner is not None and
                               original_camera.image_name in pose_refiner.names)
                    camera = pose_refiner.camera(original_camera) if refined else original_camera
                    pred = render_fn(camera, model, pipe, background, separate_sh=separate_sh,
                                     use_trained_exp=False)['render']
                    target = original_camera.original_image[:3].to(pred.device)
                    if pred.shape != target.shape or not torch.isfinite(pred).all() or not torch.isfinite(target).all():
                        raise ValueError(f'Invalid debug preview image: {original_camera.image_name}')
                    pred, target = pred.clamp(0, 1), target.clamp(0, 1)
                    # Match scripts/render_views_worker.py / all-frame evaluation:
                    # PSNR per RGB channel, then the mean over channels.
                    mse = (pred - target).square().flatten(1).mean(1)
                    score = (-10 * torch.log10(mse)).mean().item()
                    error = (pred - target).abs().mul(4).clamp(0, 1)
                    key = hashlib.sha256((split + '/' + original_camera.image_name).encode()).hexdigest()[:24]
                    gt_path = self.root / 'gt' / f'{key}.png'
                    if not gt_path.exists():
                        atomic_write(gt_path, lambda path: rgb(target).save(path, format='PNG'))
                    rgb(pred).save(temporary / f'{key}.png')
                    rgb(error).save(temporary / f'{key}_error.jpg', quality=95, subsampling=0)
                    row = dict(name=original_camera.image_name, split=split,
                        pose='current refined training pose' if refined else 'original pose',
                        psnr_db=score if math.isfinite(score) else 'inf',
                        gt=f'gt/{key}.png', render=f'{folder.name}/{key}.png',
                        error=f'{folder.name}/{key}_error.jpg', width=pred.shape[-1], height=pred.shape[-2])
                    rows.append(row)
                    if tb_writer:
                        tag = f'debug_previews/{split}/{key}'
                        tb_writer.add_image(tag + '/gt_render_error_x4', torch.cat((target, pred, error), dim=2), iteration)
                        if math.isfinite(score):
                            tb_writer.add_scalar(tag + '/psnr', score, iteration)
                    del pred, target, error, mse, camera
            snapshot = dict(iteration=iteration, frames=rows, seconds=time.monotonic() - started,
                created_unix=time.time(), antialiasing=bool(getattr(pipe, 'antialiasing', False)),
                occlusion_mode=getattr(pipe, 'occlusion_mode', 'off'),
                exposure='base appearance; per-image exposure off',
                metric='mean per-channel PSNR, then mean over selected frames; not all-frame evaluation',
                state_timing='end of training iteration')
            atomic_json(temporary / 'frames.json', snapshot)
            os.replace(temporary, folder)
            history_path = self.root / 'history.json'
            history = json.loads(history_path.read_text())
            history['snapshots'].append(snapshot)
            atomic_json(history_path, history)  # Publish only after all images exist.
            atomic_json(self.root / 'latest.json', snapshot)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            shutil.rmtree(temporary, ignore_errors=True)
        print(f'[ITER {iteration}] Debug previews: {len(rows)} frames in {snapshot["seconds"]:.1f}s; '
              f'{self.root / "index.html"}', flush=True)
        return snapshot


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Training previews · GT / render</title><style>
*{box-sizing:border-box}body{margin:0;background:#101720;color:#e7eef7;font:15px system-ui,sans-serif}
header{padding:24px 30px;border-bottom:1px solid #344150;background:#141f2d}h1{font-size:24px;margin:0 0 8px}
p{margin:8px 0;color:#b9c8d9;line-height:1.5}nav{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin-top:18px}
select,button{background:#26374c;color:#f0f5fc;border:1px solid #4f637e;border-radius:5px;padding:8px}
main{padding:22px 30px;display:grid;gap:24px}.card{background:#192535;border:1px solid #344150;border-radius:8px;overflow:hidden}
.title{padding:14px 16px;display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap}.meta{color:#afc1d9}
.images{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:2px;background:#344150}
figure{margin:0;background:#121c29}figcaption{padding:9px 12px;color:#c8d6e8;font-size:13px}
img{display:block;width:100%;height:auto;cursor:zoom-in}a{color:#8abcfb}.badge{color:#78d5bb}#status{font-size:13px}
@media(max-width:700px){header,main{padding:16px}.images{grid-template-columns:1fr}}
</style></head><body><header><h1>Training previews</h1>
<p>Fixed random frames · GT / render / absolute error ×4. Click an image for full resolution.</p>
<p>These few-frame PSNR values are diagnostics; the separate all-frame evaluation remains unchanged.</p>
<nav><label>Step <select id="step"></select></label>
<label>Frames <select id="split"><option value="both">Train + held-out</option><option value="train">Train</option><option value="held-out">Held-out</option></select></label>
<label><input type="checkbox" id="follow" checked> Follow latest</label><button id="refresh">Refresh</button></nav>
<p id="protocol"></p><p id="status">Waiting for the first preview…</p></header><main id="cards"></main>
<script>
'use strict';
let snapshots=[],latestKey='';
const $=id=>document.getElementById(id);
function elem(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;}
function show(){
  const s=snapshots.find(x=>String(x.iteration)===$('step').value);if(!s)return;
  $('cards').replaceChildren();
  for(const frame of s.frames){
    if($('split').value!=='both'&&frame.split!==$('split').value)continue;
    const card=elem('section',undefined,'card'),head=elem('div',undefined,'title');
    head.append(elem('strong',frame.split+' · '+frame.name),elem('span',
      (typeof frame.psnr_db==='number'?frame.psnr_db.toFixed(3):frame.psnr_db)+' dB · '+frame.pose,'meta'));
    const images=elem('div',undefined,'images');
    for(const [key,label] of [['gt','Actual image'],['render','Rendered image'],['error','Absolute error ×4']]){
      const f=elem('figure'),link=elem('a'),im=elem('img');link.href=frame[key];link.target='_blank';link.rel='noopener';
      im.src=frame[key];im.alt=label+' — '+frame.name;link.append(im);f.append(elem('figcaption',label),link);images.append(f);
    }card.append(head,images);$('cards').append(card);
  }
  $('protocol').textContent='Step '+s.iteration.toLocaleString()+' · AA '+(s.antialiasing?'on':'off')+
    ' · Culling '+s.occlusion_mode+' · '+s.exposure+' · Preview render/write '+s.seconds.toFixed(1)+' s';
}
async function refresh(){
  try{
    const response=await fetch('history.json',{cache:'no-store'});if(!response.ok)throw new Error('HTTP '+response.status);
    const data=await response.json();snapshots=data.snapshots;
    if(!snapshots.length){$('status').textContent='Waiting for step 1…';return;}
    const last=snapshots[snapshots.length-1],key=last.iteration+':'+last.created_unix;
    if(key!==latestKey){
      const selected=$('step').value;$('step').replaceChildren();
      for(const s of snapshots){const o=elem('option',s.iteration.toLocaleString());o.value=String(s.iteration);$('step').append(o);}
      $('step').value=$('follow').checked?String(last.iteration):selected;
      if(!$('step').value)$('step').value=String(last.iteration);latestKey=key;show();
    }
    $('status').textContent='Latest saved: step '+last.iteration.toLocaleString()+' · '+new Date(last.created_unix*1000).toLocaleString()+
      ' · '+snapshots.length+' saved previews · Checks for updates every 5 s';
  }catch(e){$('status').textContent='Cannot load previews: '+e.message+'. Open this page through the preview HTTP server.';}
}
$('step').onchange=()=>{$('follow').checked=false;show();};$('split').onchange=show;
$('follow').onchange=()=>{if($('follow').checked&&snapshots.length){$('step').value=String(snapshots[snapshots.length-1].iteration);show();}};
$('refresh').onclick=refresh;refresh();setInterval(refresh,5000);
</script></body></html>'''
