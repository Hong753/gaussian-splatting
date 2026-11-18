"""Pair full-image PSNR and the Orange4F floor crop with the original guided PGSR run."""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from utils.checkpoint_snapshots import atomic_json


def frame_scores(folder, split):
    with (folder/'per_frame.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    values = {(r['split'], r['image_name']): float(r['psnr_db']) for r in rows}
    expected = {(group, name) for group, names in [('train', split['train']), ('held-out', split['test'])] for name in names}
    if len(values) != len(rows) or values.keys() != expected:
        raise ValueError(f'Incomplete or duplicate PGSR evaluation at {folder}')
    return values


def floor_crop(model, step):
    root = model/'debug_previews'
    path = root/f'iteration_{step:06d}'/'frames.json'
    if not path.is_file():
        return None
    for row in json.loads(path.read_text())['frames']:
        if row['split'] == 'train' and Path(row['name']).name == '09882.jpg':
            pair = []
            for key in ('gt', 'render'):
                with Image.open(root/row[key]) as image:
                    w, h = image.size
                    # Foreground gray speckled floor; bounds are normalized to the reference image.
                    box = tuple(round(x*n) for x, n in zip((.62, .82, .95, .95), (w, h, w, h)))
                    pair.append(np.asarray(image.convert('L').crop(box), dtype=float)/255.)
            return pair
    return None


def gradient_error(gt, pred):
    reference = [np.diff(gt, axis=axis) for axis in (0, 1)]
    error = sum(np.abs(a-np.diff(pred, axis=axis)).sum() for axis, a in enumerate(reference))
    return float(error / max(1e-12, sum(np.abs(a).sum() for a in reference)))


def compare(model, baseline, step):
    folder = Path('psnr_all_frames')/f'iteration_{step}'
    if not (baseline/folder/'summary.json').is_file():
        print(f'Paired comparison awaits {baseline/folder}. Rerun the launcher once the control is ready.', flush=True)
        return
    plans = [json.loads((p/'pgsr_run.json').read_text()) for p in (baseline, model)]
    fields = ('fingerprints', 'variant', 'upstream_commit', 'patch_sha256', 'warmup', 'ramp',
              'position_lr_endpoints', 'normal_step', 'scale_cap_over_initial_sigma', 'max_all_points',
              'exposure', 'pose_refinement')
    if any(plans[0][k] != plans[1][k] for k in fields) or 'depth_supervision' in plans[0]:
        raise ValueError('Control and depth run have incompatible protocols')
    split = json.loads((model/'dataset/split.json').read_text())
    if json.loads((baseline/'dataset/split.json').read_text()) != split:
        raise ValueError('Control and depth run use different splits')
    control, trial = [frame_scores(p/folder, split) for p in (baseline, model)]
    results = dict(iteration=step, control=str(baseline), original_poses=True, splits={})
    for group in ('held-out', 'train'):
        keys = [k for k in control if k[0] == group]
        if keys:
            a, b = (float(np.mean([scores[k] for k in keys])) for scores in (control, trial))
            results['splits'][group] = dict(frames=len(keys), control_psnr_db=a, depth_psnr_db=b, delta_db=b-a)
    pairs = [floor_crop(p, step) for p in (baseline, model)]
    if all(p is not None for p in pairs):
        (gt, pred0), (gt1, pred1) = pairs
        if not np.array_equal(gt, gt1):
            raise ValueError('Preview ground-truth crops differ')
        results['09882_floor_crop'] = dict(split='train', normalized_xyxy=[.62,.82,.95,.95],
            metric='relative luminance gradient error (lower is better)',
            control=gradient_error(gt, pred0), depth=gradient_error(gt, pred1))
    atomic_json(model/folder/'depth_comparison.json', results)
    print(json.dumps(results, indent=2), flush=True)
