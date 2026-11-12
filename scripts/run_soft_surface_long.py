#!/usr/bin/env python3
"""100k soft-surface training and a restartable, sequential checkpoint evaluator.

Use run_soft_surface_long.sh to start two independent processes on the same GPU.
No training process forks CUDA workers. Evaluation is native AA, without a mesh.
"""
import argparse
import csv
import fcntl
import html
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from contextlib import contextmanager
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_occlusion, run_soft_surface
from scripts.view_comparisons import read_colmap_records
from utils.checkpoint_snapshots import atomic_json, atomic_write, digest, read_snapshot


def directory(model):
    return Path(model).expanduser().resolve()/'periodic_evaluation'


def prepare(argv):
    parser = run_soft_surface.parser()
    parser.set_defaults(iterations=100000, position_lr_max_steps=21000,
                        checkpoint_every=10000, training_only=True)
    parser.add_argument('--expected-frames', type=int, default=5009)
    parser.add_argument('--eval-memory-fraction', type=float, default=.20,
                        help='Evaluator allocator cap; does not reserve VRAM from training')
    parser.add_argument('--poll-seconds', type=float, default=15.)
    args = parser.parse_args(argv)
    if args.baseline and not args.keyframes:
        args.keyframes = args.baseline/'keyframes.json'
    if (args.evaluate_only or not args.training_only or not args.soft_surface or
            args.checkpoint_every < 1 or not args.keyframes or args.expected_frames < 1 or
            not 0 < args.eval_memory_fraction <= 1 or not 0 < args.poll_seconds <= 60):
        parser.error('Need a fresh soft-surface run, keyframes, positive frame/checkpoint counts and valid evaluator limits')
    plan = run_occlusion.build_plan(args)
    return prepare_experiment(args, plan, 'soft_surface', [args.keyframes, args.scene_geometry])


def prepare_experiment(args, plan, experiment, input_paths):
    """Shared durable checkpoint queue; experiment launchers own their training plan."""
    records = read_colmap_records(Path(plan['source']))
    if len(records) != args.expected_frames:
        raise ValueError(f'Expected {args.expected_frames} registered frames, found {len(records)}')
    missing = [n for n, r in records.items() if not Path(r['image_path']).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing source images: {missing[:5]}')
    selected = json.loads(args.keyframes.read_text())['selected_images']
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= set(records):
        raise ValueError('Training manifest is empty, duplicated or outside the source cameras')
    steps = sorted(set(range(args.checkpoint_every, args.iterations+1, args.checkpoint_every)) | {args.iterations})
    data = dict(schema=1, experiment=experiment, model=plan['model'], source=plan['source'], gpu=args.gpu,
        iterations=steps, frames=sorted(records), selected=sorted(selected),
        expected_frames=args.expected_frames, report_max_views=args.report_max_views,
        memory_fraction=args.eval_memory_fraction, poll_seconds=args.poll_seconds,
        training_args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        plan=plan, input_fingerprints={str(path.resolve()): digest(path) for path in input_paths})
    if args.dry_run:
        print(json.dumps(data, indent=2))
        return 0
    folder = directory(plan['model'])
    # Exclusive creation prevents accidental reuse of an old run/worker state.
    folder.mkdir(parents=True, exist_ok=False)
    atomic_json(folder/'plan.json', data)
    atomic_json(folder/'training.json', dict(state='prepared'))
    print(f'Prepared {len(steps)} checkpoints; {len(records)} frames each; GPU {args.gpu}.', flush=True)
    print(f'Plan: {folder / "plan.json"}', flush=True)
    return 0


@contextmanager
def process_lock(path):
    with Path(path).open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another process already holds {path}') from exc
        yield


def load_plan(model):
    plan = json.loads((directory(model)/'plan.json').read_text())
    if plan.get('schema') != 1 or Path(plan['model']) != Path(model).expanduser().resolve():
        raise ValueError('Periodic plan belongs to a different model or schema')
    return plan


def train(model):
    data, folder = load_plan(model), directory(model)
    with process_lock(folder/'trainer.lock'):
        try:
            for path, expected in data['input_fingerprints'].items():
                if digest(path) != expected:
                    raise ValueError(f'Input changed after preparation: {path}')
            args = SimpleNamespace(**data['training_args'])
            for name in ('source', 'model', 'keyframes', 'baseline', 'video_output', 'scene_geometry', 'point_cloud', 'cache', 'local_scale_cache'):
                if getattr(args, name, None) is not None:
                    setattr(args, name, Path(getattr(args, name)))
            launcher = run_occlusion
            if data.get('experiment') == 'fixed_lidar':
                from scripts import run_fixed_lidar
                launcher = run_fixed_lidar
            elif data.get('experiment') == 'discrete_lidar':
                from scripts import run_discrete_lidar
                launcher = run_discrete_lidar
            elif data.get('experiment') == 'local_scale':
                from scripts import run_local_scale
                launcher = run_local_scale
            plan = launcher.build_plan(args)
            if plan != data['plan']:
                raise ValueError('Training plan changed after preparation')
            atomic_json(folder/'training.json', dict(state='running', pid=os.getpid()))
            launcher.execute_plan(args, plan)
        except BaseException as exc:
            atomic_json(folder/'training.json', dict(state='failed', error=str(exc), pid=os.getpid()))
            raise
        atomic_json(folder/'training.json', dict(state='trained', pid=os.getpid()))
    return 0


def soft_diagnostics(model, start, end):
    path = Path(model)/'soft_surface_training.jsonl'
    rows = []
    if path.is_file():
        with path.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # Trainer may be appending its latest line.
                if start < row['iteration'] <= end:
                    rows.append(row)
    result = dict(logged_views=len(rows), interval=[start, end], warnings=[])
    if not rows:
        result['warnings'].append('No soft-surface diagnostics in this interval')
        return result
    for field in ('reference_fraction', 'surface_opacity_mean', 'surface_gaussians',
                  'captured_gaussians', 'normal_distance_mean', 'weighted_loss',
                  'soft_opacity_grad_norm', 'soft_xyz_grad_norm',
                  'soft_to_rgb_opacity_grad_ratio', 'soft_to_rgb_xyz_grad_ratio'):
        values = [r[field] for r in rows if r.get(field) is not None and math.isfinite(r[field])]
        if values:
            result[field] = dict(min=min(values), median=statistics.median(values), max=max(values))
    if not any(r.get('surface_gaussians', 0) > 0 for r in rows):
        result['warnings'].append('No surface Gaussians matched in any logged view; inspect reference alignment')
    if not any(r.get('reference_fraction', 0) > 0 for r in rows):
        result['warnings'].append('No supported reference pixels in any logged view')
    gradient_rows = [r for r in rows if 'soft_opacity_grad_norm' in r]
    if gradient_rows and not any(r['soft_opacity_grad_norm'] > 0 for r in gradient_rows):
        result['warnings'].append('Sampled opacity gradients are zero: distinguish saturation from missing support')
    result['gradient_protocol'] = 'deterministic sample of at most 2048 visible Gaussians, every 100 steps'
    return result


def verify_metrics(data, iteration, summary, csv_path):
    if (summary['iteration'] != iteration or summary['model'] != data['model'] or
            summary['source'] != data['source'] or summary['model_type'] != '3dgs' or
            not summary['antialiasing'] or summary['occlusion']['occlusion_mode'] != 'off'):
        raise ValueError('Evaluation used the wrong checkpoint or render protocol')
    expected = dict(all=len(data['frames']), selected=len(data['selected']),
                    unselected=len(data['frames'])-len(data['selected']))
    if summary['counts'] != expected:
        raise ValueError(f'Partial evaluation: got {summary["counts"]}, expected {expected}')
    with Path(csv_path).open() as stream:
        rows = list(csv.DictReader(stream))
    if sorted(r['image_name'] for r in rows) != data['frames']:
        raise ValueError('Per-frame CSV omits or duplicates source views')
    if sorted(r['image_name'] for r in rows if r['selected_for_training'] == '1') != data['selected']:
        raise ValueError('Per-frame CSV uses a different training split')
    for value in summary['mean_psnr_db'].values():
        if value is not None and not math.isfinite(float(value)):
            raise ValueError('Nonfinite PSNR in real-scene evaluation')


def evaluate(data, snapshot, start):
    iteration, model = snapshot['iteration'], Path(data['model'])
    metric = model/f'psnr_all_frames/iteration_{iteration}'
    previews = model/f'view_comparisons/iteration_{iteration}'
    run_occlusion.invoke(['scripts/eval_all_frames.py', '--source', data['source'], '--model', str(model),
        '--iteration', str(iteration), '--antialiasing', '--occlusion', 'off',
        '--expected-frames', str(data['expected_frames'])])
    summary = json.loads((metric/'summary.json').read_text())
    verify_metrics(data, iteration, summary, metric/'per_frame.csv')
    run_occlusion.invoke(['scripts/render.py', '--model', str(model), '--iteration', str(iteration),
        '--split', 'both', '--eval-set', 'unselected', '--max-views', str(data['report_max_views']),
        '--antialiasing', '--occlusion', 'off', '--gpu', data['gpu'], '--in-process', '--output', str(previews)])
    if read_snapshot(model, iteration) != snapshot:
        raise ValueError('Snapshot changed during evaluation')
    if not (previews/'index.html').is_file():
        raise FileNotFoundError('Image comparisons were not produced')
    summary['checkpoint'] = snapshot
    atomic_json(metric/'summary.json', summary)
    result = dict(iteration=iteration, snapshot_id=snapshot['snapshot_id'], gaussians=snapshot['gaussians'],
        checkpoint_files=snapshot['files'], counts=summary['counts'], mean_psnr_db=summary['mean_psnr_db'],
        metrics=str(metric/'summary.json'), per_frame=str(metric/'per_frame.csv'), previews=str(previews/'index.html'),
        metrics_sha256=digest(metric/'summary.json'), per_frame_sha256=digest(metric/'per_frame.csv'),
        diagnostics=experiment_diagnostics(data, start, iteration))
    atomic_json(directory(model)/f'iteration_{iteration}.json', result)
    return result


def experiment_diagnostics(data, start, end):
    if data.get('experiment') == 'discrete_lidar':
        rows, events = [], []
        for filename, destination in (('discrete_lidar_training.jsonl', rows), ('discrete_lidar_growth.jsonl', events)):
            path = Path(data['model'])/filename
            if path.is_file():
                for line in path.read_text().splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if start < row['iteration'] <= end:
                        destination.append(row)
        good = bool(rows) and all(r['positions_in_pool'] and r['unique_coordinates'] and r['xyz_frozen'] for r in rows)
        warnings = [] if good else ['Missing or failed discrete LiDAR coordinate checks']
        if not events or not any(r['added'] for r in events):
            warnings.append('No candidate activations in this interval; inspect gradient threshold and unused-neighbor coverage')
        if rows and rows[-1]['gaussians'] < rows[0]['gaussians']*.5:
            warnings.append('More than half the active population was lost in this interval')
        return dict(interval=[start, end], logged_checks=len(rows), passed=good, warnings=warnings,
                    added=sum(r['added'] for r in events), scale_pruned=sum(r['scale_pruned'] for r in events),
                    latest=rows[-1] if rows else None)
    if data.get('experiment') not in ('fixed_lidar', 'local_scale'):
        return soft_diagnostics(data['model'], start, end)
    path = Path(data['model'])/'fixed_lidar_training.jsonl'
    rows = []
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if start < row['iteration'] <= end:
                rows.append(row)
    regularize = data['plan'].get('scale_control') == 'regularize'
    hybrid = data['plan'].get('scale_control') == 'hybrid'
    good = bool(rows) and all(r['positions_unchanged'] and
        (r.get('scales_finite', False) if regularize else r['scales_within_bounds']) for r in rows)
    loss_rows = []
    loss_path = Path(data['model'])/'scale_regularization_training.jsonl'
    if (regularize or hybrid) and loss_path.is_file():
        for line in loss_path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if start < row['iteration'] <= end:
                loss_rows.append(row)
    regularizer = {}
    for key in ('scale_regularization_loss', 'regularizer_scale_grad_norm',
                'regularizer_to_rgb_scale_grad_ratio'):
        values = [r[key] for r in loss_rows if r.get(key) is not None]
        if values:
            regularizer[key] = dict(min=min(values), median=statistics.median(values), max=max(values))
    warnings = [] if good else ['Missing or failed fixed LiDAR invariant checks']
    if (regularize or hybrid) and not loss_rows:
        warnings.append('No scale regularization diagnostics in this interval')
    elif (regularize or hybrid) and not any(r.get('scale_regularization_loss', 0) > 0 for r in loss_rows):
        warnings.append('No logged positive scale penalty; check whether any scales exceed the target')
    if hybrid and rows and rows[-1].get('local_scale_audit_samples_omitted', 0):
        warnings.append('Some whole overlap-audit samples exceeded the pair budget; reported counts cover retained samples only')
    return dict(interval=[start, end], logged_checks=len(rows), passed=good,
                scale_control=data['plan'].get('scale_control', 'clamp'), regularizer=regularizer,
                latest=rows[-1] if rows else None,
                warnings=warnings)


def completed_result(model, iteration, snapshot):
    path = directory(model)/f'iteration_{iteration}.json'
    if not path.is_file():
        return None
    result = json.loads(path.read_text())
    if result.get('snapshot_id') != snapshot['snapshot_id'] or result.get('checkpoint_files') != snapshot['files']:
        return None
    for key in ('metrics', 'per_frame'):
        if not Path(result[key]).is_file() or digest(result[key]) != result[key+'_sha256']:
            return None
    return result if Path(result['previews']).is_file() else None


def write_history(data, results):
    folder = directory(data['model'])
    rows = [dict(iteration=r['iteration'], gaussians=r['gaussians'],
                 **{k+'_psnr_db': r['mean_psnr_db'][k] for k in ('all', 'selected', 'unselected')},
                 frames=r['counts']['all'], snapshot_id=r['snapshot_id']) for r in results]
    atomic_json(folder/'history.json', dict(expected_iterations=data['iterations'], completed=rows,
        complete=len(rows)==len(data['iterations']), render_protocol='3DGS + AA; no mesh/culling; no exposure'))
    def save_csv(temp):
        with temp.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['iteration', 'gaussians', 'all_psnr_db', 'selected_psnr_db',
                                                       'unselected_psnr_db', 'frames', 'snapshot_id'])
            writer.writeheader(); writer.writerows(rows)
    atomic_write(folder/'history.csv', save_csv)
    body = []
    for result, row in zip(results, rows):
        links = ' '.join(f'<a href="{html.escape(os.path.relpath(result[k], folder))}">{label}</a>'
                         for k, label in [('metrics', 'PSNR'), ('per_frame', 'all frames'), ('previews', 'GT / render / error')])
        links += f' <a href="iteration_{row["iteration"]}.json">Diagnostics</a>'
        cells = [row[k] for k in ('iteration', 'gaussians', 'frames', 'all_psnr_db', 'selected_psnr_db', 'unselected_psnr_db')]
        body.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in cells)+f'<td>{links}</td></tr>')
    title = {'local_scale':'Local KNN scale checkpoints', 'fixed_lidar':'Fixed LiDAR checkpoints',
             'discrete_lidar':'Discrete LiDAR checkpoints'}.get(data.get('experiment'), 'Soft surface checkpoints')
    page = ('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30">'
        f'<title>{title}</title><style>body{{font:16px sans-serif;margin:2em}}td,th{{padding:.6em;text-align:left}}</style>'
        f'<h1>{title}</h1><p>All registered frames; AA on; mesh-free inference. '
        'Selected = training keyframes. Unselected = all remaining frames. Image galleries are sampled.</p>'
        '<table><tr><th>Step</th><th>Gaussians</th><th>Frames</th><th>All PSNR</th><th>Selected PSNR</th>'
        '<th>Unselected PSNR</th><th>Reports</th></tr>'+''.join(body)+'</table>')
    atomic_write(folder/'index.html', lambda temp: temp.write_text(page))


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def watch(model, parent_pid=None, memory_fraction=None):
    data, folder = load_plan(model), directory(model)
    if memory_fraction is not None:
        data['memory_fraction'] = memory_fraction
    os.environ['CUDA_VISIBLE_DEVICES'] = data['gpu']
    results, cuda_ready = [], False
    with process_lock(folder/'evaluator.lock'):
        try:
            write_history(data, results)
            for offset, iteration in enumerate(data['iterations']):
                while True:
                    state = json.loads((folder/'training.json').read_text())
                    stopped = (state['state'] in ('trained', 'failed') or
                               (state.get('pid') and not alive(state['pid'])) or
                               (state['state'] == 'prepared' and parent_pid and not alive(parent_pid)))
                    snapshot = read_snapshot(model, iteration)
                    if snapshot is None:
                        if stopped:
                            raise RuntimeError(f'Trainer stopped before publishing iteration {iteration}; state={state}')
                        atomic_json(folder/'worker.json', dict(state='waiting', iteration=iteration,
                            completed=[r['iteration'] for r in results]))
                        time.sleep(data['poll_seconds'])
                        continue
                    result = completed_result(model, iteration, snapshot)
                    if result is None:
                        if not cuda_ready:
                            import torch
                            if not torch.cuda.is_available():
                                raise RuntimeError('Checkpoint evaluation requires the CUDA-enabled 3dgs environment')
                            torch.cuda.set_per_process_memory_fraction(data['memory_fraction'])
                            cuda_ready = True
                        atomic_json(folder/'worker.json', dict(state='evaluating', iteration=iteration,
                            completed=[r['iteration'] for r in results], pid=os.getpid()))
                        try:
                            result = evaluate(data, snapshot, data['iterations'][offset-1] if offset else 0)
                        except torch.cuda.OutOfMemoryError as exc:
                            # Only one evaluator; preserve every queued checkpoint. Retry after
                            # allocations have been released, including once training finishes.
                            atomic_json(folder/'worker.json', dict(state='waiting_for_memory', iteration=iteration,
                                error=str(exc), completed=[r['iteration'] for r in results]))
                            if stopped:
                                raise RuntimeError('Evaluation exceeds its VRAM cap even after training stopped; '
                                    'restart watch with a larger --eval-memory-fraction') from exc
                            print(f'[{iteration}] Evaluation deferred after OOM; training continues.', flush=True)
                        if result is None:
                            import gc
                            gc.collect(); torch.cuda.empty_cache()
                            time.sleep(data['poll_seconds'])
                            continue
                    results.append(result)
                    write_history(data, results)
                    print(f'[{iteration}] {result["counts"]["all"]} frames: {result["mean_psnr_db"]}', flush=True)
                    for warning in result['diagnostics']['warnings']:
                        print(f'[{iteration}] DIAGNOSTIC: {warning}', flush=True)
                    break
        except BaseException as exc:
            atomic_json(folder/'worker.json', dict(state='failed', error=str(exc), completed=[r['iteration'] for r in results]))
            raise
        atomic_json(folder/'worker.json', dict(state='complete', completed=[r['iteration'] for r in results]))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'prepare':
        return prepare(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'watch'))
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--parent-pid', type=int)
    parser.add_argument('--eval-memory-fraction', type=float,
                        help='Override evaluator cap on restart without changing the training plan')
    args = parser.parse_args(argv)
    if args.phase == 'train':
        return train(args.model)
    if args.eval_memory_fraction is not None:
        if not 0 < args.eval_memory_fraction <= 1:
            parser.error('--eval-memory-fraction must be in (0, 1]')
        # Runtime override only; the checkpoint and training provenance are unchanged.
        return watch(args.model, args.parent_pid, memory_fraction=args.eval_memory_fraction)
    return watch(args.model, args.parent_pid)


if __name__ == '__main__':
    raise SystemExit(main())
