"""Checkpoint publication, full-trajectory fidelity and concurrent worker recovery."""
import argparse
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts import run_soft_surface_long as long
from scripts import render
from utils.checkpoint_snapshots import atomic_write, atomic_json, publish_snapshot, read_snapshot
from utils.soft_surface import sample_rgb_gradients, added_gradient_stats
from test_scene_occlusion import manifest
from test_view_comparisons import dataset, checkpoint


class PeriodicTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.keyframes, self.names, self.selected = dataset(self.root)
        self.scene = manifest(self.root)
        report = json.loads(self.scene.read_text())
        report.update(source=str(self.source), training_images=self.selected)
        atomic_json(self.scene, report)
        self.model = self.root/'long'
        self.base = self.root/'base'
        checkpoint(self.base, 30000)

    def prepare(self, iterations=20000):
        argv = ['--source', str(self.source), '--model', str(self.model), '--keyframes', str(self.keyframes),
                '--scene-geometry', str(self.scene), '--baseline', str(self.base), '--gpu', '1',
                '--iterations', str(iterations), '--expected-frames', '16']
        long.prepare(argv)
        return long.load_plan(self.model)

    def training_files(self):
        self.model.mkdir(exist_ok=True)
        (self.model/'keyframes.json').write_bytes(self.keyframes.read_bytes())
        (self.model/'cfg_args').write_text(repr(argparse.Namespace(source_path=str(self.source),
            model_type='3dgs', eval=False, keyframe_manifest='keyframes.json', antialiasing=True)))

    def publish(self, iteration):
        self.training_files()
        pc = SimpleNamespace(get_xyz=torch.tensor([[float(iteration), 0, 2.]]))
        pc.capture = lambda: (pc.get_xyz.clone(), {'state': {'momentum': torch.ones(1,3)}})
        def save(step):
            path = self.model/f'point_cloud/iteration_{step}/point_cloud.ply'
            atomic_write(path, lambda temp: temp.write_text(json.dumps(pc.get_xyz.tolist())))
            self.assertIsNone(read_snapshot(self.model, step))
        scene = SimpleNamespace(model_path=str(self.model), gaussians=pc, save=save)
        return publish_snapshot(scene, iteration)

    def test_publication_exposes_matching_state_and_rejects_mutations(self):
        snapshot = self.publish(10000)
        self.assertEqual(read_snapshot(self.model,10000), snapshot)
        state, iteration = torch.load(self.model/'chkpnt10000.pth', weights_only=True)
        self.assertEqual(iteration, 10000)
        self.assertEqual(state[0].tolist(), json.loads((self.model/'point_cloud/iteration_10000/point_cloud.ply').read_text()))
        with self.assertRaises(FileExistsError):
            self.publish(10000)
        path = self.model/'point_cloud/iteration_10000/point_cloud.ply'
        path.write_text(path.read_text().replace('10000','20000'))  # Same size, different weights.
        with self.assertRaisesRegex(ValueError,'fingerprint'):
            read_snapshot(self.model, 10000)

    def test_interrupted_files_never_publish_a_ready_checkpoint(self):
        with patch('torch.save', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.publish(10000)
        self.assertIsNone(read_snapshot(self.model,10000))
        self.assertFalse(list(self.model.rglob('*.tmp')))
        target = self.root/'existing'; target.write_text('complete')
        def broken(temp):
            temp.write_text('partial')
            raise RuntimeError('interrupted')
        with self.assertRaises(RuntimeError):
            atomic_write(target, broken)
        self.assertEqual(target.read_text(),'complete')

    def test_100k_plan_preserves_schedule_and_has_no_100k_baseline_requirement(self):
        data = self.prepare(100000)
        self.assertEqual(data['iterations'], list(range(10000,100001,10000)))
        schedule = data['plan']['training_schedule']
        self.assertEqual((schedule['position_warmup_steps'],schedule['position_ramp_steps'],schedule['position_lr_max_steps']),
                         (7000,2000,21000))
        self.assertAlmostEqual(schedule['position_lr_init'],3.2e-6)
        self.assertAlmostEqual(schedule['position_lr_final'],3.2e-8)
        self.assertEqual([s['name'] for s in data['plan']['stages']], ['train'])
        command = data['plan']['stages'][0]['argv']
        self.assertEqual(command[command.index('--test_iterations')+1],'-1')
        self.assertEqual(command[command.index('--occlusion_mode')+1],'off')
        self.assertIn('--publish_checkpoint_snapshots',command)
        save_start, checkpoint_start = command.index('--save_iterations'), command.index('--checkpoint_iterations')
        self.assertEqual(list(map(int,command[save_start+1:checkpoint_start])), data['iterations'])
        self.assertEqual(list(map(int,command[checkpoint_start+1:command.index('--publish_checkpoint_snapshots')])),data['iterations'])
        with self.assertRaises(FileExistsError):
            self.prepare(100000)

    def test_prepare_rejects_partial_dataset_before_starting_any_worker(self):
        with self.assertRaisesRegex(ValueError,'Expected 5009'):
            long.prepare(['--source', str(self.source), '--model',str(self.model),'--keyframes',str(self.keyframes),
                          '--scene-geometry',str(self.scene)])
        self.assertFalse(long.directory(self.model).exists())

    def write_fake_evaluation(self, command):
        iteration = int(command[command.index('--iteration')+1])
        if command[0].endswith('eval_all_frames.py'):
            data = long.load_plan(self.model)
            folder = self.model/f'psnr_all_frames/iteration_{iteration}'
            atomic_json(folder/'summary.json', dict(iteration=iteration, model=str(self.model), source=str(self.source),
                model_type='3dgs',antialiasing=True,occlusion=dict(occlusion_mode='off'),
                counts=dict(all=16,selected=6,unselected=10), mean_psnr_db=dict(all=23.,selected=24.,unselected=22.)))
            with (folder/'per_frame.csv').open('w',newline='') as stream:
                writer=csv.writer(stream); writer.writerow(['image_name','selected_for_training','psnr_db'])
                writer.writerows((n,int(n in data['selected']),23.) for n in data['frames'])
        else:
            self.assertEqual(command[command.index('--split')+1],'both')
            self.assertEqual(command[command.index('--eval-set')+1],'unselected')
            self.assertEqual(command[command.index('--gpu')+1],'1')
            folder = self.model/f'view_comparisons/iteration_{iteration}'
            folder.mkdir(parents=True,exist_ok=True)
            (folder/'index.html').write_text('GT | render | error')

    def test_serial_worker_retries_oom_and_resumes_without_skipping_or_repeating(self):
        data = self.prepare()
        self.publish(10000); self.publish(20000)
        folder = long.directory(self.model)
        atomic_json(folder/'training.json', dict(state='running',pid=os.getpid()))
        calls=[]
        def invoke(command):
            iteration = int(command[command.index('--iteration')+1])
            calls.append((command[0],iteration))
            if len(calls)==1:
                raise torch.cuda.OutOfMemoryError('test GPU is temporarily full')
            self.write_fake_evaluation(command)
        def after_wait(_):
            atomic_json(folder/'training.json',dict(state='trained'))
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_per_process_memory_fraction') as cap, \
             patch('torch.cuda.empty_cache'), patch.object(long.time,'sleep',side_effect=after_wait), \
             patch.object(long.run_occlusion,'invoke',side_effect=invoke):
            self.assertEqual(long.watch(self.model),0)
            cap.assert_called_once_with(.20)
        self.assertEqual(calls,[('scripts/eval_all_frames.py',10000),('scripts/eval_all_frames.py',10000),
                               ('scripts/render.py',10000),('scripts/eval_all_frames.py',20000),('scripts/render.py',20000)])
        history=json.loads((folder/'history.json').read_text())
        self.assertTrue(history['complete'])
        self.assertEqual([r['iteration'] for r in history['completed']], [10000,20000])
        with patch.object(long.run_occlusion,'invoke',side_effect=AssertionError('rerendered completed snapshot')), \
             patch('torch.cuda.is_available',side_effect=AssertionError('initialized CUDA while replaying results')):
            long.watch(self.model)
        self.assertEqual(json.loads((folder/'worker.json').read_text())['state'],'complete')

    def test_missing_snapshot_or_partial_metrics_is_a_failure(self):
        data=self.prepare()
        self.publish(10000)
        folder=long.directory(self.model)
        atomic_json(folder/'training.json',dict(state='trained'))
        with patch('torch.cuda.is_available',return_value=True), patch('torch.cuda.set_per_process_memory_fraction'), \
             patch.object(long.run_occlusion,'invoke',side_effect=self.write_fake_evaluation):
            with self.assertRaisesRegex(RuntimeError,'before publishing iteration 20000'):
                long.watch(self.model)
        self.assertEqual(json.loads((folder/'worker.json').read_text())['completed'],[10000])
        metric=self.model/'psnr_all_frames/iteration_10000'
        summary=json.loads((metric/'summary.json').read_text())
        summary['counts']['all']=15
        with self.assertRaisesRegex(ValueError,'Partial evaluation'):
            long.verify_metrics(data,10000,summary,metric/'per_frame.csv')
        summary['counts']['all']=16
        lines=(metric/'per_frame.csv').read_text().splitlines()
        (metric/'per_frame.csv').write_text('\n'.join(lines[:-1]+[lines[1]])+'\n')
        with self.assertRaisesRegex(ValueError,'omits or duplicates'):
            long.verify_metrics(data,10000,summary,metric/'per_frame.csv')

    def test_interrupted_preview_is_not_marked_complete(self):
        data=self.prepare(); snapshot=self.publish(10000)
        def invoke(command):
            if command[0].endswith('render.py'):
                raise RuntimeError('preview failed')
            self.write_fake_evaluation(command)
        with patch.object(long.run_occlusion,'invoke',side_effect=invoke):
            with self.assertRaisesRegex(RuntimeError,'preview failed'):
                long.evaluate(data,snapshot,0)
        self.assertIsNone(long.completed_result(self.model,10000,snapshot))

    def test_unselected_previews_exist_even_without_native_eval_split(self):
        self.prepare(); self.publish(10000)
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            render.main(['--model',str(self.model),'--iteration','10000','--split','both',
                         '--eval-set','unselected','--max-views','3','--dry-run'])
        self.assertIn('train: 3/6',output.getvalue())
        self.assertIn('eval: 3/10',output.getvalue())

    def test_gradient_audit_is_deterministic_and_measures_added_force(self):
        pc=SimpleNamespace(_xyz=torch.nn.Parameter(torch.zeros(10,3)),_opacity=torch.nn.Parameter(torch.zeros(10,1)))
        pc._xyz.grad=torch.ones_like(pc._xyz)
        pc._opacity.grad=torch.ones_like(pc._opacity)
        state=torch.random.get_rng_state().clone()
        sample=sample_rgb_gradients(pc,torch.ones(10,dtype=torch.bool),limit=4)
        torch.testing.assert_close(state,torch.random.get_rng_state())
        self.assertEqual(sample[0].tolist(),[0,3,6,9])
        pc._opacity.grad.mul_(3)
        stats=added_gradient_stats(pc,sample)
        self.assertEqual(stats['soft_to_rgb_opacity_grad_ratio'],2.)
        self.assertEqual(stats['soft_xyz_grad_norm'],0.)
        pc._xyz.grad=None
        sample=sample_rgb_gradients(pc,torch.arange(10))
        self.assertIsNone(added_gradient_stats(pc,sample)['soft_to_rgb_xyz_grad_ratio'])

    def test_worker_lock_prevents_duplicate_evaluators(self):
        path=self.root/'lock'
        with long.process_lock(path):
            with self.assertRaisesRegex(RuntimeError,'already holds'):
                with long.process_lock(path):
                    pass


if __name__=='__main__':
    unittest.main()
