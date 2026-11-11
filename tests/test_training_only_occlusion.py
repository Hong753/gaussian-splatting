"""Training keeps the mesh gate; inference and its reports need no mesh asset."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import run_occlusion, render, render_ablation_videos
from scripts.occlusion_options import apply_overrides, inference_pipeline, inference_settings, metric_directory
from scripts.view_comparisons import atomic_json
from utils.lidar_occlusion import file_digest
from test_view_comparisons import dataset, checkpoint
from test_scene_occlusion import manifest


class TrainingOnlyOcclusionTests(unittest.TestCase):
    def test_validation_pipeline_does_not_disable_training_or_load_missing_mesh(self):
        provider=object()
        pipe=SimpleNamespace(occlusion_mode='mesh',occlusion_backend='ray',antialiasing=True,
                             _occlusion_provider=provider)
        cfg=dict(occlusion_mode='mesh',occlusion_backend='ray',occlusion_geometry='/missing/scene.json',
                 occlusion_inference_mode='off')
        with patch('utils.scene_occlusion.load_scene',side_effect=AssertionError('Inference accessed the mesh')):
            evaluation=inference_pipeline(pipe,cfg)
            self.assertEqual(evaluation.occlusion_mode,'off')
            self.assertTrue(evaluation.antialiasing)
            self.assertFalse(hasattr(evaluation,'_occlusion_provider'))
            self.assertEqual(inference_settings(SimpleNamespace(**cfg))['occlusion_mode'],'off')
            for requested in (None,'saved','off'):
                spec=apply_overrides(dict(type='3dgs',cfg=cfg.copy()),SimpleNamespace(occlusion=requested))
                self.assertEqual(spec['cfg']['occlusion_mode'],'off')
                self.assertEqual(spec['training_occlusion_mode'],'mesh')
                self.assertNotIn('render_variant',spec)
        self.assertEqual(pipe.occlusion_mode,'mesh')
        self.assertIs(pipe._occlusion_provider,provider)
        self.assertEqual(cfg['occlusion_mode'],'mesh')
        self.assertIs(inference_pipeline(pipe,{**cfg,'occlusion_inference_mode':'saved'}),pipe)
        with self.assertRaisesRegex(ValueError,'inference_mode'):
            inference_settings({**cfg,'occlusion_inference_mode':'typo'})

    def test_explicit_mesh_diagnostic_remains_available_and_separates_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);scene=manifest(root)
            cfg=dict(occlusion_mode='mesh',occlusion_geometry=str(scene),occlusion_inference_mode='off')
            spec=apply_overrides(dict(type='3dgs',cfg=cfg),SimpleNamespace(occlusion='on'))
            self.assertEqual(spec['cfg']['occlusion_mode'],'mesh')
            self.assertTrue(spec['render_variant'].startswith('occlusion_on_'))
            self.assertNotEqual(metric_directory(root,30000,spec['render_variant']),metric_directory(root,30000))

    def test_full_workflow_preserves_culling_and_schedule_but_all_inference_is_off(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source,keyframes,names,selected=dataset(root)
            scene=manifest(root)
            report=json.loads(scene.read_text());report.update(source=str(source),training_images=selected,
                                                             point_cloud=str(root/'input.ply'))
            atomic_json(scene,report)
            atomic_json(root/'visibility/metadata.json',dict(complete=True,scene_sha256=file_digest(scene),images=names))
            baseline=root/'baseline';baseline.mkdir()
            (baseline/'cfg_args').write_text('Namespace(model_type="3dgs", antialiasing=True)')
            checkpoint(baseline,30000)
            model=root/'trained'
            argv=['--source',str(source),'--model',str(model),'--keyframes',str(keyframes),
                  '--baseline',str(baseline),'--scene-geometry',str(scene),'--occlusion-backend','ray',
                  '--inference-occlusion','off','--position-warmup-steps','7000',
                  '--position-ramp-steps','2000','--position-lr-scale','.02']
            args=run_occlusion.parser().parse_args(argv)
            plan=run_occlusion.build_plan(args)
            self.assertEqual(plan['gpu'],'0')
            self.assertEqual(plan['inference_occlusion'],'off')
            train=plan['stages'][0]['argv']
            self.assertEqual(train[train.index('--occlusion_mode')+1],'mesh')
            self.assertEqual(train[train.index('--occlusion_inference_mode')+1],'off')
            self.assertEqual(train[train.index('--position_warmup_steps')+1],'7000')
            self.assertEqual(train[train.index('--densification_phase_offset')+1],'7000')
            self.assertAlmostEqual(float(train[train.index('--position_lr_init')+1]),.00016*.02)
            self.assertEqual([s['name'] for s in plan['stages']],
                ['train','trained_off_psnr','trained_off_images','baseline_off_psnr','baseline_off_images','random_video','actual_video'])
            for stage in plan['stages'][1:]:
                cmd=stage['argv'];mode=cmd[cmd.index('--occlusion')+1]
                self.assertEqual(mode,'off')
                if stage['name'].endswith('_video'):
                    self.assertEqual(cmd[cmd.index('--occlusion')+2],'off')
                elif stage['name'].endswith('_images'):
                    self.assertEqual(cmd[cmd.index('--split')+1],'both')
            # One command must finish the diagnostic partial cache before training.
            atomic_json(root/'visibility/metadata.json',dict(complete=False,scene_sha256=file_digest(scene),images=names))
            prepare_args=run_occlusion.parser().parse_args(argv+['--prepare-visibility'])
            prepared=run_occlusion.build_plan(prepare_args)
            self.assertEqual([s['name'] for s in prepared['stages'][:2]],['visibility','train'])
            visibility=prepared['stages'][0]['argv']
            self.assertEqual(visibility[visibility.index('--split')+1],'all')
            self.assertEqual(visibility[visibility.index('--gpu')+1],'0')
            # Evaluation-only must recover the saved policy without the mesh or
            # its visibility cache. This also covers a model copied off-server.
            checkpoint(model,30000)
            (model/'cfg_args').write_text("Namespace(occlusion_mode='mesh', occlusion_backend='ray', occlusion_inference_mode='off')")
            scene.unlink()
            resumed=run_occlusion.parser().parse_args([v for v in argv if v not in ('--inference-occlusion','off')]+['--evaluate-only'])
            evaluation=run_occlusion.build_plan(resumed)
            self.assertEqual(evaluation['inference_occlusion'],'off')
            self.assertFalse(any(s['name']=='train' for s in evaluation['stages']))

    def test_saved_policy_reaches_image_and_video_entry_points_without_mesh(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source,keyframes,_,_=dataset(root)
            models=[]
            for label,mode in [('baseline','off'),('trained','mesh')]:
                path=root/label;path.mkdir();checkpoint(path,30000)
                cfg=dict(source_path=str(source),eval=True,keyframe_manifest='keyframes.json',antialiasing=True,
                         occlusion_mode=mode,occlusion_geometry=str(root/'absent.json'),occlusion_inference_mode='off')
                (path/'cfg_args').write_text(repr(SimpleNamespace(**cfg)).replace('namespace(','Namespace(',1))
                (path/'keyframes.json').write_bytes(keyframes.read_bytes());models.append(path)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(render.main(['--models',str(models[1]),'--split','both','--dry-run']),0)
            for video in ('random','actual'):
                args=render_ablation_videos.parser().parse_args(['--video',video,'--source',str(source),
                    '--models',*map(str,models),'--output',str(root/video),'--duration','1','--fps','5'])
                plan=render_ablation_videos.build_plan(args)
                self.assertEqual([m['cfg']['occlusion_mode'] for m in plan['models']],['off','off'])
                self.assertEqual(plan['models'][1]['training_occlusion_mode'],'mesh')
                self.assertTrue(all(m['inference_occlusion_resolved'] for m in plan['models']))


if __name__=='__main__':
    unittest.main()
