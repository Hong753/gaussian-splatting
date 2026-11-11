"""Cold imports must work without relying on training to preload scene first."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
# Only the native KNN extension is unavailable in the CPU test environment.
# The actual scene, camera_utils and cameras modules must remain unmocked.
PREAMBLE = """
import sys, types
knn = types.ModuleType('simple_knn')
knn.__path__ = []
extension = types.ModuleType('simple_knn._C')
def unavailable(*args, **kwargs):
    raise AssertionError('Native KNN must not execute during camera loading')
extension.distCUDA2 = unavailable
sys.modules['simple_knn'] = knn
sys.modules['simple_knn._C'] = extension
"""


class CameraImportTests(unittest.TestCase):
    def run_fresh(self, script):
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(ROOT/'tests'), env.get('PYTHONPATH', '')])
        result = subprocess.run([sys.executable, '-c', PREAMBLE+textwrap.dedent(script)],
            cwd=ROOT, env=env, text=True, capture_output=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)

    def test_scene_and_camera_modules_allow_either_import_order(self):
        for first in ('utils.camera_utils', 'scene', 'scene.cameras'):
            with self.subTest(first=first):
                self.run_fresh(f"""
                    import importlib
                    importlib.import_module({first!r})
                    from utils.camera_utils import loadCam, cameraList_from_camInfos, camera_to_JSON
                    from scene.cameras import Camera
                    from scene import Scene
                    assert all(callable(x) for x in (loadCam, cameraList_from_camInfos, camera_to_JSON, Camera, Scene))
                """)

    def test_in_process_all_frame_render_uses_real_camera_loader_from_cold_start(self):
        self.run_fresh("""
            import hashlib, json, tempfile
            from pathlib import Path
            from unittest.mock import patch
            import numpy as np
            import torch
            from scripts import render
            from test_view_comparisons import dataset, checkpoint
            from scripts import render_model_comparison

            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, selected, names, _ = dataset(root)
                model = root/'model'
                model.mkdir()
                checkpoint(model, 30000)
                (model/'keyframes.json').write_bytes(selected.read_bytes())
                mesh = root/'mesh.npz'
                np.savez(mesh, vertices=np.array([[0,0,2], [1,0,2], [0,1,2]], np.float32),
                         faces=np.array([[0,1,2]], np.int32))
                geometry = root/'scene.json'
                geometry.write_text(json.dumps(dict(format='scene_occlusion_mesh_v1', complete=True,
                    mesh=mesh.name, mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(), relative_margin=.05)))
                (model/'cfg_args').write_text(
                    f"Namespace(source_path={str(source)!r}, eval=True, keyframe_manifest='keyframes.json', "
                    f"antialiasing=True, occlusion_mode='mesh', occlusion_inference_mode='off', "
                    f"occlusion_backend='ray', occlusion_geometry={str(geometry)!r})")
                seen = []
                def renderer(spec, aa):
                    assert spec['cfg']['occlusion_mode'] == 'mesh'
                    assert spec['cfg']['occlusion_backend'] == 'ray'
                    assert spec['inference_occlusion_resolved'] and aa
                    def predict(camera):
                        from scene.cameras import Camera
                        assert isinstance(camera, Camera)
                        seen.append(camera.image_name)
                        return .8*camera.original_image[:3]+.05
                    return predict
                assert 'utils.camera_utils' not in sys.modules and 'scene' not in sys.modules
                # Real image/camera/report paths; only CUDA projection transfers
                # and the rasterizer are replaced for this CPU regression.
                with patch.object(torch.Tensor, 'cuda', lambda tensor, *a, **kw: tensor), \
                     patch.object(render_model_comparison, 'load_renderer', side_effect=renderer):
                    assert render.main(['--model', str(model), '--iteration', '30000', '--split', 'all',
                        '--antialiasing', '--occlusion', 'on', '--occlusion-backend', 'ray',
                        '--occlusion-geometry', str(geometry), '--gpu', '0', '--in-process',
                        '--output', str(root/'report')]) == 0
                assert set(seen) == set(names)
                assert (root/'report/all/index.html').is_file()
                assert len(json.loads((root/'report/all/model_0/frames.json').read_text())) == len(names)
                assert len(list((model/'psnr_all_frames/iteration_30000').glob('occlusion_on_*/summary.json'))) == 1
        """)


if __name__ == '__main__':
    unittest.main()
