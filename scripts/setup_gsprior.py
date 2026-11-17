#!/usr/bin/env python3
"""Apply audited integration patches to a dedicated pinned upstream checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

COMMIT = 'dcb7c89fb6b60f068b440de45d064ecc7fbcba55'
ROOT = Path(__file__).resolve().parents[1]


def integration_digest():
    h=hashlib.sha256()
    for name in ('scripts/setup_gsprior.py','scripts/gsprior_runtime.py',
                 'utils/checkpoint_snapshots.py','utils/training_previews.py','utils/region_loss.py'):
        h.update((ROOT/name).read_bytes())
    return h.hexdigest()


def patch_sources(repo):
    def edit(name, old, new, count=1):
        path=repo/name; content=path.read_text()
        if content.count(old)!=count: raise ValueError(f'Unexpected upstream source: {name}: {old[:80]!r}')
        path.write_text(content.replace(old,new))
    edit('scene/dataset_readers.py', '       storePly(ply_path, xyz, rgb)', '        storePly(ply_path, xyz, rgb)')
    edit('scene/dataset_readers.py', 'if not os.path.exists(ply_path) or True:', 'if not os.path.exists(ply_path):')
    edit('scene/dataset_readers.py', 'focal_length_x = intr.params[0]\n            FovY',
         'focal_length_x = focal_length_y = intr.params[0]\n            FovY')
    edit('scene/dataset_readers.py', 'os.path.join(images_folder, os.path.basename(extr.name))', 'os.path.join(images_folder, extr.name)')
    edit('scene/dataset_readers.py', 'os.path.basename(image_path).split(".")[0]', 'extr.name')
    edit('scene/gaussian_model.py', 'from pytorch3d.transforms import quaternion_to_matrix',
         'def quaternion_to_matrix(quaternion):\n    return build_rotation(quaternion)')
    edit('scene/gaussian_model.py', 'self._xyz = self.get_xyz - 1e-7*(sdf * grad_norm)',
         'from experiment_adapter import apply_pull\n                apply_pull(self, sdf, grad_norm)')
    # Dense LiDAR may already exceed the upstream 6M growth cap. It must not
    # pass negative capacity into torch.quantile or delete initialization rows.
    for method, signature in [('split','grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent, max_radii2D, N=2'),
                               ('clone','grads, grad_threshold, scene_extent')]:
        old=f'    def densify_and_{method}(self, {signature}):\n'
        edit('scene/gaussian_model.py',old,old+'        if len(self._xyz) >= self.max_all_points: return\n')
    edit('train.py','from utils.tsdf_prior_gen import *','import experiment_adapter as experiment')
    edit('train.py','    tb_writer = prepare_output_and_logger(dataset)',
         '    dataset.antialiasing = False\n    dataset.model_type = "gsprior"\n    dataset.keyframe_manifest = os.path.join(dataset.model_path, "keyframes.json")\n    tb_writer = prepare_output_and_logger(dataset)')
    edit('train.py','    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")',
         '    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")\n    experiment.start(scene, opt, pipe, render, background)')
    edit('train.py', '''        if (iteration in saving_iterations):
            print("\\n[ITER {}] Saving Gaussians".format(iteration))
            scene.save(iteration)
''', '')
    path=repo/'train.py'; content=path.read_text()
    begin=content.index('            _,tsdf, vol_origin, voxel_size, mesh = run_tsdf_fusion(')
    end=content.index('        gt_image, gt_image_gray =',begin)
    content=content[:begin]+'            experiment.update_prior(scene, pipe, render, background, iteration, tsdf_trunc)\n'+content[end:]
    # Publish checkpoints only after the complete iteration, before evaluator reads.
    begin=content.index('            if (iteration in checkpoint_iterations):')
    end=content.index('    app_model.save_weights(scene.model_path, opt.iterations)',begin)
    content=content[:begin]+'            experiment.finish_iteration(scene, opt, pipe, render, background, app_model, iteration)\n\n'+content[end:]
    path.write_text(content)
    edit('train.py', 'parser.add_argument(\'--tsdf_iter\', type=int, default=[20000,30000,40000])',
         'parser.add_argument(\'--tsdf_iter\', type=int, nargs="+", default=[20000,30000,40000])')
    edit('train.py', 'parser.add_argument(\'--tsdf_trunc\', type=int, default=[24*4,12*4,6*4])',
         'parser.add_argument(\'--tsdf_trunc\', type=float, nargs="+", default=[24*4,12*4,6*4])')
    edit('train.py','    for iteration in range(first_iter, opt.iterations + 1):',
         '    for iteration in range(first_iter, opt.iterations + 1):\n        if len(gaussians.get_xyz)==0: raise RuntimeError("Empty Gaussian population")')
    edit('train.py','"Points": f"{len(gaussians.get_xyz)}"','"Gaussians": f"{len(gaussians.get_xyz):,}"')
    edit('arguments/__init__.py', 'self.preload_img = True', 'self.preload_img = False')
    edit('utils/fusion.py','FUSION_GPU_MODE = 1', 'FUSION_GPU_MODE = 1\ncuda.init()')
    edit('utils/fusion.py','if (voxel_idx > vol_dim_x*vol_dim_y*vol_dim_z)',
         'if (voxel_idx >= vol_dim_x*vol_dim_y*vol_dim_z)')
    # Five-channel plane gradients and depth gradients also receive expanded
    # gradients from reductions; materialize at the Python/CUDA boundary.
    raster=repo/'submodules/diff-plane-rasterization/diff_plane_rasterization/__init__.py'
    content=raster.read_text()
    content=content.replace('num_rendered, color, radii, out_observe, out_all_map, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)',
                            'num_rendered, color, radii, out_observe, out_all_map, out_plane_depth, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)')
    pos=content.index('        # Restore necessary values from context') if '        # Restore necessary values from context' in content else -1
    if pos<0: raise ValueError('Inspect upstream rasterizer backward before updating pin')
    content=content[:pos]+'''        if grad_out_color is not None: grad_out_color = grad_out_color.contiguous()
        if grad_out_all_map is not None: grad_out_all_map = grad_out_all_map.contiguous()
        if grad_out_plane_depth is not None: grad_out_plane_depth = grad_out_plane_depth.contiguous()
'''+content[pos:]
    raster.write_text(content)
    for name in ('checkpoint_snapshots.py','training_previews.py','region_loss.py'):
        shutil.copyfile(ROOT/'utils'/name,repo/'utils'/name)
    shutil.copyfile(ROOT/'scripts/gsprior_runtime.py',repo/'experiment_adapter.py')
    for name in ('train.py','scene/dataset_readers.py','scene/gaussian_model.py','experiment_adapter.py'):
        compile((repo/name).read_text(),str(repo/name),'exec')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True)
    args=p.parse_args();repo=args.repo.resolve()
    head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if head!=COMMIT: raise ValueError(f'Expected {COMMIT}, found {head}')
    marker=repo/'orange4f_adapter.json'
    if marker.exists():
        saved=json.loads(marker.read_text())
        if saved.get('integration_digest') != integration_digest():
            raise ValueError('Integration code changed; use a new GSPRIOR_REPO and GSPRIOR_ENV directory')
        for name,expected in saved['files'].items():
            if hashlib.sha256((repo/name).read_bytes()).hexdigest()!=expected:
                raise ValueError('Patched checkout changed; use a new GSPRIOR_REPO directory')
        return
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip():
        raise ValueError('Use a fresh dedicated checkout; do not modify a working GSPrior repo')
    patch_sources(repo)
    names=['train.py','scene/dataset_readers.py','scene/gaussian_model.py','arguments/__init__.py',
           'utils/fusion.py','experiment_adapter.py','utils/checkpoint_snapshots.py',
           'utils/training_previews.py','utils/region_loss.py',
           'submodules/diff-plane-rasterization/diff_plane_rasterization/__init__.py']
    marker.write_text(json.dumps(dict(upstream=head,integration_digest=integration_digest(),
        files={n:hashlib.sha256((repo/n).read_bytes()).hexdigest() for n in names}),indent=2))
    print(f'GSPrior integration ready: {repo}',flush=True)


if __name__=='__main__': main()
