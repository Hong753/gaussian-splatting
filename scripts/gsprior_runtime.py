"""Adapters copied into the pinned GSPrior checkout by setup_gsprior.py.

The released renderer/losses/population policy remain upstream. These adapters
provide dataset-independent TSDF fusion, fixed-XYZ controls and common reports.
"""
import csv
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from utils.checkpoint_snapshots import atomic_json, atomic_write


def config():
    return json.loads(Path(os.environ['GSPRIOR_RUN_CONFIG']).read_text())


@torch.no_grad()
def sample_sdf(model, points):
    """Sample XYZ-ordered TSDF correctly with bounded CUDA temporary storage."""
    volume = model._prior_volume
    origin = torch.as_tensor(model.vol_origin, device=points.device, dtype=points.dtype)
    size = torch.tensor(volume.shape, device=points.device, dtype=points.dtype)-1
    output = []
    for block in points.split(131072):
        coordinates = (block-origin)/model.voxel_size
        valid = ((coordinates >= 0) & (coordinates <= size)).all(-1)
        grid = (2*coordinates/size-1)[:, [2,1,0]].reshape(1,1,-1,1,3)
        values = F.grid_sample(volume[None,None], grid, mode='bilinear',
                               padding_mode='border', align_corners=True).reshape(-1)
        output.append(torch.where(valid, values, torch.ones_like(values)))
    return torch.cat(output).reshape(-1,1) if output else points.new_empty((0,1))


def set_tsdf(model, tsdf, origin, voxel, truncation, interp=None):
    model.tsdf = tsdf
    model._prior_volume = torch.as_tensor(tsdf, device=model.get_xyz.device, dtype=torch.float32)
    model.vol_origin, model.voxel_size, model.tsdf_trunc = origin, voxel, truncation


def opacity(model, iteration=None, tsdf_mask=False, exp_name=None):
    value = model.opacity_activation(model._opacity)
    if iteration is not None and not model.gs_state and tsdf_mask:
        return sample_sdf(model, model.get_xyz), value
    return value


@torch.no_grad()
def apply_pull(model, sdf, normal):
    if getattr(model, '_fixed_xyz', False): return
    # Preserve the upstream pull coefficient, but update the Parameter in-place
    # so Adam continues to reference the position tensor actually rendered.
    model._xyz.sub_(1e-7*sdf*normal)


def install_model_controls(model, fixed):
    from types import MethodType
    model._fixed_xyz = fixed
    model.set_tsdf = MethodType(set_tsdf, model)
    model.get_sdf = MethodType(sample_sdf, model)
    model.get_opacity = MethodType(opacity, model)
    previous_prune = model.prune_points

    def prune(mask):
        if len(mask) and bool(mask.all()):
            raise RuntimeError('GSPrior pruning would delete the entire model; no rows were deleted')
        previous_prune(mask)
        if fixed: model._xyz.requires_grad_(False)
    model.prune_points = prune
    if fixed:
        model._xyz.requires_grad_(False)
        def update_lr(iteration):
            for group in model.optimizer.param_groups:
                if group['name']=='xyz': group['lr']=0.
            return 0.
        model.update_learning_rate = update_lr
        # Both upstream pre-prior and post-prior growth paths call these methods.
        def no_growth(*args, **kwargs):
            model.max_radii2D.zero_()
        model.densify_and_clone = no_growth
        model.densify_and_split = no_growth
        previous_population = model.densify_and_prune
        def population(*args, **kwargs):
            result = previous_population(*args, **kwargs)
            model.xyz_gradient_accum.zero_(); model.xyz_gradient_accum_abs.zero_()
            model.denom.zero_(); model.denom_abs.zero_(); model.max_radii2D.zero_()
            return result
        model.densify_and_prune = population
        update_lr(0)


def start(scene, opt, pipe, render, background):
    from utils.training_previews import TrainingPreviewer, select_names, DEFAULTS
    cfg = config()
    install_model_controls(scene.gaussians, cfg['fixed_positions'])
    scene.gaussians.max_all_points = cfg['max_all_points']
    print(f'GSPrior growth budget: {cfg["max_all_points"]:,}; initial: {len(scene.gaussians.get_xyz):,}', flush=True)
    pipe.antialiasing = False
    all_cameras = {c.image_name:c for c in scene.getTrainCameras()+scene.getTestCameras()}
    chosen = select_names([c.image_name for c in scene.getTrainCameras()],
                          [c.image_name for c in scene.getTestCameras()], DEFAULTS)
    frames = []
    for split, names in chosen.items():
        for name in names:
            cam = all_cameras[name]
            # Only these four preview RGB tensors are retained, on CPU.
            cam.original_image = cam.get_image()[0].cpu()
            frames.append((split,cam))
    scene._previewer = TrainingPreviewer(Path(scene.model_path)/'debug_previews', frames, DEFAULTS,
        dict(source=cfg['original_source'], eligible_train=len(scene.getTrainCameras()),
             eligible_heldout=len(scene.getTestCameras()), heldout_protocol='Shared sharp-stride3 complement'))
    scene._preview_render = lambda camera, model, pipeline, bg, **kw: render(
        camera, model, pipeline, bg, return_plane=False, return_depth_normal=False)
    atomic_json(Path(scene.model_path)/'run_protocol.json', dict(**cfg,
        fixed_xyz_controls='XYZ gradients, Adam LR, cloning, splitting and direct TSDF pull disabled' if cfg['fixed_positions'] else 'Upstream learnable XYZ, growth and TSDF pull',
        renderer='official GSPrior diff_plane_rasterization; no AA compensation',
        code_adaptations=['data paths and split', 'scene TSDF bounds and physical truncation',
            'bounded correct XYZ TSDF interpolation', 'in-place position pull',
            'growth budget max(6M, 1.5 x initialization)', 'safe population capacity handling',
            'previews/checkpoints/evaluation']))


@torch.no_grad()
def update_prior(scene, pipe, render, background, iteration, truncation):
    """Fuse only selected training views of the current model, without cloning it."""
    from utils.fusion import TSDFVolume
    from tqdm import tqdm
    cfg = config(); grid = cfg['grid']; model = scene.gaussians
    volume = TSDFVolume(np.array(grid['bounds'], dtype=np.float32),
        voxel_size=grid['voxel_size'], sdf_trunc=truncation)
    try:
        for camera in tqdm(scene.getTrainCameras(), desc=f'TSDF {iteration} (train views only)'):
            out = render(camera, model, pipe, background, return_plane=True, return_depth_normal=True)
            depth = out['plane_depth'].squeeze().clone()
            valid = torch.isfinite(depth) & (depth>0)
            if 'depth_normal' in out:
                normal = F.normalize(out['depth_normal'].permute(1,2,0), dim=-1)
                ray = F.normalize(camera.get_rays(), dim=-1)
                valid &= (ray*normal).sum(-1).abs() >= math.cos(math.radians(80))
            depth[~valid] = 0
            pose = camera.world_view_transform.T.inverse().cpu().numpy()
            volume.integrate(out['render'].permute(1,2,0).clamp(0,1).cpu().numpy(),
                depth.cpu().numpy(), camera.get_k().cpu().numpy(), pose, obs_weight=1.)
        tsdf, _ = volume.get_volume()
        origin = volume._vol_origin.copy()
        observed = int(np.count_nonzero(tsdf < 1.))
        if observed==0: raise RuntimeError('TSDF fusion produced no surface evidence')
    finally:
        del volume
    folder = Path(scene.model_path)/'prior_cache'/f'iteration_{iteration}'
    folder.mkdir(parents=True, exist_ok=False)
    atomic_write(folder/'tsdf.npy', lambda p: save_npy(p, tsdf))
    atomic_json(folder/'metadata.json',dict(iteration=iteration, origin=origin.tolist(),
        voxel_size=grid['voxel_size'], dimensions=list(tsdf.shape), truncation=truncation,
        train_views=len(scene.getTrainCameras()), observed_voxels=observed))
    model.set_tsdf(tsdf, origin, grid['voxel_size'], truncation)


def save_npy(path, array):
    with Path(path).open('wb') as stream: np.save(stream, array)


@torch.no_grad()
def finish_iteration(scene, opt, pipe, render, background, app_model, iteration):
    cfg = config()
    if len(scene.gaussians.get_xyz)==0: raise RuntimeError('GSPrior model became empty')
    if scene._previewer.due(iteration, opt.iterations):
        scene._previewer.capture(iteration, scene.gaussians, scene._preview_render, pipe, background)
    if iteration in cfg['checkpoints']:
        scene.save(iteration)
        atomic_write(Path(scene.model_path)/f'chkpnt{iteration}.pth',
            lambda p:torch.save((scene.gaussians.capture(), iteration),p))
        app_model.save_weights(scene.model_path, iteration)
        atomic_json(Path(scene.model_path)/'point_cloud'/f'iteration_{iteration}'/'ready.json',
            dict(iteration=iteration, gaussians=len(scene.gaussians.get_xyz), antialiasing=False,
                 state_timing='after training iteration', source=cfg['original_source']))


def evaluate_all(scene, pipe, render, background, iteration):
    """Stream all frames through the official plane rasterizer; keep image reports."""
    from tqdm import tqdm
    cfg = config(); root = Path(scene.model_path)/'psnr_all_frames'/f'iteration_{iteration}'
    root.mkdir(parents=True, exist_ok=True)
    train = {c.image_name for c in scene.getTrainCameras()}
    cameras = sorted(scene.getTrainCameras()+scene.getTestCameras(), key=lambda c:c.image_name)
    if len(cameras)!=cfg['all_frames']: raise ValueError('Incomplete all-frame evaluation')
    totals = dict(all=0., selected=0., unselected=0.); counts = dict.fromkeys(totals,0)
    # Comparison reports retain 32 evenly spaced train and 32 held-out images.
    from utils.training_previews import TrainingPreviewer, DEFAULTS
    frames=[]
    for split, pool in [('train',[c for c in cameras if c.image_name in train]),
                        ('held-out',[c for c in cameras if c.image_name not in train])]:
        if pool:
            frames += [(split,pool[i]) for i in np.unique(np.linspace(0,len(pool)-1,min(32,len(pool))).astype(int))]
    with torch.no_grad(), (root/'per_frame.csv').open('w') as stream:
        writer=csv.writer(stream); writer.writerow(['image_name','selected_for_training','psnr_db'])
        for camera in tqdm(cameras, desc='GSPrior PSNR all frames'):
            pred=render(camera,scene.gaussians,pipe,background,return_plane=False,return_depth_normal=False)['render'].clamp(0,1)
            gt=camera.get_image()[0].clamp(0,1)
            score=(-10*torch.log10((pred-gt).square().flatten(1).mean(1))).mean().item()
            group='selected' if camera.image_name in train else 'unselected'
            writer.writerow([camera.image_name,int(group=='selected'),score])
            for key in ('all',group): totals[key]+=score; counts[key]+=1
    means={k:totals[k]/counts[k] if counts[k] else None for k in totals}
    atomic_json(root/'summary.json',dict(iteration=iteration, model=scene.model_path,
        source=cfg['original_source'], model_type='gsprior', antialiasing=False, counts=counts,
        mean_psnr_db=means, metric='mean per-channel PSNR, then mean over images'))
    # Load only sampled comparison targets; release them after saving the report.
    for _,cam in frames: cam.original_image=cam.get_image()[0].cpu()
    preview=TrainingPreviewer(Path(scene.model_path)/'view_comparisons'/f'iteration_{iteration}',frames,DEFAULTS,
        dict(source=cfg['original_source'],heldout_protocol='Shared sharp-stride3 complement'))
    preview.capture(iteration,scene.gaussians,scene._preview_render,pipe,background)
    preview_ids={id(cam) for _,cam in scene._previewer.frames}
    for _,cam in frames:
        if id(cam) not in preview_ids: cam.original_image=None
    atomic_json(root/'complete.json', dict(iteration=iteration, frames=len(cameras), comparisons=len(frames)))
    return means


def smoke():
    """Run the real upstream CUDA RGB/backward and fixed-center mutation paths."""
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from arguments import OptimizationParams
    from argparse import ArgumentParser
    from utils.graphics_utils import BasicPointCloud
    from scene.cameras import Camera
    p=ArgumentParser(); group=OptimizationParams(p); opt=group.extract(p.parse_args([]))
    camera=Camera(0,np.eye(3),np.zeros(3),math.pi/2,math.pi/2,32,32,'','smoke',0,
                  preload_img=False,data_device='cpu')
    pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,debug=True,antialiasing=False)
    for fixed in (False, True):
        model=GaussianModel(3)
        xyz=np.array([[-.1,-.1,2],[.1,-.1,2],[-.1,.1,2],[.1,.1,2]],np.float32)
        model.create_from_pcd(BasicPointCloud(xyz,np.full_like(xyz,.5),np.zeros_like(xyz)),1.)
        model.training_setup(opt); install_model_controls(model,fixed)
        with torch.no_grad(): model._scaling[:,2] -= 2.
        before=model.get_xyz.detach().clone()
        image=render(camera,model,pipe,torch.zeros(3,device='cuda'),return_plane=False)['render']
        (image-.7).square().mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for g in model.optimizer.param_groups for p in g['params'])
        model.optimizer.step(); model.optimizer.zero_grad(set_to_none=True)
        if fixed: assert torch.equal(before,model.get_xyz) and not model.get_xyz.requires_grad
        else: assert not torch.equal(before,model.get_xyz)
        # Exercise the plane/depth backward used once geometric losses turn on.
        plane=render(camera,model,pipe,torch.zeros(3,device='cuda'),return_plane=True,return_depth_normal=True)
        assert torch.isfinite(plane['plane_depth']).all() and torch.isfinite(plane['depth_normal']).all()
        (plane['render'].mean()+.001*plane['plane_depth'].mean()+.001*plane['rendered_normal'].square().mean()).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for g in model.optimizer.param_groups for p in g['params'])
        model.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.denom[:]=1;model.denom_abs[:]=1;model.xyz_gradient_accum[:]=1
            model.xyz_gradient_accum_abs[:]=1;model._opacity[:]=0
            model.densify_and_prune(.0002,.0008,.005,100.,None,iteration=600)
            if fixed: assert torch.equal(before,model.get_xyz)
            parameter=model._xyz
            apply_pull(model,torch.ones_like(model._opacity),torch.ones_like(model.get_xyz))
            assert model._xyz is parameter
            if fixed:
                assert torch.equal(before,model.get_xyz)
                model.prune_points(torch.tensor([True,False,False,False],device='cuda'))
                assert torch.equal(before[1:],model.get_xyz) and not model.get_xyz.requires_grad
    # Build and run the actual fusion CUDA kernel before committing to 40k steps.
    from utils.fusion import TSDFVolume
    fusion=TSDFVolume(np.array([[-.2,.2],[-.2,.2],[1.8,2.2]],np.float32),.05,.2)
    try:
        fusion.integrate(np.zeros((16,16,3),np.float32),np.full((16,16),2.,np.float32),
                         np.array([[16,0,8],[0,16,8],[0,0,1]],np.float32),np.eye(4,dtype=np.float32))
        volume,_=fusion.get_volume()
        surface=SimpleNamespace(_prior_volume=torch.as_tensor(volume,device='cuda'),
                                vol_origin=fusion._vol_origin,voxel_size=.05)
        sampled=sample_sdf(surface,torch.tensor([[0,0,1.9],[0,0,2.],[0,0,2.1]],device='cuda')).flatten()
        torch.testing.assert_close(sampled,torch.tensor([.5,0.,-.5],device='cuda'),atol=1e-4,rtol=1e-4)
    finally:
        del fusion
    print(json.dumps(dict(passed=True,official_gsprior_cuda_rgb_backward=True,
        plane_depth_backward=True,tsdf_cuda_fusion_and_sampling=True,
        fixed_xyz_adam_growth_pull_and_prune=True,antialiasing=False)),flush=True)
