"""Saved-state diagnostic using the real native render and backward paths."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.view_comparisons import model_info, image_id
from scripts.eval_all_frames import choose_iteration
from utils.occlusion_diagnostics import (classify, depth_points, depth_rgb,
                                         projected_overlay, write_report)

PARAMETERS = ('_xyz', '_opacity', '_scaling', '_rotation', '_features_dc', '_features_rest')


def gradient_statistics(parameters, gradients, culled, chunk=131072):
    import torch
    union = torch.zeros(len(culled), dtype=torch.bool, device=culled.device)
    summary = {}
    for name, grad in zip(parameters, gradients):
        nonzero, hidden_nonzero, maximum, finite = 0, 0, 0., True
        if grad is not None and grad.numel():
            for start in range(0, len(culled), chunk):
                block = grad[start:start+chunk].reshape(len(grad[start:start+chunk]), -1)
                hidden = culled[start:start+chunk]
                magnitude = block.abs().amax(dim=1)
                nz = magnitude > 0
                union[start:start+chunk] |= nz
                nonzero += int(nz.sum().item())
                hidden_nonzero += int((nz & hidden).sum().item())
                if hidden.any():
                    maximum = max(maximum, float(magnitude[hidden].max().item()))
                finite &= bool(torch.isfinite(block).all().item())
        summary[name.removeprefix('_')] = dict(nonzero_rows=nonzero, culled_nonzero_rows=hidden_nonzero,
            culled_max_abs_gradient=maximum if np.isfinite(maximum) else None,
            finite=finite, connected=grad is not None)
    return dict(parameters=summary, nonzero_gaussians=int(union.sum().item()),
        culled_nonzero_gaussians=int((union & culled).sum().item()),
        all_finite=all(row['finite'] for row in summary.values()))


def probe_verdict(on, off, count):
    if not on['all_finite'] or not off['all_finite']:
        return 'FAILED: non-finite image gradients'
    if on['culled_nonzero_gaussians']:
        return 'FAILED: culled Gaussians received image gradients'
    if count == 0:
        return 'INCONCLUSIVE: no Gaussians were culled for this view'
    if not on['nonzero_gaussians'] and not off['nonzero_gaussians']:
        return 'INCONCLUSIVE: both image losses produced zero parameter gradients'
    return ('PASS: culled image gradients are exactly zero; '
            f"{off['culled_nonzero_gaussians']:,} of those Gaussians receive gradients with culling OFF")


class InspectorBackend:
    def __init__(self, args):
        import torch
        from scene.gaussian_model import GaussianModel
        from gaussian_renderer import render
        from scripts.occlusion_options import configure_pipeline
        from utils.lidar_occlusion import file_digest
        if not torch.cuda.is_available():
            raise RuntimeError('Activate the CUDA-enabled 3dgs environment on your GPU host')
        self.torch, self.render, self.args = torch, render, args
        self.spec = model_info(args.model)
        self.cfg = self.spec['cfg']
        if self.spec['type'] != '3dgs':
            raise ValueError('This diagnostic supports native 3DGS checkpoints')
        self.iteration, self.ply = choose_iteration(self.spec['path'], args.iteration)
        self.depth_dir = self.cfg.get('train_depth_occlusion_dir', '')
        self.is_da3 = bool(self.depth_dir)
        if not self.is_da3 and self.cfg.get('occlusion_mode', 'off') != 'lidar':
            raise ValueError('Choose a trained LiDAR or DA3 occlusion run with saved culling settings')
        self.names = sorted(self.spec['splits'][args.split])
        if self.is_da3:
            self.names = [n for n in self.names if n in self.spec['splits']['train']]
        if not self.names or (args.image and args.image not in self.names):
            raise ValueError('Choose an available image in this split (DA3 supports its training images only)')
        print(f'Loading iteration {self.iteration}: {self.ply}', flush=True)
        before = self.ply.stat()
        self.model = GaussianModel(self.cfg.get('sh_degree', 3))
        self.model.load_ply(str(self.ply))
        self.parameters = [getattr(self.model, key) for key in PARAMETERS]
        for parameter in self.parameters:
            parameter.requires_grad_(False)
        self.xyz = self.model.get_xyz.detach().cpu().numpy()
        if not np.isfinite(self.xyz).all():
            raise ValueError('Checkpoint contains non-finite Gaussian centers')
        self.checkpoint_sha256 = file_digest(self.ply)
        after = self.ply.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('Checkpoint changed while loading; retry after its save completes')
        aa = self.cfg.get('antialiasing', True)
        common = dict(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=aa)
        self.pipe, self.off_pipe = SimpleNamespace(**common), SimpleNamespace(**common)
        configure_pipeline(self.pipe, self.cfg)
        configure_pipeline(self.off_pipe, {**self.cfg, 'occlusion_mode': 'off'})
        if self.is_da3:
            from utils.da3_occlusion import TrainingDepthCache
            self.provider = TrainingDepthCache(self.depth_dir, self.cfg['source_path'],
                [SimpleNamespace(image_name=n) for n in self.spec['splits']['train']],
                self.cfg.get('resolution', -1), self.pipe.occlusion_edge_pixels,
                point_cloud=self.spec['path']/'input.ply')
        else:
            from utils.occlusion_depth import LidarDepthCache
            self.provider = LidarDepthCache(self.pipe.occlusion_geometry, self.pipe.occlusion_cache,
                                           self.pipe.occlusion_edge_pixels)
            self.pipe._occlusion_provider = self.provider
        self.background = torch.tensor([1., 1., 1.] if self.cfg.get('white_background') else [0., 0., 0.], device='cuda')
        self.root = args.output or self.spec['path']/'occlusion_diagnostics'/f'iteration_{self.iteration}'
        self.current = None
        print(f'{len(self.xyz):,} Gaussians; {len(self.names):,} selectable images; AA={aa}', flush=True)

    def camera(self, name):
        from utils.camera_utils import loadCam
        r = self.spec['records'][name]
        info = SimpleNamespace(uid=0, image_path=r['image_path'], image_name=name,
            R=np.asarray(r['R']), T=np.asarray(r['T']), FovX=r['fov_x'], FovY=r['fov_y'],
            depth_path='', depth_params=None, is_test=False)
        return loadCam(SimpleNamespace(resolution=self.cfg.get('resolution', -1), data_device='cpu',
            train_test_exp=False), 0, info, 1., False, False)

    def render_state(self, camera, tiles, enabled, check_gradients=False, hidden=None):
        torch = self.torch
        kwargs = dict(occlusion_tiles=tiles) if enabled and self.is_da3 else {}
        try:
            for parameter in self.parameters:
                parameter.requires_grad_(check_gradients)
            with torch.enable_grad() if check_gradients else torch.no_grad():
                result = self.render(camera, self.model, self.pipe if enabled else self.off_pipe,
                                     self.background, use_trained_exp=False, **kwargs)
                keep = result.get('occlusion_keep')
                if enabled and keep is None:
                    raise RuntimeError('The real renderer did not return an occlusion mask')
                pred = result['render']
                if not torch.isfinite(pred).all():
                    raise RuntimeError('Native render contains non-finite pixels')
                stats = None
                if check_gradients:
                    from utils.loss_utils import l1_loss, ssim
                    target = camera.original_image[:3].to(pred.device)
                    image = pred if camera.alpha_mask is None else pred*camera.alpha_mask.to(pred.device)
                    weight = self.cfg.get('optimization_settings', {}).get('lambda_dssim', .2)
                    loss = (1-weight)*l1_loss(image, target) + weight*(1-ssim(image, target))
                    if not torch.isfinite(loss):
                        raise RuntimeError('Non-finite diagnostic photometric loss')
                    grads = torch.autograd.grad(loss, self.parameters, allow_unused=True)
                    hidden = ~keep if enabled else hidden
                    stats = gradient_statistics(PARAMETERS, grads, hidden)
                    stats['loss'] = float(loss.item())
                    del grads, loss
                return (pred.detach().cpu(), result['radii'].detach().cpu().numpy(),
                        keep.detach().cpu().numpy() if keep is not None else None, stats)
        finally:
            for parameter in self.parameters:
                parameter.requires_grad_(False)

    def analyze(self, name, check_gradients=False):
        from utils.view_tangent import camera_intrinsics
        from utils.image_utils import psnr
        torch = self.torch
        if name not in self.names:
            raise ValueError('Image is not in the available diagnostic split')
        camera = self.camera(name)
        tiles = self.provider.tiles(camera, self.model.get_xyz.device)
        print(f'Inspecting {name}; native resolution {camera.image_width}x{camera.image_height}', flush=True)
        on, radii_on, keep, on_grad = self.render_state(camera, tiles, True, check_gradients)
        hidden = torch.as_tensor(~keep, device=self.model.get_xyz.device)
        off, radii_off, _, off_grad = self.render_state(camera, tiles, False, check_gradients, hidden)
        labels, counts = classify(keep, radii_on, radii_off)
        w2c = camera.world_view_transform.detach().cpu().numpy().T
        c2w = np.linalg.inv(w2c)
        k = camera_intrinsics(camera, 'cpu').numpy()
        surface = self.provider.surface(camera, self.model.get_xyz.device)
        cloud = depth_points(surface['depth'].cpu().numpy(), surface['intrinsic'].cpu().numpy(), c2w)
        rgb = lambda t: (t.detach().cpu().permute(1, 2, 0).numpy()*255).round().clip(0, 255).astype(np.uint8)
        gt = camera.original_image[:3].cpu()
        gt_rgb = rgb(gt)
        colored_depth, depth_range = depth_rgb(tiles.cpu().numpy(), camera.image_width, camera.image_height)
        images = dict(gt=('Actual image', gt_rgb), culling_off=('Culling OFF', rgb(off)),
            culling_on=('Culling ON', rgb(on)), difference=('Absolute ON/OFF difference x10', rgb((on-off).abs().mul(10))),
            centers=('Gaussian centers over GT (no z-buffer)', projected_overlay(self.xyz, labels, w2c, k, gt_rgb)),
            depth=('Conservative occlusion depth bounds; black = unknown', colored_depth))
        summary = dict(image=name, iteration=self.iteration, checkpoint=str(self.ply),
            checkpoint_sha256=self.checkpoint_sha256, culling_source='DA3' if self.is_da3 else 'LiDAR surfels',
            antialiasing=self.pipe.antialiasing, counts=counts,
            culled_fraction_all=counts['culled']/max(counts['total'], 1),
            culled_fraction_raster_candidates=counts['culled_with_raster_footprint']/max(counts['raster_candidates_without_culling'], 1),
            reliable_tile_fraction=float(self.torch.isfinite(tiles).float().mean().item()),
            depth_color_range={key: float(v) if v is not None else None for key, v in depth_range.items()},
            mean_absolute_render_difference=float((on-off).abs().mean().item()),
            psnr_db={key: (lambda v: v if np.isfinite(v) else 'inf')(psnr(image, gt).mean().item())
                     for key, image in (('on', on), ('off', off))},
            definitions=dict(active='kept and rasterizer radius > 0; pixel contribution is not guaranteed',
                culled='exact inverse of native renderer occlusion_keep, not ground-truth visibility',
                other='kept but no raster footprint', labels_npy='one uint8 label per original checkpoint PLY row'),
            culling_margin=self.pipe.occlusion_margin, culling_sigma=self.pipe.occlusion_sigma,
            appearance='base appearance without per-image exposure; saved fixed white/black background',
            gradient_probe=None, unchanged_checkpoint=True)
        if check_gradients:
            summary['gradient_probe'] = dict(on=on_grad, off=off_grad,
                verdict=probe_verdict(on_grad, off_grad, counts['culled']),
                scope='L1 + SSIM image gradients on this saved view/state; no optimizer step or depth/surface loss')
        directory = Path(self.root)/image_id(name)
        write_report(directory, summary, images)
        np.save(directory/'labels.npy', labels, allow_pickle=False)
        np.save(directory/'occlusion_tiles.npy', tiles.cpu().numpy(), allow_pickle=False)
        self.current = dict(name=name, camera=camera, c2w=c2w, tiles=tiles, labels=labels,
            radii_on=radii_on, radii_off=radii_off, summary=summary, directory=directory,
            images=images, surface=cloud)
        print(f"Active {counts['active']:,}; depth-culled {counts['culled']:,}; other {counts['other']:,}", flush=True)
        if check_gradients:
            print(summary['gradient_probe']['verdict'], flush=True)
        print(f'Diagnostic report: {directory}/index.html', flush=True)
        del hidden
        torch.cuda.empty_cache()
        return self.current

    def gaussian_details(self, index):
        from utils.occlusion_culling import conservative_keep, rotation_matrix
        if self.current is None or not 0 <= index < len(self.xyz):
            raise ValueError('Select a valid checkpoint Gaussian ID after inspecting a view')
        sl = slice(index, index+1)
        with self.torch.no_grad():
            scales, rotations = self.model._scaling[sl].exp(), self.model._rotation[sl]
            keep, details = conservative_keep(self.current['camera'], self.model.get_xyz[sl], scales,
                rotations, self.current['tiles'], self.pipe.occlusion_margin, self.pipe.occlusion_sigma,
                self.pipe.occlusion_chunk_size, return_details=True)
            if bool(keep[0]) != (self.current['labels'][index] != 2):
                raise RuntimeError('Single-Gaussian decision disagrees with full renderer mask')
            matrix = (rotation_matrix(rotations)[0]*scales[0][None, :]).cpu().numpy()
        values = {key: value[0].cpu().numpy().tolist() for key, value in details.items()}
        if not keep[0]:
            reason = 'Culled: nearest tested Gaussian support is behind a finite surface bound plus margin'
        elif not values['reliable']:
            reason = 'Kept: support is outside the valid conservative test (e.g. near camera or outside image)'
        elif not np.isfinite(values['surface_bound_z']):
            reason = 'Kept: the tested footprint does not have complete reliable depth coverage'
        else:
            reason = 'Kept: nearest support overlaps the surface depth or safety margin'
        for key, value in values.items():
            if isinstance(value, float) and not np.isfinite(value): values[key] = None
        values.update(gaussian_id=index, reason=reason, radius_on=int(self.current['radii_on'][index]),
                      radius_off=int(self.current['radii_off'][index]), margin=self.pipe.occlusion_margin,
                      sigma=self.pipe.occlusion_sigma)
        return values, matrix
