#!/usr/bin/env python3
"""Train Scaffold-GS using this repository's dataset and camera pipeline.

The anchor model and update rules are adapted from city-super/Scaffold-GS.
"""
import os
import sys
import json
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
from random import randint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from tqdm import tqdm
from arguments import ModelParams, PipelineParams
from models.scaffold_arguments import ScaffoldOptimizationParams
from models.scaffold_gaussian import ScaffoldGaussianModel
from models.scaffold_renderer import prefilter_voxel, render
from scene import Scene
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from utils import training_previews
from utils.checkpoint_snapshots import publish_snapshot


def make_model(dataset):
    # Appearance embeddings are disabled: they are per-training-camera and
    # are not defined for novel camera views or the all-frame evaluator.
    return ScaffoldGaussianModel(feat_dim=dataset.scaffold_feat_dim,
                                 n_offsets=dataset.scaffold_n_offsets,
                                 voxel_size=dataset.scaffold_voxel_size,
                                 update_init_factor=16, appearance_dim=0)


def evaluate_training_views(scene, gaussians, pipe, background, iteration, use_exposure):
    gaussians.eval()
    try:
        with torch.no_grad():
            cams = scene.getTrainCameras()
            values = []
            for idx in range(5, 30, 5):
                cam = cams[idx % len(cams)]
                visible = prefilter_voxel(cam, gaussians, pipe, background)
                image = render(cam, gaussians, pipe, background, visible_mask=visible)["render"]
                if use_exposure:
                    exposure = gaussians.get_exposure_from_name(cam.image_name)
                    image = (image.permute(1, 2, 0) @ exposure[:, :3]).permute(2, 0, 1) + exposure[:, 3, None, None]
                values.append(psnr(image.clamp(0, 1), cam.original_image.cuda().clamp(0, 1)).mean().item())
            print(f"\n[ITER {iteration}] Evaluating train: PSNR {sum(values) / len(values):.4f}")
    finally:
        gaussians.train()


def preview_render(camera, gaussians, pipe, background, **unused):
    result = render(camera, gaussians, pipe, background,
                    visible_mask=prefilter_voxel(camera, gaussians, pipe, background))
    return result


def train(dataset, opt, pipe, save_iterations, test_iterations, preview_config=None,
          publish=False, start_checkpoint=None):
    sensed_path = getattr(opt, 'sensed_membership', '')
    static = getattr(opt, 'disable_anchor_pruning', False)
    if static and not opt.disable_anchor_growth:
        raise ValueError('Static parent population requires disabled growth')
    if sensed_path and not (opt.freeze_anchors and opt.disable_anchor_growth and static and not dataset.train_test_exp):
        raise ValueError('Measured updates require fixed parent positions/population and shared image exposure disabled')
    dataset.antialiasing = pipe.antialiasing
    os.makedirs(dataset.model_path, exist_ok=True)
    with open(os.path.join(dataset.model_path, "cfg_args"), "w") as stream:
        stream.write(str(Namespace(**vars(dataset))))
    gaussians = make_model(dataset)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    views, start = [], 0
    cameras = {c.image_name: c for c in scene.getTrainCameras()}
    if start_checkpoint:
        state, start = torch.load(start_checkpoint, map_location='cpu', weights_only=False)
        gaussians.restore(state, opt)
        if not 0 <= start < opt.iterations:
            raise ValueError('Resume step must precede the requested final iteration')
        if any((Path(dataset.model_path)/f'point_cloud/iteration_{s}/ready.json').exists()
               for s in save_iterations if s > start):
            raise FileExistsError('Later published checkpoints exist; resume into a fresh model directory')
    membership = None
    if static:
        from utils.scaffold_observations import Membership, sensed_step
    if sensed_path:
        membership = Membership(sensed_path, gaussians.get_anchor, cameras)
    policy = (membership.identity if membership else None, static)
    if start_checkpoint and (gaussians.training_state or {}).get('observation_policy', (None, False)) != policy:
        raise ValueError('Checkpoint uses different parent observations or pruning policy')
    adjust = not (static and opt.disable_anchor_growth)
    previewer = training_previews.create_previewer(scene, dataset,
        preview_config or training_previews.DEFAULTS, start_iteration=start)
    if start_checkpoint and gaussians.training_state:
        state = gaussians.training_state
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        views = [cameras[name] for name in state['pending_views']]
    gaussians.train()
    bar = tqdm(range(start + 1, opt.iterations + 1), desc="Training Scaffold-GS", unit="iter", initial=start, total=opt.iterations)
    for iteration in bar:
        gaussians.update_learning_rate(iteration)
        if dataset.train_test_exp:
            gaussians.exposure_optimizer.param_groups[0]["lr"] = gaussians.exposure_scheduler_args(iteration)
        if not views:
            views = scene.getTrainCameras().copy()
        cam = views.pop(randint(0, len(views) - 1))
        visible = prefilter_voxel(cam, gaussians, pipe, bg)
        gating = {}
        if static:
            selected = membership.mask(cam.image_name) if membership else torch.ones_like(visible)
            gating = dict(update_anchors=selected)
        result = render(cam, gaussians, pipe, bg, visible_mask=visible,
                        retain_grad=adjust and iteration < opt.update_until, **gating)
        image = result["render"]
        if dataset.train_test_exp:
            exposure = gaussians.get_exposure_from_name(cam.image_name)
            image = (image.permute(1, 2, 0) @ exposure[:, :3]).permute(2, 0, 1) + exposure[:, 3, None, None]
            image = image.clamp(0, 1)
        target = cam.original_image.cuda()
        ll1 = l1_loss(image, target)
        scale = result["scaling"]
        if static:
            scale = scale[result['active_children']]
        if len(gaussians.get_anchor) == 0:
            raise RuntimeError('The entire Scaffold model is empty')
        scale_loss = scale.prod(dim=1).mean() if scale.numel() else image.sum()*0
        loss = ((1.0 - opt.lambda_dssim) * ll1 + opt.lambda_dssim * (1.0 - ssim(image, target))
                + 0.01 * scale_loss)
        loss.backward()
        with torch.no_grad():
            if adjust and opt.start_stat < iteration < opt.update_until:
                gaussians.training_statis(result["viewspace_points"], result["neural_opacity"],
                                          result["visibility_filter"], result["selection_mask"], visible)
            if static:
                parents = result['active_parents']
                if len(parents):
                    active = torch.zeros_like(visible)
                    active[parents] = True
                    sensed_step(gaussians.optimizer, active)
            else:
                gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            if dataset.train_test_exp:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
            if adjust and opt.update_from < iteration < opt.update_until and iteration % opt.update_interval == 0:
                gaussians.adjust_anchor(check_interval=opt.update_interval,
                                         success_threshold=opt.success_threshold,
                                         grad_threshold=opt.densify_grad_threshold,
                                         min_opacity=opt.min_opacity)
            if iteration % 10 == 0:
                bar.set_postfix(loss=f"{loss.item():.6f}", anchors=len(gaussians.get_anchor),
                                active=int(result['visibility_filter'].sum()))
            if iteration == 1 or iteration % 100 == 0 or iteration == opt.iterations:
                with (Path(dataset.model_path)/'scaffold_training.jsonl').open('a') as stream:
                    coverage = {} if not membership else dict(image=cam.image_name, observation_source=membership.kind,
                        frustum_parents=int(visible.sum()), eligible_parents_in_frustum=int((selected & visible).sum()),
                        updated_parents=len(parents), eligible_rendered_children=int(result['active_children'].sum()))
                    stream.write(json.dumps(dict(**coverage, iteration=iteration, anchors=len(gaussians.get_anchor),
                        active_gaussians=int(result['visibility_filter'].sum()), loss=loss.item(),
                        freeze_anchors=gaussians.freeze_anchors, anchor_growth=not gaussians.disable_anchor_growth))+'\n')
            if iteration in test_iterations:
                evaluate_training_views(scene, gaussians, pipe, bg, iteration, dataset.train_test_exp)
            if previewer and previewer.due(iteration, opt.iterations):
                gaussians.eval()
                try:
                    previewer.capture(iteration, gaussians, preview_render, pipe, bg)
                finally:
                    gaussians.train()
            if iteration in save_iterations:
                print(f"\n[ITER {iteration}] Saving anchors, networks and optimizer state")
                gaussians.training_state = dict(observation_policy=policy, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                    pending_views=[c.image_name for c in views])
                # TorchScript tracing uses random example inputs. Saving must not affect sampling.
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    if publish:
                        publish_snapshot(scene, iteration)
                    else:
                        scene.save(iteration)
                        torch.save((gaussians.capture(), iteration), Path(dataset.model_path)/f'chkpnt{iteration}.pth')
    print("Scaffold-GS training complete")


def main():
    parser = ArgumentParser(description=__doc__)
    mp, op, pp = ModelParams(parser), ScaffoldOptimizationParams(parser), PipelineParams(parser)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--port", type=int, default=6009, help="Reserved; Scaffold-GS live viewer is not supported")
    parser.add_argument("--disable_viewer", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--publish_snapshots', action='store_true')
    parser.add_argument('--start_checkpoint')
    training_previews.add_arguments(parser)
    args = parser.parse_args()
    if args.model_type != "scaffold":
        parser.error("Use --model_type scaffold with the Scaffold-GS trainer")
    if args.eval:
        parser.error("Scaffold-GS training currently requires --eval off; use all-frame PSNR after training")
    if args.scaffold_voxel_size <= 0 or args.scaffold_n_offsets < 1 or args.scaffold_feat_dim < 1:
        parser.error("Scaffold voxel size, offset count and feature dimension must be positive")
    if args.publish_snapshots and not (Path(args.model_path)/'keyframes.json').is_file():
        parser.error('Published snapshots require the saved keyframes.json')
    safe_state(args.quiet)
    train(mp.extract(args), op.extract(args), pp.extract(args), set(args.save_iterations + [args.iterations]),
          set(args.test_iterations), training_previews.options(args), args.publish_snapshots, args.start_checkpoint)


if __name__ == "__main__":
    main()
