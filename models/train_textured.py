#!/usr/bin/env python3
"""Two-stage Textured Gaussians training on this repository's COLMAP cameras."""
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from random import randint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tqdm import tqdm
from arguments import ModelParams, PipelineParams
from models.textured_gaussian import TexturedGaussianModel, _backend
from scene import Scene
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim


def train(dataset, iterations, pretrain_iterations, cap, texture_resolution):
    os.makedirs(dataset.model_path, exist_ok=True)
    with open(os.path.join(dataset.model_path, "cfg_args"), "w") as stream:
        stream.write(str(Namespace(**vars(dataset))))
    _, _, strategy_class = _backend()
    model = TexturedGaussianModel(sh_degree=dataset.sh_degree, max_gaussians=cap,
                                  texture_resolution=texture_resolution)
    scene = Scene(dataset, model)
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                              dtype=torch.float32, device="cuda")

    for stage, steps in (("2dgs", pretrain_iterations), ("textured", iterations)):
        if stage == "textured":
            model.add_textures()
        optimizers = model.optimizer_setup()
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / steps))
        # Follow the released method: MCMC pretraining, no relocation or growth
        # after texture allocation (MCMC positional noise remains enabled).
        strategy = strategy_class(cap_max=cap, refine_stop_iter=25000 if stage == "2dgs" else 0)
        strategy.check_sanity(model.splats, optimizers)
        state = strategy.initialize_state()
        views = []
        progress = tqdm(range(steps), desc=f"Training {stage}", unit="iter")
        for step in progress:
            if not views:
                views = scene.getTrainCameras().copy()
            camera = views.pop(randint(0, len(views) - 1))
            degree = model.sh_degree if stage == "textured" else min(step // 1000, model.sh_degree)
            prediction, info = model.render(camera, background, degree)
            target = camera.original_image.to("cuda")
            loss = 0.8 * l1_loss(prediction, target) + 0.2 * (1.0 - ssim(prediction, target))
            loss.backward()
            strategy.step_post_backward(model.splats, optimizers, state, step, info,
                                        lr=optimizers["means"].param_groups[0]["lr"])
            for optimizer in optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if step % 10 == 0:
                progress.set_postfix(loss=f"{loss.item():.5f}", gaussians=len(model.splats["means"]))
        if stage == "2dgs":
            model.save(os.path.join(dataset.model_path, "pretrained_2dgs.pt"))
        else:
            path = Path(dataset.model_path) / "point_cloud" / f"iteration_{iterations}"
            path.mkdir(parents=True, exist_ok=True)
            model.save(path / "textured.pt")
    print("Textured Gaussians training complete")


def main():
    parser = ArgumentParser(description=__doc__)
    mp, pp = ModelParams(parser), PipelineParams(parser)
    parser.add_argument("--iterations", type=int, default=30000,
                        help="Texture refinement steps after 2DGS pretraining")
    parser.add_argument("--textured_pretrain_iterations", type=int, default=30000)
    parser.add_argument("--textured_max_gaussians", type=int, default=10000)
    parser.add_argument("--textured_resolution", type=int, default=50)
    parser.add_argument("--disable_viewer", action="store_true")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.model_type != "textured":
        parser.error("Use --model_type textured")
    if args.eval or args.depths or args.train_test_exp or args.antialiasing:
        parser.error("Textured backend does not support --eval, --depths, exposure or Graphdeco AA")
    if min(args.iterations, args.textured_pretrain_iterations, args.textured_max_gaussians, args.textured_resolution) < 1:
        parser.error("Iteration counts, Gaussian cap and texture resolution must be positive")
    safe_state(args.quiet)
    train(mp.extract(args), args.iterations, args.textured_pretrain_iterations,
          args.textured_max_gaussians, args.textured_resolution)


if __name__ == "__main__":
    main()
