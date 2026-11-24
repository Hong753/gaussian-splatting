"""Training hooks imported inside the pinned AmbiSuR source tree."""

from contextlib import ExitStack
from functools import lru_cache
import json
import fcntl
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud
from joint_report import (
    dashboard,
    depth_images,
    export_mesh,
    gallery,
    panel,
    save_image,
    score,
    summary,
    to_image,
    video,
    write_json,
)
from joint_surface import Surface
from joint_tetmesh import marching_tetrahedra


def settings():
    run = Path(os.environ["JOINT_RUN"])
    return run, json.loads((run / "config.json").read_text()), json.loads((run / "cameras.json").read_text())


@lru_cache(maxsize=4)
def calibrated_rays(w, h, fx, fy, cx, cy, scale):
    y, x = torch.meshgrid(
        torch.arange(int(h / scale), device="cuda"),
        torch.arange(int(w / scale), device="cuda"),
        indexing="ij",
    )
    return torch.stack(((x * scale - cx) / fx, (y * scale - cy) / fy, torch.ones_like(x)), dim=-1)


class View(Camera):
    def __init__(self, row, run):
        pose = np.array(row["w2c"])
        w, h = row["width"], row["height"]
        fx, fy, cx, cy = row["K"]
        super().__init__(
            row["id"],
            pose[:3, :3].T,
            pose[:3, 3],
            2 * math.atan(w / (2 * fx)),
            2 * math.atan(h / (2 * fy)),
            w,
            h,
            row.get("image", ""),
            row["name"],
            row["id"],
            preload_img=False,
            depth_dict={},
            data_device="cpu",
        )
        self.Cx, self.Cy = cx, cy
        self.projection_matrix[2, 0] = 2 * (cx + 0.5) / w - 1
        self.projection_matrix[2, 1] = 2 * (cy + 0.5) / h - 1
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.prior_file = run / "da3" / f"{self.uid:08d}.npz"

    def get_rays(self, scale=1.0):
        return calibrated_rays(self.image_width, self.image_height, self.Fx, self.Fy, self.Cx, self.Cy, scale)

    def get_image(self):
        with Image.open(self.image_path) as im:
            arr = np.array(im.convert("RGB").resize(self.resolution, Image.Resampling.LANCZOS))
        rgb = torch.from_numpy(arr).cuda().permute(2, 0, 1).float() / 255
        return rgb, (0.299 * rgb[:1] + 0.587 * rgb[1:2] + 0.114 * rgb[2:3])

    @lru_cache(maxsize=4)
    def prior(self):
        with np.load(self.prior_file) as p:
            k = p["K"]
            depth = torch.from_numpy(p["depth"]).cuda()
            valid = torch.from_numpy(p["valid"]).cuda()
        # Map the calibrated DA3 output grid to this camera before comparing z.
        x = (torch.arange(self.image_width, device=depth.device) - self.Cx) / self.Fx * k[0, 0] + k[0, 2]
        y = (torch.arange(self.image_height, device=depth.device) - self.Cy) / self.Fy * k[1, 1] + k[1, 2]
        u, v = x.round().long(), y.round().long()
        inside = ((u >= 0) & (u < depth.shape[1]))[None, :] & ((v >= 0) & (v < depth.shape[0]))[:, None]
        u, v = u.clamp(0, depth.shape[1] - 1), v.clamp(0, depth.shape[0] - 1)
        z = depth[v[:, None], u]
        return z, inside & valid[v[:, None], u] & torch.isfinite(z) & (z > 0)


class JointScene:
    def __init__(self, dataset, gs):
        self.run, self.config, self.manifest = settings()
        self.model_path = dataset.model_path
        self.gaussians = gs
        gs.disable_trunc, gs.trunc_sigma = dataset.disable_trunc, dataset.trunc_sigma
        self.fit = [View(r, self.run) for r in self.manifest["frames"] if r["split"] == "train"]
        centers = np.stack([-c.R @ c.T for c in self.fit])
        self.cameras_extent = float(1.1 * np.linalg.norm(centers - centers.mean(0), axis=1).max())
        if self.cameras_extent <= 0:
            raise ValueError("Training cameras have zero baseline")
        axes = np.stack([c.R[:, 2] for c in self.fit])
        min_dis, max_dis = dataset.multi_view_min_dis, dataset.multi_view_max_dis
        for i, c in enumerate(self.fit):
            distance = np.linalg.norm(centers - centers[i], axis=1)
            angle = axes @ axes[i]
            ids = np.flatnonzero(
                (distance > min_dis)
                & (distance < max_dis)
                & (angle > math.cos(math.radians(dataset.multi_view_max_angle)))
            )
            c.nearest_id = ids[np.argsort(distance[ids])[: dataset.multi_view_num]].tolist()
        count = sum(bool(c.nearest_id) for c in self.fit)
        print(f"Fitting cameras with eligible multiview neighbors: {count}/{len(self.fit)}", flush=True)
        if not count:
            raise ValueError(
                "No fitting cameras meet the multiview distance/angle limits. Check scene scale."
            )
        with np.load(self.run / "da3/seed.npz") as p:
            xyz = p["xyz"]
            cloud = BasicPointCloud(
                points=xyz, colors=p["rgb"].astype(np.float32) / 255, normals=np.zeros_like(xyz)
            )
        gs.create_from_pcd(cloud, self.cameras_extent)

    def getTrainCameras(self):
        return self.fit

    def getTestCameras(self):
        return []  # Held-out images are accessed exclusively by the evaluator.


class Evaluator:
    def __init__(self, run, manifest, surface, renderer):
        self.run, self.data, self.surface, self.render = run, manifest, surface, renderer
        rng = random.Random(42)
        self.preview_rows = []
        for split in ("train", "test"):
            choices = [r for r in manifest["frames"] if r["split"] == split]
            self.preview_rows.extend(rng.sample(choices, min(4, len(choices))))

    @torch.no_grad()
    def preview(self, step):
        surf = self.surface
        mesh = surf.mesh()[:2] if surf.ids is not None else None
        rows = []
        for r in self.preview_rows:
            cam = View(r, self.run)
            gt, _ = cam.get_image()
            pkg = self.render(cam)
            pred = pkg["render"].clamp(0, 1)
            psnr = score(gt, pred)
            imgs = [to_image(t.permute(1, 2, 0)) for t in (gt, pred, 4 * (gt - pred).abs())]
            labels = ["Ground truth", "Gaussian RGB", "RGB error x4"]
            gd = pkg["plane_depth"][0]
            if mesh is not None:
                md, n, _ = surf.render(cam, *mesh)
                imgs.extend(depth_images(gd, md) + [to_image((n * 0.5 + 0.5) * (md > 0)[..., None])])
                labels.extend(["Gaussian z", "Mesh first-hit z", "Mesh normals"])
            else:
                imgs.append(depth_images(gd, gd)[0])
                labels.append("Gaussian z")
            rel = f"previews/{step:06d}/{r['id']:08d}.jpg"
            save_image(self.run / rel, panel(imgs, labels, 440))
            rows.append(dict(name=r["name"], split=r["split"], panel=rel, psnr_db=psnr))
        metrics = summary(rows)
        item = dict(step=step, scope="fixed preview views", metrics=metrics, frames=rows)
        write_json(self.run / f"previews/{step:06d}/metrics.json", item)
        msg = f"Step {step:,} · " + (
            "joint Gaussian and mesh training" if mesh is not None else "Gaussian warm-up"
        )
        dashboard(self.run, msg, rows, metrics)
        html_path = self.run / f"previews/{step:06d}/index.html"
        from joint_report import publish

        publish(
            html_path,
            (self.run / "index.html")
            .read_text()
            .replace('<meta http-equiv="refresh" content="15">', "")
            .replace('<meta charset="utf-8">', '<meta charset="utf-8"><base href="../../">'),
        )
        print(
            f"{msg} | preview PSNR: train {metrics['train']['psnr_db']:.2f} dB, held-out {metrics['test']['psnr_db']:.2f} dB",
            flush=True,
        )
        return rows, metrics

    @torch.no_grad()
    def evaluate(self, out, step):
        out.mkdir(parents=True, exist_ok=True)
        with (out / "evaluation.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            required = ["metrics.json", "index.html", "original.mp4", "random.mp4"]
            if self.surface.ids is not None:
                required += ["mesh.ply", "random_mesh.mp4"]
            if all((out / name).is_file() for name in required):
                print(f"Full evaluation already published in {out}", flush=True)
                return
            self._evaluate(out, step)

    @torch.no_grad()
    def save_surface(self, path):
        v, f, e, w = self.surface.mesh()
        dc = self.surface.gs._features_dc[self.surface.ids, 0] * 0.28209479177387814 + 0.5
        color = torch.lerp(dc[e[:, 0] // 9], dc[e[:, 1] // 9], w)
        export_mesh(path, v.cpu().numpy(), f.cpu().numpy(), color.cpu().numpy())
        return v, f

    def _evaluate(self, out, step):
        mesh = self.save_surface(out / "mesh.ply") if self.surface.ids is not None else None
        rows = []
        frames = self.data["frames"]
        with video(
            out / "original.mp4", max(r["width"] for r in frames), max(r["height"] for r in frames)
        ) as append:
            for i, r in enumerate(frames, 1):
                cam = View(r, self.run)
                gt, _ = cam.get_image()
                pred = self.render(cam, geometry=False)["render"].clamp(0, 1)
                psnr = score(gt, pred)
                frame = to_image(pred.permute(1, 2, 0))
                name = f"{r['id']:08d}_{Path(r['name']).stem}"
                save_image(out / f"frames/{name}.png", frame)
                rel = f"comparisons/{name}.jpg"
                save_image(
                    out / rel,
                    panel(
                        [
                            to_image(gt.permute(1, 2, 0)),
                            frame,
                            to_image((4 * (gt - pred).abs()).permute(1, 2, 0)),
                        ],
                        ["Ground truth", "Gaussian RGB", "RGB error x4"],
                    ),
                )
                append(frame)
                rows.append(
                    dict(
                        name=r["name"],
                        split=r["split"],
                        psnr_db=psnr,
                        comparison=rel,
                        render=f"frames/{name}.png",
                    )
                )
                if i % 100 == 0:
                    print(f"Evaluated {i}/{len(frames)} registered images", flush=True)
        novel = self.data["random"]
        with ExitStack() as stack:
            w, h = novel[0]["width"], novel[0]["height"]
            append = stack.enter_context(video(out / "random.mp4", w, h))
            append_mesh = (
                stack.enter_context(video(out / "random_mesh.mp4", w, h)) if mesh is not None else None
            )
            for r in novel:
                cam = View(r, self.run)
                pred = self.render(cam, geometry=False)["render"]
                append(to_image(pred.permute(1, 2, 0)))
                if append_mesh:
                    z, n, _ = self.surface.render(cam, *mesh)
                    append_mesh(to_image((n * 0.5 + 0.5) * (z > 0)[..., None]))
        metrics = summary(rows)
        write_json(
            out / "metrics.json",
            dict(step=step, scope="all registered cameras", metrics=metrics, frames=rows),
        )
        gallery(out, rows, metrics, mesh is not None)
        print(f"Full-frame PSNR: {json.dumps(metrics)}. Outputs: {out}", flush=True)


class JointTraining:
    def __init__(self, scene, gs, pipe, background, opt):
        self.scene, self.gs, self.opt = scene, gs, opt
        self.run, self.config = scene.run, scene.config
        from gaussian_renderer import render

        self.render = lambda cam, geometry=True: render(
            cam, gs, pipe, background, return_plane=geometry, return_depth_normal=False, ray_reg=-1
        )
        self.surface = Surface(gs, self.config, marching_tetrahedra)
        self.eval = Evaluator(self.run, scene.manifest, self.surface, self.render)

    def loss(self, step, cam, pkg, gt):
        cfg = self.config
        if step < cfg["mesh_from"] or (step - cfg["mesh_from"]) % cfg["mesh_every"]:
            return 0.0
        if self.surface.ids is None:
            self.surface.initialize(self.scene.fit, self.render)
        neighbor = self.scene.fit[random.choice(cam.nearest_id)] if cam.nearest_id else None
        loss, stats = self.surface.loss(cam, neighbor, pkg, gt, step)
        if (step - cfg["mesh_from"]) % 100 == 0 or step == self.opt.iterations:
            with (self.run / "mesh_training.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(step=step, **stats)) + "\n")
        return loss

    def after_step(self, step, stack):
        self.surface.finish_step()
        end = step == self.opt.iterations
        if step == 1 or step % self.config["preview_every"] == 0 or step in self.config["checkpoints"] or end:
            rows, metrics = self.eval.preview(step)
        if step in self.config["checkpoints"] or end:
            out = self.run / "checkpoints" / f"step_{step:06d}.pt"
            out.parent.mkdir(exist_ok=True)
            tmp = out.with_suffix(".partial")
            state = dict(
                step=step,
                gaussians=self.gs.capture(),
                surface=self.surface.state(),
                opt=vars(self.opt),
                scene_config=self.config,
                stack=[c.uid for c in (stack or [])],
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(),
                python_rng=random.getstate(),
                numpy_rng=np.random.get_state(),
            )
            torch.save(state, tmp)
            tmp.replace(out)
            if self.surface.ids is not None:
                self.eval.save_surface(out.with_name(f"mesh_{step:06d}.ply"))
                dashboard(self.run, f"Step {step:,} · joint checkpoint published.", rows, metrics)
            print(f"Saved joint checkpoint {out}", flush=True)
        if end:
            self.gs.save_ply(str(self.run / "model" / "gaussians.ply"))
            self.eval.evaluate(self.run / "final", step)
            dashboard(
                self.run, f"Completed step {step:,}. Full evaluation and videos are ready.", rows, metrics
            )

    def resume(self, path):
        state = torch.load(path, weights_only=False)
        self.gs.restore(state["gaussians"], self.opt)
        self.surface.restore(state["surface"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        cams = {c.uid: c for c in self.scene.fit}
        return state["step"], [cams[i] for i in state["stack"]]


def evaluate_checkpoint(step):
    from gaussian_renderer import render

    run, conf, data = settings()
    state = torch.load(run / "checkpoints" / f"step_{step:06d}.pt", weights_only=False, map_location="cpu")
    gs = GaussianModel(3)
    # Positions and appearance are sufficient for rendering a checkpoint. Keep
    # Adam moments off the evaluation GPU so a mid-run render fits a second card.
    captured = state.pop("gaussians")
    gs.active_sh_degree = captured[0]
    for key, value in zip(
        ("_xyz", "_knn_f", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"),
        captured[1:8],
    ):
        setattr(gs, key, torch.nn.Parameter(value.detach().cuda(), requires_grad=False))
    del captured
    gs.disable_trunc, gs.trunc_sigma = False, 2.0
    surface = Surface(gs, conf, marching_tetrahedra)
    if state["surface"] is not None:
        for key in ("ids", "spacing", "tets", "observed", "sdf"):
            setattr(surface, key, state["surface"][key].cuda())
    del state
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    bg = torch.zeros(3, device="cuda")

    def renderer(cam, geometry=True):
        return render(cam, gs, pipe, bg, return_plane=geometry, return_depth_normal=False, ray_reg=-1)

    out = run / ("final" if step == conf["iterations"] else f"evaluation_{step:06d}")
    Evaluator(run, data, surface, renderer).evaluate(out, step)
    if out.name == "final":
        dashboard(run, f"Completed step {step:,}. Full evaluation and videos are ready.")


if __name__ == "__main__":
    import sys

    evaluate_checkpoint(int(sys.argv[1]))
