"""Free-camera rendering for 3DGS and Scaffold checkpoints."""
import json
import math
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import numpy as np

MODES = ("RGB", "Child rings", "Anchor centers")


def camera_matrices(wxyz, position, fovy, width, height):
    import viser.transforms as vtf
    c2w = np.eye(4)
    c2w[:3, :3] = vtf.SO3(np.asarray(wxyz)).as_matrix()
    c2w[:3, 3] = position
    fovx = 2 * math.atan(math.tan(fovy / 2) * width / height)
    return np.linalg.inv(c2w).astype(np.float32), fovx


def projected_centers(xyz, w2c, focal, width, height):
    local = xyz @ w2c[:3, :3].T + w2c[:3, 3]
    ids = np.flatnonzero(np.isfinite(local).all(axis=1) & (local[:, 2] > .2))
    uv = local[ids, :2] / local[ids, 2:] * focal + [(width-1)/2, (height-1)/2]
    inside = ((uv >= -.5) & (uv < [width-.5, height-.5])).all(axis=1)
    return ids[inside], uv[inside], local[ids[inside]]


def projected_axes(local, scales, wxyz, camera_rotation, focal):
    """Project the decoded 3D one-sigma ellipsoid with the perspective Jacobian."""
    from scipy.spatial.transform import Rotation
    # The CUDA quaternion formula gives the identity for a zero quaternion.
    q = wxyz[:, [1, 2, 3, 0]].copy()
    q[(q == 0).all(axis=1), 3] = 1.
    basis = (camera_rotation @ Rotation.from_quat(q).as_matrix()) * scales[:, None, :]
    jac = np.broadcast_to(np.eye(2, 3), (len(local), 2, 3)).copy()
    jac[:, :, 2] = -local[:, :2] / local[:, 2:]
    pixels = (jac * (focal / local[:, 2])[:, None, None]) @ basis
    variances, axes = np.linalg.eigh(pixels @ pixels.transpose(0, 2, 1))
    return np.sqrt(np.maximum(variances[:, ::-1], 0)), np.degrees(np.arctan2(axes[:, 1, 1], axes[:, 0, 1]))


def overlay_geometry(rgb, mode, xyz, w2c, focal, scales=None, rotations=None):
    import cv2
    h, w = rgb.shape[:2]
    ids, uv, local = projected_centers(xyz, w2c, focal, w, h)
    if not len(ids):
        return rgb
    image = np.ascontiguousarray(rgb * .65, dtype=np.uint8)
    centers = np.rint(uv).astype(int)
    if mode == "Anchor centers":
        for center in centers:
            cv2.circle(image, tuple(center), 1, (60, 235, 255), -1, cv2.LINE_AA)
    else:
        ok = np.isfinite(scales[ids]).all(axis=1) & np.isfinite(rotations[ids]).all(axis=1)
        if ok.any():
            radii, degrees = projected_axes(local[ok], scales[ids[ok]], rotations[ids[ok]], w2c[:3, :3], focal)
            for center, radius, angle in zip(centers[ok], radii, degrees):
                if radius[0] >= .7 and radius[0] < 1e6:
                    cv2.ellipse(image, tuple(center), tuple(np.maximum(1, np.rint(radius).astype(int))),
                                angle, 0, 360, (255, 184, 77), 1, cv2.LINE_AA)
    return image


class NativeBackend:
    def __init__(self, ply, cfg):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to render 3DGS on this machine")
        from gaussian_renderer import render
        from scene.cameras import MiniCam
        from scene.gaussian_model import GaussianModel
        from utils.graphics_utils import getProjectionMatrix

        self.torch, self.MiniCam, self.projection = torch, MiniCam, getProjectionMatrix
        self.render = render
        self.model = GaussianModel(cfg.get("sh_degree", 3))
        self.model.load_ply(str(ply))
        self.background = torch.full((3,), float(cfg.get("white_background", False)), device="cuda")

    def render_image(self, wxyz, pos, fovy, width, height, antialiasing, mode="RGB"):
        w2c, fovx = camera_matrices(wxyz, pos, fovy, width, height)
        with self.torch.no_grad():
            view = self.torch.as_tensor(w2c, device="cuda").T.contiguous()
            proj = self.projection(.01, 100., fovx, fovy).T.cuda()
            cam = self.MiniCam(width, height, fovy, fovx, .01, 100., view, view @ proj)
            pipe = SimpleNamespace(debug=False, antialiasing=antialiasing,
                                   compute_cov3D_python=False, convert_SHs_python=False)
            rgb = self.render(cam, self.model, pipe, self.background)["render"]
            return rgb.permute(1, 2, 0).mul(255).round().byte().cpu().numpy()

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()


class ScaffoldBackend:
    def __init__(self, ply, cfg):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to render Scaffold-GS on this machine")
        if cfg.get("scaffold_upstream"):
            from scripts.upstream_scaffold import load_saved_model, parameters, runtime
            root = ply.parent.parent.parent
            plan = json.loads((root/'upstream.json').read_text())
            api = runtime(plan)
            dataset, _, _ = parameters(api, root, plan)
            self.model = load_saved_model(api, dataset, ply.parent)
            self.renderer = api.renderer
        else:
            from models import scaffold_renderer
            from models.scaffold_gaussian import ScaffoldGaussianModel
            self.renderer = scaffold_renderer
            self.model = ScaffoldGaussianModel(feat_dim=cfg.get("scaffold_feat_dim", 32),
                        n_offsets=cfg.get("scaffold_n_offsets", 10),
                        voxel_size=cfg.get("scaffold_voxel_size", .001), update_init_factor=16, appearance_dim=0)
            self.model.load_ply(str(ply))
            self.model.eval()
        from scene.cameras import MiniCam
        from utils.graphics_utils import getProjectionMatrix
        self.torch, self.MiniCam, self.projection = torch, MiniCam, getProjectionMatrix
        self.background = torch.full((3,), float(cfg.get("white_background", False)), device="cuda")

    def render_image(self, wxyz, pos, fovy, width, height, antialiasing, mode="RGB"):
        torch = self.torch
        w2c, fovx = camera_matrices(wxyz, pos, fovy, width, height)
        with torch.no_grad():
            view = torch.as_tensor(w2c, device="cuda").T.contiguous()
            proj = self.projection(.01, 100., fovx, fovy).T.cuda()
            cam = self.MiniCam(width, height, fovy, fovx, .01, 100., view, view @ proj)
            pipe = SimpleNamespace(debug=False, antialiasing=antialiasing, compute_cov3D_python=False)
            visible = self.renderer.prefilter_voxel(cam, self.model, pipe, self.background)
            if not bool(visible.any()):
                bg = np.rint(self.background.cpu().numpy()*255).astype(np.uint8)
                return np.broadcast_to(bg, (height, width, 3)).copy()
            rgb = self.renderer.render(cam, self.model, pipe, self.background, visible_mask=visible)["render"]
            image = rgb.clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
            if mode == "RGB":
                return image
            focal = .5*height/math.tan(fovy*.5)
            ids = visible.nonzero(as_tuple=True)[0]
            limit = 12000 if mode == "Anchor centers" else 400
            sample = ids[::max(1, math.ceil(len(ids)/limit))]
            if mode == "Anchor centers":
                xyz = self.model.get_anchor[sample].cpu().numpy()
                return overlay_geometry(image, mode, xyz, w2c, focal)
            xyz, _, _, scales, rotations = self.renderer.generate_neural_gaussians(cam, self.model, sample)
            return overlay_geometry(image, mode, xyz.cpu().numpy(), w2c, focal, scales.cpu().numpy(),
                                    rotations.cpu().numpy())

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()


def _run_backend(channel, ply, cfg):
    try:
        backend = ScaffoldBackend(ply, cfg)
        channel.send((True, None))
        while (args := channel.recv()) is not None:
            try:
                channel.send((True, backend.render_image(*args)))
            except Exception as exc:
                channel.send((False, f"{type(exc).__name__}: {exc}"))
        backend.close()
    except EOFError:
        pass
    except Exception as exc:
        channel.send((False, f"{type(exc).__name__}: {exc}"))
    finally:
        channel.close()


class UpstreamBackend:
    """Keep the upstream CUDA extension in its own process across camera updates."""
    def __init__(self, ply, cfg):
        ctx = multiprocessing.get_context("spawn")
        self.channel, child = ctx.Pipe()
        self.process = ctx.Process(target=_run_backend, args=(child, ply, cfg), daemon=True)
        self.process.start()
        child.close()
        try:
            self._receive()
        except BaseException:
            self.close()
            raise

    def _receive(self):
        ok, result = self.channel.recv()
        if not ok:
            raise RuntimeError(result)
        return result

    def render_image(self, *args):
        self.channel.send(args)
        return self._receive()

    def close(self):
        try:
            if self.process.is_alive():
                self.channel.send(None)
        except (EOFError, BrokenPipeError, OSError):
            pass
        finally:
            self.channel.close()
            self.process.join(2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()
