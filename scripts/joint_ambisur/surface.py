"""Gaussian-dependent pivot positions, signed field and differentiable first-hit mesh."""

import itertools
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import Delaunay, cKDTree


def tetrahedra(points):
    xyz = points.detach().cpu().double().numpy()
    tets = Delaunay(xyz).simplices.copy()
    # The marching table assumes a consistent tet winding.
    for chunk in np.array_split(tets, max(1, math.ceil(len(tets) / 65536))):
        delta = xyz[chunk[:, 1:]] - xyz[chunk[:, :1]]
        flipped = np.linalg.det(delta) < 0
        chunk[flipped, :2] = chunk[flipped, 1::-1]
    return torch.from_numpy(tets.astype(np.int64)).to(points.device)


def pivots(means, scales, rotations, spacing):
    corners = means.new_tensor(list(itertools.product((-1.0, 1.0), repeat=3)))
    # Keep the signed field locally resolvable after AmbiSuR flattens a Gaussian.
    radius = 2 * scales + 0.35 * spacing[:, None]
    xyz = means[:, None] + (radius[:, None] * corners) @ rotations.transpose(1, 2)
    return torch.cat([means[:, None], xyz], 1).reshape(-1, 3), radius.amax(1).repeat_interleave(9)


def extract(points, radius, sdf, tets, observed, marching):
    with torch.no_grad():
        cells = tets[observed[tets].all(1)]
    endpoints, _, face_batches, edge_batches = marching(points[None], cells, sdf[None], radius[None, :, None])
    ends, values = endpoints[0]
    edge, faces = edge_batches[0], face_batches[0]
    w = values[:, 0].abs() / values.abs().sum(1).clamp_min(1e-12)
    verts = torch.lerp(ends[:, 0], ends[:, 1], w)
    with torch.no_grad():
        good = (ends[:, 0] - ends[:, 1]).norm(dim=1) <= 2 * radius[edge].sum(1)
        faces = faces[good[faces].all(1)]
        # Long same-sign edges can also connect separate walls.
        corners = verts[faces]
        scale = radius[edge].amax(1)[faces].amax(1)
        faces = faces[((corners - corners.roll(1, 1)).norm(dim=2).amax(1) <= 5 * scale)]
    return verts, faces, edge, w


def camera_xyz(cam, xyz):
    return xyz @ cam.world_view_transform[:3, :3] + cam.world_view_transform[3, :3]


def pixels(cam, xyz):
    z = xyz[:, 2].clamp_min(1e-7)
    return xyz[:, 0] / z * cam.Fx + cam.Cx, xyz[:, 1] / z * cam.Fy + cam.Cy


def lookup(cam, xyz, depth, valid):
    u, v = pixels(cam, xyz)
    ix = u.round().long().clamp(0, cam.image_width - 1)
    iy = v.round().long().clamp(0, cam.image_height - 1)
    good = (
        (xyz[:, 2] > cam.znear)
        & (u >= 0)
        & (u <= cam.image_width - 1)
        & (v >= 0)
        & (v <= cam.image_height - 1)
        & valid[iy, ix]
    )
    return depth[iy, ix], good


def raster(ctx, cam, verts, faces):
    import nvdiffrast.torch as dr

    shape = (cam.image_height, cam.image_width)
    if not faces.numel():
        zero = verts.new_zeros(shape) + verts.sum() * 0
        return zero, zero[..., None].expand(-1, -1, 3), zero
    xyz = camera_xyz(cam, verts)
    z = xyz[:, 2]
    # nvdiffrast row 0 is NDC y=-1. Our row 0 is the top of the COLMAP image.
    xclip = 2 * cam.Fx / shape[1] * xyz[:, 0] + (2 * (cam.Cx + 0.5) / shape[1] - 1) * z
    yclip = 2 * cam.Fy / shape[0] * xyz[:, 1] + (2 * (cam.Cy + 0.5) / shape[0] - 1) * z
    a = (cam.zfar + cam.znear) / (cam.zfar - cam.znear)
    b = -2 * cam.znear * cam.zfar / (cam.zfar - cam.znear)
    clip = torch.stack((xclip, yclip, a * z + b, z), -1)[None].contiguous()
    tri = faces.int().contiguous()
    rast, _ = dr.rasterize(ctx, clip, tri, resolution=shape)
    depth, _ = dr.interpolate(z[None, :, None].contiguous(), rast, tri)
    fid = rast[0, :, :, 3].long() - 1
    hit = fid >= 0
    corners = xyz[faces]
    normals = F.normalize(
        torch.linalg.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), dim=-1
    )
    normals = normals * torch.where((normals * corners[:, 0]).sum(-1, keepdim=True) > 0, -1.0, 1.0)
    normal = normals[fid.clamp_min(0)] * hit[..., None]
    coverage = dr.antialias(hit[None, :, :, None].float(), rast, clip, tri)[0, :, :, 0]
    return depth[0, :, :, 0], normal, coverage


def robust(a, b):
    return F.smooth_l1_loss(a, b, reduction="none", beta=0.03)


def masked(values, mask):
    return torch.where(mask, values, 0.0).sum() / mask.sum().clamp_min(1)


def reprojection(cam, neighbor, depth, hit, gt, sample_max=8192):
    ids = torch.nonzero(hit.flatten()).flatten()
    if not ids.numel():
        return depth.sum() * 0, 0
    if len(ids) > sample_max:
        ids = ids[torch.randint(len(ids), (sample_max,), device=ids.device)]
    y, x = ids // cam.image_width, ids % cam.image_width
    rays = torch.stack(((x - cam.Cx) / cam.Fx, (y - cam.Cy) / cam.Fy, torch.ones_like(x)), 1)
    local = rays * depth[y, x, None]
    world = (local - cam.world_view_transform[3, :3]) @ cam.world_view_transform[:3, :3].T
    xyz = camera_xyz(neighbor, world)
    src_z, src_mask = neighbor.prior()
    other, valid = lookup(neighbor, xyz.detach(), src_z, src_mask)
    valid &= (other - xyz[:, 2].detach()).abs() < 0.05 * other
    u, v = pixels(neighbor, xyz)
    grid = torch.stack((2 * u / (neighbor.image_width - 1) - 1, 2 * v / (neighbor.image_height - 1) - 1), 1)
    image, _ = neighbor.get_image()
    photo = F.grid_sample(image[None], grid[None, :, None], align_corners=True)[0, :, :, 0]
    return masked(robust(photo, gt[:, y, x]).mean(0), valid), int(valid.sum())


class Surface:
    def __init__(self, gs, config, marching):
        self.gs, self.cfg, self.marching = gs, config, marching
        self.ids = self.sdf = self.optimizer = None
        self.tet_seconds = 0.0
        self.ctx = None
        self.pending = None

    def positions(self):
        g, idx = self.gs, self.ids
        from utils.general_utils import build_rotation

        return pivots(
            g.get_xyz[idx],
            g.scaling_activation(g._scaling[idx]),
            build_rotation(g._rotation[idx]),
            self.spacing,
        )

    def retriangulate(self, points):
        start = time.monotonic()
        self.tets = tetrahedra(points)
        self.tet_seconds += time.monotonic() - start
        print(
            f"Mesh connectivity: {len(self.tets):,} tetrahedra ({self.tet_seconds:.1f}s cumulative)",
            flush=True,
        )

    @torch.no_grad()
    def initialize(self, cams, render):
        weights = self.gs.get_opacity.flatten()
        available = torch.nonzero(weights > 0.05).flatten()
        if len(available) < 4:
            raise ValueError("Fewer than four supported Gaussians at mesh initialization")
        n = min(len(available), self.cfg["mesh_gaussians"])
        self.ids = available[torch.multinomial(weights[available], n, replacement=False)].sort().values
        centers = self.gs.get_xyz[self.ids].detach().cpu().numpy()
        distances = cKDTree(centers).query(centers, k=min(4, n))[0][:, -1]
        self.spacing = torch.as_tensor(
            np.maximum(distances, 1e-5), device=weights.device, dtype=weights.dtype
        )
        p, r = self.positions()
        self.retriangulate(p)
        total, count = torch.zeros_like(r), torch.zeros_like(r)
        ordered = sorted(cams, key=lambda c: c.uid)
        indices = np.linspace(0, len(ordered) - 1, min(len(ordered), self.cfg["init_views"]), dtype=int)
        for seq, idx in enumerate(indices, 1):
            cam = ordered[idx]
            pkg = render(cam)
            z = pkg["plane_depth"].squeeze(0)
            valid = (pkg["rendered_alpha"].squeeze(0) > 0.7) & (z > 0) & torch.isfinite(z)
            xyz = camera_xyz(cam, p)
            sample, ok = lookup(cam, xyz, z, valid)
            prior, support = cam.prior()
            truth, trusted = lookup(cam, xyz, prior, support)
            # Near a trusted DA3 surface, give it precedence over an inconsistent GS surface.
            sample = torch.where(trusted, truth, sample)
            delta = (sample - xyz[:, 2]) / r.clamp_min(1e-5)
            ok = (ok | trusted) & (delta >= -2) & (delta <= 3)
            total += torch.where(ok, delta.clamp(-1, 1), 0.0)
            count += ok
            if seq % 32 == 0:
                print(f"Signed field initialization {seq}/{len(indices)} fitting cameras", flush=True)
        self.observed = count >= 2
        self.sdf = torch.nn.Parameter(
            torch.where(self.observed, total / count.clamp_min(1), torch.ones_like(r))
        )
        self.optimizer = torch.optim.Adam([self.sdf], lr=0.01)
        if not (self.sdf[self.observed] < 0).any():
            raise RuntimeError("No supported surface crossings. Inspect DA3 and Gaussian depth previews.")

    def mesh(self):
        p, r = self.positions()
        v, f, e, w = extract(p, r, self.sdf, self.tets, self.observed, self.marching)
        return v, f, e, w

    def render(self, cam, v, f):
        if self.ctx is None:
            import nvdiffrast.torch as dr

            self.ctx = dr.RasterizeCudaContext(device=v.device)
        return raster(self.ctx, cam, v, f)

    @torch.no_grad()
    def propose_sites(self, cam, pkg, prior, trusted, depth):
        room = self.cfg["mesh_gaussians"] + self.cfg["extra_sites"] - len(self.ids)
        if room <= 0:
            return
        # Prefer existing visible Gaussians on a trusted surface that is missing
        # or at the wrong depth in the *current mesh*. Keep earlier pivot IDs stable.
        candidates = (pkg["out_observe"].flatten() > 0) & (self.gs.get_opacity.flatten() > 0.05)
        candidates[self.ids] = False
        ids = candidates.nonzero().flatten()
        xyz = camera_xyz(cam, self.gs.get_xyz[ids])
        truth, ok = lookup(cam, xyz, prior, trusted)
        current, _ = lookup(cam, xyz, depth, trusted)
        error = (current - truth).abs() / truth.clamp_min(1e-5)
        ok &= ((xyz[:, 2] - truth).abs() < 0.04 * truth) & (error > 0.08)
        ids = ids[ok]
        if not len(ids):
            return
        n = min(512, room, len(ids))
        selected = ids[torch.multinomial(error[ok].clamp(max=1), n, replacement=False)]
        self.pending = (selected, cam)

    @torch.no_grad()
    def finish_step(self):
        if self.optimizer is None:
            return
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.pending is None:
            return
        selected, cam = self.pending
        self.pending = None
        self.ids = torch.cat((self.ids, selected))
        centers = self.gs.get_xyz[self.ids].detach().cpu().numpy()
        distances = cKDTree(centers).query(centers[-len(selected) :], k=4)[0][:, -1]
        extra = self.spacing.new_tensor(np.maximum(distances, 1e-5))
        self.spacing = torch.cat((self.spacing, extra))
        p, r = self.positions()
        count = 9 * len(selected)
        xyz = camera_xyz(cam, p[-count:])
        z, valid = lookup(cam, xyz, *cam.prior())
        delta = (z - xyz[:, 2]) / r[-count:].clamp_min(1e-5)
        valid &= (delta >= -2) & (delta <= 3)
        added = torch.where(valid, delta.clamp(-1, 1), 1.0)
        self.observed = torch.cat((self.observed, valid))
        old = self.sdf
        self.sdf = torch.nn.Parameter(torch.cat((old.detach(), added)))
        group = self.optimizer.param_groups[0]
        state = self.optimizer.state.pop(old)
        for key in ("exp_avg", "exp_avg_sq"):
            state[key] = torch.cat((state[key], torch.zeros_like(added)))
        group["params"] = [self.sdf]
        self.optimizer.state[self.sdf] = state
        self.retriangulate(p)
        print(
            f"Added {len(selected)} supported mesh sites at missing first hits; {len(self.ids)} sites",
            flush=True,
        )

    def loss(self, cam, other, pkg, gt, iteration):
        p, r = self.positions()
        refresh = (
            iteration > self.cfg["mesh_from"]
            and (iteration - self.cfg["mesh_from"]) % self.cfg["retet_every"] == 0
        )
        prior, trusted = cam.prior()
        with torch.no_grad():
            xyz = camera_xyz(cam, p)
            target, valid = lookup(cam, xyz, prior, trusted)
            delta = (target - xyz[:, 2]) / r.clamp_min(1e-5)
            valid &= (delta >= -2) & (delta <= 3)
            # Permit new crossings in regions observed after the initialization sweep.
            self.observed |= valid
        field = masked(robust(self.sdf, delta.clamp(-1, 1)), valid)
        v, f, _, _ = extract(p, r, self.sdf, self.tets, self.observed, self.marching)
        md, mn, coverage = self.render(cam, v, f)
        gd = pkg["plane_depth"][0].clamp_min(1e-6)
        gs_ok = (pkg["rendered_alpha"][0].detach() > 0.7) & (gd.detach() > 1e-5) & torch.isfinite(gd.detach())
        hit = md.detach() > 0
        overlap = hit & gs_ok
        # Train both representations. The reverse term requires independent DA3 agreement.
        mz = md.clamp_min(1e-6).log()
        gz = gd.log()
        depth = masked(robust(mz, gz.detach()), overlap)
        anchored = hit & trusted & ((md.detach() - prior).abs() < 0.05 * prior)
        reverse = masked(robust(gz, mz.detach()), anchored & gs_ok)
        anchor = masked(robust(mz, prior.clamp_min(1e-6).log()), trusted & hit)
        gn = F.normalize(pkg["rendered_normal"], dim=0).permute(1, 2, 0)
        normal = masked(1 - (gn.detach() * mn).sum(-1).abs().clamp(max=1), overlap)
        cover = masked(1 - coverage, trusted)
        photo, photo_n = reprojection(cam, other, md, hit, gt) if other is not None else (md.sum() * 0, 0)
        if refresh:
            self.propose_sites(cam, pkg, prior, trusted, md.detach())
            if self.pending is None:
                self.retriangulate(p)
        loss = (
            0.10 * depth
            + 0.025 * reverse
            + 0.10 * anchor
            + 0.02 * normal
            + 0.025 * cover
            + 0.10 * photo
            + 0.02 * field
        )
        stats = dict(
            sites=len(self.ids),
            faces=len(f),
            mesh_hits=int(hit.sum()),
            paired=int(overlap.sum()),
            prior_hits=int((trusted & hit).sum()),
            reverse_hits=int((anchored & gs_ok).sum()),
            photo_samples=photo_n,
            signed_samples=int(valid.sum()),
            coverage=float(hit.float().mean()),
            depth=float(depth.detach()),
            reverse=float(reverse.detach()),
            anchor=float(anchor.detach()),
            normal=float(normal.detach()),
            photo=float(photo.detach()),
            field=float(field.detach()),
            missing=float(cover.detach()),
            topology_seconds=self.tet_seconds,
        )
        return loss, stats

    def state(self):
        if self.ids is None:
            return None
        return {
            **{k: getattr(self, k).detach() for k in ("ids", "spacing", "tets", "observed", "sdf")},
            "optimizer": self.optimizer.state_dict(),
            "tet_seconds": self.tet_seconds,
        }

    def restore(self, state):
        if state is None:
            return
        for key in ("ids", "spacing", "tets", "observed", "tet_seconds"):
            setattr(self, key, state[key])
        self.sdf = torch.nn.Parameter(state["sdf"])
        self.optimizer = torch.optim.Adam([self.sdf], lr=0.01)
        self.optimizer.load_state_dict(state["optimizer"])
