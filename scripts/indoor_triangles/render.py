"""Small, brute-force CPU reference for perspective triangle compositing."""

from dataclasses import dataclass

import numpy as np

OFFSETS = np.array([[-.25, -.25], [.25, -.25], [-.25, .25], [.25, .25]])


def pixels(camera):
    return np.stack(np.meshgrid(np.arange(camera.width),np.arange(camera.height)),axis=-1)


@dataclass(frozen=True)
class Camera:
    center: np.ndarray
    rotation: np.ndarray  # World to camera. Camera x right, y down, z forward.
    intrinsics: np.ndarray  # fx, fy, cx, cy in pixel-center coordinates.
    width: int
    height: int

    def local(self, points):
        return (points - self.center) @ self.rotation.T

    def project(self, points):
        xyz = self.local(points)
        return xyz[..., :2] / xyz[..., 2:] * self.intrinsics[:2] + self.intrinsics[2:]

    def rays(self, pixels):
        xy = (pixels - self.intrinsics[2:]) / self.intrinsics[:2]
        directions = np.concatenate((xy, np.ones_like(xy[..., :1])), axis=-1)
        return unit(directions @ self.rotation)


def unit(value):
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-12)


def cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def normals(vertices, faces):
    a, b, c = np.moveaxis(vertices[faces], 1, 0)
    return unit(np.cross(b - a, c - a))


def edge_neighbors(faces):
    """Neighbors indexed by the directed edges 0→1, 1→2, 2→0."""
    incidence = {}
    for i, face in enumerate(faces):
        for e in range(3):
            edge = tuple(sorted((face[e], face[(e + 1) % 3])))
            incidence.setdefault(edge, []).append((i, e))
    result = np.full(faces.shape, -1, dtype=int)
    for linked in incidence.values():
        if len(linked) == 2:
            (i, e), (j, g) = linked
            result[i, e], result[j, g] = j, i
    return result


def clip_front(polygon, near):
    clipped = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        if a[2] >= near:
            clipped.append(a)
        if (a[2] >= near) != (b[2] >= near):
            clipped.append(a + (b - a) * ((near - a[2]) / (b[2] - a[2])))
    return np.asarray(clipped)


def screen_edges(camera, pixels, triangles, near):
    """Distance to three original edges, with a clipped polygon at the near plane."""
    local = camera.local(triangles)
    regular = (local[..., 2] >= near).all(axis=1)
    safe_z = np.where(regular[:, None, None], local[..., 2:], 1.)
    xy = local[..., :2] / safe_z * camera.intrinsics[:2] + camera.intrinsics[2:]
    delta = np.roll(xy, -1, axis=1) - xy
    winding = np.sign(cross2(delta[:, 0], delta[:, 1]))
    distance = cross2(delta, pixels[:, None, None, :] - xy[None])
    distance *= winding[:, None] / np.maximum(np.linalg.norm(delta, axis=-1), 1e-10)
    for i in np.flatnonzero(~regular):
        poly = clip_front(local[i], near)
        if len(poly) < 3:
            distance[:, i] = -np.inf
            continue
        q = poly[:, :2] / poly[:, 2:] * camera.intrinsics[:2] + camera.intrinsics[2:]
        edge = np.roll(q, -1, axis=0) - q
        sign = np.sign(cross2(q, np.roll(q, -1, axis=0)).sum())
        d = cross2(edge, pixels[:, None] - q) * sign
        d /= np.maximum(np.linalg.norm(edge, axis=-1), 1e-10)
        distance[:, i] = d.min(axis=1, keepdims=True)
    return distance, regular


def plane_hits(vertices, faces, center, directions):
    """Metric range, world intersection and perspective-correct barycentrics."""
    a, b, c = np.moveaxis(vertices[faces], 1, 0)
    e, g = b - a, c - a
    n = np.cross(e, g)
    denom = (directions[:, None] * n).sum(axis=-1)
    valid = np.abs(denom) > 1e-12
    depth = ((a - center) * n).sum(axis=1) / np.where(valid, denom, 1.)
    depth = np.where(valid, depth, 0.)
    rel = center + depth[..., None] * directions[:, None] - a
    ee, eg, gg = (e * e).sum(-1), (e * g).sum(-1), (g * g).sum(-1)
    re, rg = (rel * e).sum(-1), (rel * g).sum(-1)
    det = np.maximum(ee * gg - eg * eg, 1e-20)
    beta1 = (gg * re - eg * rg) / det
    beta2 = (ee * rg - eg * re) / det
    beta = np.stack((1 - beta1 - beta2, beta1, beta2), axis=-1)
    return depth, beta


def clamp_to_triangle(beta, triangles):
    """Nearest point on a triangle for an outside plane hit."""
    a, b = triangles, np.roll(triangles, -1, axis=1)
    edge = b - a
    p = (beta[..., None] * triangles).sum(axis=-2)
    relative = p[..., None, :] - a
    t = np.clip((relative * edge).sum(-1) / np.maximum((edge * edge).sum(-1), 1e-16), 0, 1)
    residual = relative - t[..., None] * edge
    chosen = np.argmin((residual * residual).sum(-1), axis=-1)
    v = np.take_along_axis(t, chosen[..., None], axis=-1)
    eye = np.eye(3)
    border = (1 - v) * eye[chosen] + v * eye[(chosen + 1) % 3]
    return np.where((beta >= 0).all(axis=-1, keepdims=True), beta, border)


def atlas_basis(uv, shape):
    """Bilinear sampling in a fixed-size atlas, with flattened texel indices."""
    h, w = shape
    xy = np.clip(uv, 0, 1) * [w - 1, h - 1]
    x, y = np.moveaxis(np.floor(xy).astype(int), -1, 0)
    x1, y1 = np.minimum(x + 1, w - 1), np.minimum(y + 1, h - 1)
    dx, dy = np.moveaxis(xy - np.stack((x, y), axis=-1), -1, 0)
    ids = np.stack((y*w + x, y*w + x1, y1*w + x, y1*w + x1), axis=-1)
    weights = np.stack(((1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy), axis=-1)
    return ids, weights


def gaussian_basis(face_uv, shape):
    """Equal-byte fixed barycentric RBFs on a chart, with a positive base weight.

    Centers are the chart's lattice inverted into each face's barycentric plane.
    A chart sample therefore follows the face, including after a conforming split.
    The narrowest precomputed lattice neighborhood is used for this CPU variant.
    """
    h, w = shape
    xy = np.clip(face_uv, 0, 1) * [w - 1, h - 1]
    nearest = np.rint(xy).astype(int)
    offsets = np.array([(x, y) for y in (-1, 0, 1) for x in (-1, 0, 1)])
    coords = nearest[..., None, :] + offsets
    inside = ((coords >= 0) & (coords < [w, h])).all(-1)
    distance = ((xy[..., None, :] - coords)**2).sum(-1)
    weights = np.exp(-distance / (2 * .58**2)) * inside
    # Include one shared base RGB texel, with no extra trainable parameters.
    ids = np.clip(coords[..., 1], 0, h-1) * w + np.clip(coords[..., 0], 0, w-1)
    ids = np.concatenate((ids, np.zeros_like(ids[..., :1])), axis=-1)
    weights = np.concatenate((weights, np.full_like(weights[..., :1], .025)), axis=-1)
    return ids, weights / weights.sum(axis=-1, keepdims=True)


def render(model, camera, pixels, tau=.8, near=.05, color_basis=False):
    """Render continuous sample coordinates. No depth mixing across pixel samples."""
    render.subsamples += len(pixels)
    vertices, faces = model.vertices, model.faces
    if not len(faces):
        n=len(pixels)
        empty=np.empty((n,0))
        basis=(np.empty(0,int),np.empty((0,3),int),np.empty((0,3))) if color_basis else None
        return dict(rgb=np.broadcast_to(model.background,(n,3)).copy(),hit=np.full(n,-1,int),
                    hard_hit=np.full(n,-1,int),weight=empty,depth=empty,beta=np.empty((n,0,3)),
                    transmittance=np.ones(n),basis=basis)
    triangles = vertices[faces]
    directions = camera.rays(pixels)
    depth, beta = plane_hits(vertices, faces, camera.center, directions)
    distance, regular = screen_edges(camera, pixels, triangles, near)
    # Half-plane ownership for shared edges, with exact ties owned by face index.
    shared = (model.neighbors >= 0) & regular[:, None]
    tied = np.abs(distance) < 1e-8
    owns = (model.neighbors < 0) | (np.arange(len(faces))[:, None] < model.neighbors)
    inside_edge = (distance > 1e-8) | (tied & owns)
    inside = inside_edge.all(axis=-1)
    soft_gate = np.all(inside_edge | ~shared, axis=-1)
    if (~regular).any():
        # After clipping, screen distances describe the full clipped polygon.
        # The original shared-edge ownership is still encoded by barycentrics.
        clipped_beta = beta[:, ~regular][:, :, [2, 0, 1]]
        original_edge = ((clipped_beta > 1e-8) |
                         ((np.abs(clipped_beta) <= 1e-8) & owns[~regular]))
        clipped_gate = (original_edge | (model.neighbors[~regular] < 0)).all(axis=-1)
        inside[:, ~regular] = (distance[:, ~regular, 0] >= -1e-8) & clipped_gate
        soft_gate[:, ~regular] = clipped_gate
    front = (depth * (directions @ camera.rotation[2])[:, None] >= near)
    exterior = np.min(np.where(shared, np.inf, distance), axis=-1)
    smooth = np.clip((exterior + tau) / (2 * tau), 0, 1)
    smooth = smooth**2 * (3 - 2 * smooth)
    soft = model.opacity * smooth * soft_gate
    alpha = np.where(model.hard, inside, soft) * front
    order = np.argsort(np.where(alpha > 0, depth, np.inf), axis=1, kind="stable")
    rank = np.take_along_axis(alpha, order, axis=1)
    transmission = np.cumprod(1 - rank, axis=1)
    before = np.concatenate((np.ones_like(rank[:, :1]), transmission[:, :-1]), axis=1)
    ordered_weight = before * rank
    weight = np.empty_like(ordered_weight)
    np.put_along_axis(weight, order, ordered_weight, axis=1)
    rows, indices = np.nonzero(weight)
    inside_beta = clamp_to_triangle(beta[rows, indices], triangles[indices])
    if model.vertex_color:
        ids, basis = faces[indices], inside_beta
    else:
        uv = (inside_beta[..., None] * model.uv[indices]).sum(axis=1)
        sampler = atlas_basis if model.shader == "atlas" else gaussian_basis
        ids, basis = sampler(uv, model.colors.shape[:2])
    rgb = transmission[:, -1:] * model.background
    coefficients = weight[rows, indices, None] * basis
    contribution = (coefficients[..., None] * model.colors.reshape(-1, 3)[ids]).sum(axis=1)
    np.add.at(rgb, rows, contribution)
    first_slot = np.argmax(rank > 0, axis=1)
    hit = order[np.arange(len(pixels)), first_slot]
    no_hit = ~np.any(rank > 0, axis=1)
    hit = np.where(no_hit, -1, hit)
    opaque = (rank == 1) & model.hard[order] & (before > 0)
    hard_slot = np.argmax(opaque, axis=1)
    hard_hit = np.where(opaque.any(axis=1), order[np.arange(len(pixels)), hard_slot], -1)
    return dict(rgb=rgb, hit=hit, hard_hit=hard_hit, weight=weight, depth=depth, beta=beta,
                transmittance=transmission[:, -1],
                basis=(rows, ids, coefficients) if color_basis else None)


render.subsamples = 0
