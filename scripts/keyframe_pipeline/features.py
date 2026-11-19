"""SIFT tracks with conservative associations to visible LiDAR surface patches."""

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .common import project, read_gray, visible_points


def surface_normals(points, radius):
    tree = cKDTree(points)
    normals = np.zeros_like(points)
    valid = np.zeros(len(points), dtype=bool)
    for start in range(0, len(points), 4096):
        query = points[start:start + 4096]
        distance, neighbors = tree.query(query, k=min(20, len(points)), workers=-1)
        local = points[neighbors]
        centered = local - local.mean(axis=1, keepdims=True)
        values, vectors = np.linalg.eigh(np.einsum("nki,nkj->nij", centered, centered))
        normals[start:start + len(query)] = vectors[:, :, 0]
        valid[start:start + len(query)] = ((values[:, 0] < .03 * np.maximum(values.sum(axis=1), 1e-12)) &
                                          (values[:, 1] > .1 * values[:, 2]) & (distance[:, -1] < radius))
    return normals, valid


def extract_features(source, folder, image, camera, points, normals, planar, args):
    gray, K, scale = read_gray(source, folder, image, camera, args.feature_max_side)
    sift = cv2.SIFT_create(nfeatures=args.max_features)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    uv = np.array([kp.pt for kp in keypoints], dtype=float).reshape(-1, 2) + .5
    if descriptors is None:
        descriptors = np.empty((0, 128), dtype=np.float32)
    # Missing geometry is explicit. Image-first matching also keeps these pixels.
    xyz = np.full((len(uv), 3), np.nan)
    normal = np.full_like(xyz, np.nan)
    visible, pixels, _ = visible_points(points, image, K, gray.shape)
    # Test occlusions using the entire sampled cloud before selecting planes.
    usable = planar[visible]
    visible, pixels = visible[usable], pixels[usable]
    if len(visible) and len(uv):
        distance, nn = cKDTree(pixels).query(uv)
        anchors, candidate_normals = points[visible[nn]], normals[visible[nn]]
        rays = np.column_stack(((uv - K[2:]) / K[:2], np.ones(len(uv)))) @ image.R
        denom = np.einsum("ni,ni->n", rays, candidate_normals)
        depth = np.einsum("ni,ni->n", anchors - image.C, candidate_normals) / np.where(np.abs(denom) > 1e-8, denom, 1e-8)
        intersections = image.C + depth[:, None] * rays
        keep = ((distance <= args.anchor_pixels) & (depth > .2) &
                (np.abs(denom) / np.linalg.norm(rays, axis=1) > .15) &
                (np.linalg.norm(intersections - anchors, axis=1) < args.surface_radius))
        xyz[keep], normal[keep] = intersections[keep], candidate_normals[keep]
    if not args.image_first:
        keep = np.isfinite(xyz[:, 0])
        uv, descriptors, xyz, normal = uv[keep], descriptors[keep], xyz[keep], normal[keep]
    descriptors = np.sqrt(descriptors / np.maximum(descriptors.sum(axis=1, keepdims=True), 1e-8)).astype(np.float32)
    return dict(uv=uv, descriptors=descriptors, xyz=xyz, normals=normal, K=K, scale=scale)


def candidate_pairs(images, spatial_neighbors=6, temporal_neighbors=2):
    centers = np.array([im.C for im in images])
    forward = np.array([im.R[2] for im in images])
    _, nearest = cKDTree(centers).query(centers, k=min(len(images), spatial_neighbors * 4 + 1))
    nearest = np.asarray(nearest).reshape(len(images), -1)
    pairs, streams = set(), defaultdict(list)
    for i, im in enumerate(images):
        streams[str(Path(im.name).parent)].append(i)
        count = 0
        for j in nearest[i]:
            if i == j or forward[i] @ forward[j] < .3:
                continue
            pairs.add(tuple(sorted((i, int(j)))))
            count += 1
            if count >= spatial_neighbors:
                break
    for stream in streams.values():
        for k, i in enumerate(stream):
            for j in stream[k + 1:k + 1 + temporal_neighbors]:
                pairs.add((i, j))
    return sorted(pairs)


def match_pair(a, b, image_a, image_b, args, stats=None):
    if stats is not None:
        stats["mutual"] = 0
    if min(len(a["uv"]), len(b["uv"])) < 2:
        return np.empty((0, 2), dtype=int)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))

    def ratio_matches(x, y):
        return {m.queryIdx: m.trainIdx for pair in matcher.knnMatch(x, y, k=2)
                if len(pair) == 2 for m, n in [pair] if m.distance < args.ratio * n.distance}

    ab = ratio_matches(a["descriptors"], b["descriptors"])
    ba = ratio_matches(b["descriptors"], a["descriptors"])
    matches = np.array([(i, j) for i, j in ab.items() if ba.get(j) == i], dtype=int).reshape(-1, 2)
    if stats is not None:
        stats["mutual"] = len(matches)
    if args.image_first:
        if len(matches) < 8:
            return matches[:0]
        src, dst = a["uv"][matches[:, 0]], b["uv"][matches[:, 1]]
        # A homography also covers nearly pure rotation and planar overlap.
        # Either model only verifies 2D matches; neither supplies BA depth.
        F, fm = cv2.findFundamentalMat(src, dst, cv2.FM_RANSAC, 2., .995)
        H, hm = cv2.findHomography(src, dst, cv2.RANSAC, 2.)
        masks = [mask.ravel().astype(bool) for matrix, mask in ((F, fm), (H, hm))
                 if matrix is not None and mask is not None and np.isfinite(matrix).all()]
        return matches[max(masks, key=np.count_nonzero)] if masks else matches[:0]
    if not len(matches):
        return matches
    i, j = matches.T
    pred_b, zb = project(a["xyz"][i], image_b.R, image_b.C, b["K"])
    pred_a, za = project(b["xyz"][j], image_a.R, image_a.C, a["K"])
    keep = ((za > .2) & (zb > .2) &
            (np.linalg.norm(pred_b - b["uv"][j], axis=1) < args.match_pixels) &
            (np.linalg.norm(pred_a - a["uv"][i], axis=1) < args.match_pixels) &
            (np.linalg.norm(a["xyz"][i] - b["xyz"][j], axis=1) < args.match_distance) &
            (np.abs(np.sum(a["normals"][i] * b["normals"][j], axis=1)) > .85))
    return matches[keep]


def anchored_track(obs, features, rotations, centers, intrinsics, args):
    """Hold out one pixel and require a LiDAR anchor consistent with the fit observations."""
    anchors = [j for j, (i, k) in enumerate(obs) if np.isfinite(features[i]["xyz"][k, 0])]
    if not anchors:
        return None
    # Keep an anchor even when a long track is subsampled.
    if len(obs) > 10:
        keep = np.linspace(0, len(obs) - 1, 10).round().astype(int)
        if not set(anchors).intersection(keep):
            nearest = np.argmin(abs(keep - anchors[0]))
            keep[nearest] = anchors[0]
        obs = [obs[j] for j in keep]
        anchors = [j for j, (i, k) in enumerate(obs) if np.isfinite(features[i]["xyz"][k, 0])]
    candidates = [j for j in range(len(obs)) if len(anchors) > 1 or j != anchors[0]]
    reserve = candidates[(obs[0][1] + 31 * obs[0][0]) % len(candidates)]
    fit = [o for j, o in enumerate(obs) if j != reserve]
    fit_cameras = [i for i, _ in fit]
    uv = np.array([features[i]["uv"][k] for i, k in fit])
    # Choose an observed plane inside the fixed map. A wrong-depth match must
    # agree in every fit view, including those without a nearby LiDAR return.
    best = None
    best_error = np.inf
    support = [obs[j] for j in anchors if j != reserve]
    points = np.array([features[i]["xyz"][k] for i, k in support])
    normals = np.array([features[i]["normals"][k] for i, k in support])
    for (i, k), xyz, normal in zip(support, points, normals):
        if len(support) > 1 and ((np.linalg.norm(points - xyz, axis=1) > args.match_distance).any() or
                                 (np.abs(normals @ normal) < .85).any()):
            continue
        local = np.einsum("nij,nj->ni", rotations[fit_cameras], xyz - centers[fit_cameras])
        if (local[:, 2] <= .2).any():
            continue
        K = intrinsics[fit_cameras]
        err = np.linalg.norm(local[:, :2] / local[:, 2:] * K[:, :2] + K[:, 2:] - uv, axis=1)
        if err.max() <= args.match_pixels and err.mean() < best_error:
            best, best_error = (i, k), err.mean()
    if best is None:
        return None
    return dict(obs=obs, reserve=reserve, lidar_observation=best)


def build_tracks(features, pair_matches, max_tracks=8000, *, images=None, args=None, stats=None):
    """Merge pair matches, refusing tracks with two features in one image."""
    parent, members = {}, {}

    def root(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = root(parent[x])
        return parent[x]

    for i, j, matches in pair_matches:
        for a, b in matches:
            x, y = root((i, int(a))), root((j, int(b)))
            if x == y:
                continue
            mx, my = members.get(x, {x[0]: x[1]}), members.get(y, {y[0]: y[1]})
            if mx.keys() & my.keys():
                continue
            if len(mx) < len(my):
                x, y, mx, my = y, x, my, mx
            parent[y] = x
            members[x] = {**mx, **my}
            members.pop(y, None)
    tracks = []
    n_multiview = n_anchored = 0
    if images is not None:
        rotations = np.stack([im.R for im in images])
        centers = np.stack([im.C for im in images])
        intrinsics = np.stack([f["K"] for f in features])
    for root_key, observations in members.items():
        if root(root_key) != root_key or len(observations) < 3:
            continue
        obs = sorted(observations.items())
        n_multiview += 1
        if images is not None:
            if not any(np.isfinite(features[i]["xyz"][k, 0]) for i, k in obs):
                continue
            n_anchored += 1
            track = anchored_track(obs, features, rotations, centers, intrinsics, args)
            if track is not None:
                tracks.append(track)
            continue
        # Evenly retain at most ten observations, including trajectory ends.
        if len(obs) > 10:
            obs = [obs[i] for i in np.linspace(0, len(obs) - 1, 10).round().astype(int)]
        xyz = np.array([features[i]["xyz"][k] for i, k in obs])
        center = np.median(xyz, axis=0)
        anchor_index = int(np.argmin(np.linalg.norm(xyz - center, axis=1)))
        i, k = obs[anchor_index]
        normal = features[i]["normals"][k]
        anchor = xyz[anchor_index]
        # Start on the fixed local LiDAR plane; tangential position can adjust.
        center -= normal * ((center - anchor) @ normal)
        tracks.append(dict(obs=obs, xyz=center, anchor=anchor, normal=normal))
    if stats is not None:
        stats.update(three_view_image_tracks=n_multiview, with_lidar_anchor=n_anchored,
                     fit_geometry_consistent=len(tracks))
    # Round-robin by least represented camera avoids filling the budget in one room.
    tracks.sort(key=lambda t: (-len(t["obs"]), t["obs"]))
    by_camera = defaultdict(list)
    for idx, track in enumerate(tracks):
        for i, _ in track["obs"]:
            by_camera[i].append(idx)
    chosen, chosen_set = [], set()
    for round_index in range(max((len(v) for v in by_camera.values()), default=0)):
        for i in sorted(by_camera):
            ids = by_camera[i]
            if round_index < len(ids) and ids[round_index] not in chosen_set:
                chosen.append(ids[round_index])
                chosen_set.add(ids[round_index])
                if len(chosen) == max_tracks:
                    return [tracks[k] for k in chosen]
    return [tracks[k] for k in chosen]
