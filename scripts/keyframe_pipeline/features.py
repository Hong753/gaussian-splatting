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
    empty = dict(uv=np.empty((0, 2)), descriptors=np.empty((0, 128), np.float32),
                 xyz=np.empty((0, 3)), normals=np.empty((0, 3)), K=K, scale=scale)
    if descriptors is None:
        return empty
    # SIFT uses integer-centered OpenCV pixels; COLMAP centers are offset by .5.
    uv = np.asarray([kp.pt for kp in keypoints]) + .5
    visible, pixels, _ = visible_points(points, image, K, gray.shape)
    # Occlusion is tested with ALL points, including nonplanar occluders.
    usable = planar[visible]
    visible, pixels = visible[usable], pixels[usable]
    if not len(visible):
        return empty
    distance, nn = cKDTree(pixels).query(uv)
    anchors, normal = points[visible[nn]], normals[visible[nn]]
    rays = np.column_stack(((uv - K[2:]) / K[:2], np.ones(len(uv)))) @ image.R
    denom = np.einsum("ni,ni->n", rays, normal)
    depth = np.einsum("ni,ni->n", anchors - image.C, normal) / np.where(np.abs(denom) > 1e-8, denom, 1e-8)
    xyz = image.C + depth[:, None] * rays
    keep = ((distance <= args.anchor_pixels) & (depth > .2) &
            (np.abs(denom) / np.linalg.norm(rays, axis=1) > .15) &
            (np.linalg.norm(xyz - anchors, axis=1) < args.surface_radius))
    descriptors = descriptors[keep]
    descriptors = np.sqrt(descriptors / np.maximum(descriptors.sum(axis=1, keepdims=True), 1e-8)).astype(np.float32)
    return dict(uv=uv[keep], descriptors=descriptors, xyz=xyz[keep], normals=normal[keep], K=K, scale=scale)


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


def match_pair(a, b, image_a, image_b, args):
    if min(len(a["uv"]), len(b["uv"])) < 2:
        return np.empty((0, 2), dtype=int)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))

    def ratio_matches(x, y):
        return {m.queryIdx: m.trainIdx for pair in matcher.knnMatch(x, y, k=2)
                if len(pair) == 2 for m, n in [pair] if m.distance < args.ratio * n.distance}

    ab = ratio_matches(a["descriptors"], b["descriptors"])
    ba = ratio_matches(b["descriptors"], a["descriptors"])
    matches = np.array([(i, j) for i, j in ab.items() if ba.get(j) == i], dtype=int).reshape(-1, 2)
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


def build_tracks(features, pair_matches, max_tracks=8000):
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
    for root_key, observations in members.items():
        if root(root_key) != root_key or len(observations) < 3:
            continue
        obs = sorted(observations.items())
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
