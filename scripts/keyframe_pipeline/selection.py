"""Quality-aware, capped LiDAR coverage selection with trajectory safeguards."""

import argparse
import heapq
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import percentile_filter

from .common import (atomic_json, digest, file_stamp, image_path, load_cloud,
                     prepare_output, read_gray, read_model, visible_points)


def choose_frames(observations, quality, target, seeds, n_patches, views=2):
    """Lazy greedy maximization of sum_p min(views, sum_b best_quality[p,b]).

    Each observation has unique surface patch IDs and one view bin per patch.
    A repeat view only helps if it improves that patch's observation quality.
    """
    best = np.zeros((n_patches, 16), dtype=np.float32)
    totals = np.zeros(n_patches, dtype=np.float32)
    selected, chosen = [], set()

    def marginal(i, update=False):
        patch, bins, weights = observations[i]
        weights = weights * quality[i]
        delta = np.maximum(0, weights - best[patch, bins])
        gain = np.minimum(views, totals[patch] + delta) - np.minimum(views, totals[patch])
        if update:
            best[patch, bins] = np.maximum(best[patch, bins], weights)
            totals[patch] += delta
            chosen.add(i)
            selected.append(i)
        return float(gain.sum())

    for i in sorted(set(seeds)):
        marginal(i, True)
    heap = [(-marginal(i), i, len(selected)) for i in range(len(observations)) if i not in chosen]
    heapq.heapify(heap)
    while len(selected) < target and heap:
        neg_gain, i, step = heapq.heappop(heap)
        if step == len(selected):
            if -neg_gain <= 1e-8:
                break
            marginal(i, True)
        else:
            heapq.heappush(heap, (-marginal(i), i, len(selected)))
    return sorted(selected), totals


def observations_for_view(points, patch_ids, image, camera, patch_size):
    scale = min(1., 640 / max(camera.width, camera.height))
    h, w = round(camera.height * scale), round(camera.width * scale)
    K = camera.params * np.array([w / camera.width, h / camera.height] * 2)
    ids, _, z = visible_points(points, image, K, (h, w))
    if not len(ids):
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32)
    # Prefer the nearest visible representative in each patch.
    order = np.argsort(z, kind="stable")
    _, first = np.unique(patch_ids[ids[order]], return_index=True)
    ids, z = ids[order[first]], z[order[first]]
    rays = image.C - points[ids]
    azimuth = np.floor((np.arctan2(rays[:, 1], rays[:, 0]) + np.pi) / (2 * np.pi) * 8).astype(int) % 8
    elevation = (np.abs(rays[:, 2]) > np.linalg.norm(rays[:, :2], axis=1)).astype(int)
    bins = azimuth + 8 * elevation
    # Saturating projected patch size: very distant views contribute less.
    weights = np.clip(patch_size * min(K[:2]) / z / 30., .1, 1.).astype(np.float32)
    return patch_ids[ids].astype(np.int32), bins.astype(np.int32), weights


def select(args):
    source = args.source.expanduser().resolve()
    model, cameras, all_images, stamps = read_model(source)
    cloud_path = (args.point_cloud or model / "points3D.ply").expanduser().resolve()
    reserved = set(sorted(im.name for im in all_images)[::8]) if args.eval else set()
    if args.exclude_list:
        excluded = {line.strip() for line in args.exclude_list.read_text().splitlines() if line.strip()}
        unknown = excluded - {im.name for im in all_images}
        if unknown:
            raise ValueError(f"Unknown excluded images: {sorted(unknown)[:5]}")
        reserved |= excluded
    images = [im for im in all_images if im.name not in reserved]
    if not images:
        raise ValueError("No candidates after exclusions")
    settings = dict(version=1, source=str(source), images=args.images, model=stamps,
                    cloud=file_stamp(cloud_path), target=args.target_frames,
                    voxel_size=args.voxel_size, max_points=args.max_points,
                    patch_size=args.patch_size, max_gap=args.max_gap, eval=args.eval,
                    reserved=sorted(reserved), min_relative_sharpness=args.min_relative_sharpness,
                    image_files_sha256=digest([file_stamp(image_path(source, args.images, im.name)) for im in images]))
    output = prepare_output(args.output, settings)
    if (output / "keyframes.json").exists():
        print(f"Reusing {output / 'keyframes.json'}", flush=True)
        return output / "keyframes.json"
    points, voxel = load_cloud(cloud_path, args.voxel_size, args.max_points)
    _, patch_ids = np.unique(np.floor(points / args.patch_size).astype(np.int64), axis=0, return_inverse=True)
    cache = output / "coverage_cache"
    cache.mkdir(exist_ok=True)
    observations, sharpness, unclipped = [], [], []
    for i, im in enumerate(images):
        path = cache / f"{im.id}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                obs = tuple(saved[k] for k in ("patch", "bins", "weights"))
                sharp, good = float(saved["sharpness"]), float(saved["unclipped"])
        else:
            gray, _, _ = read_gray(source, args.images, im, cameras[im.camera_id], 960)
            sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            good = float(np.mean((gray > 5) & (gray < 250)))
            obs = observations_for_view(points, patch_ids, im, cameras[im.camera_id], args.patch_size)
            np.savez_compressed(path.with_suffix(".tmp.npz"), patch=obs[0], bins=obs[1], weights=obs[2], sharpness=sharp, unclipped=good)
            path.with_suffix(".tmp.npz").replace(path)
        observations.append(obs)
        sharpness.append(sharp)
        unclipped.append(good)
        if i % 100 == 0 or i == len(images) - 1:
            print(f"Scored {i + 1}/{len(images)} images", flush=True)
    sharpness, unclipped = np.asarray(sharpness), np.asarray(unclipped)
    quality = np.zeros(len(images))
    relative_sharpness = np.zeros(len(images))
    streams = {}
    for i, im in enumerate(images):
        streams.setdefault(str(Path(im.name).parent), []).append(i)
    seeds = []
    for stream in streams.values():
        local = sharpness[stream]
        reference = percentile_filter(local, 90, size=min(25, len(local)), mode="nearest")
        relative = np.minimum(1., local / np.maximum(reference, 1e-8))
        relative_sharpness[stream] = relative
        # Compare nearby views, not the raw texture strength of different rooms.
        quality[stream] = np.sqrt(relative) * np.maximum(.1, unclipped[stream])
        for start in range(0, len(stream), args.max_gap):
            group = stream[start:start + args.max_gap]
            seeds.append(max(group, key=lambda i: quality[i] * np.sqrt(max(1, len(observations[i][0])))))
        # Preserve ends of each named camera stream.
        for group in (stream[:min(3, len(stream))], stream[-min(3, len(stream)):]):
            seeds.append(max(group, key=lambda i: quality[i]))
    if not any(len(obs[0]) for obs in observations):
        raise ValueError("No LiDAR points project into any candidate image; check the map frame and intrinsics")
    eligible = np.flatnonzero(relative_sharpness >= args.min_relative_sharpness)
    eligible = np.union1d(eligible, seeds)
    target = min(args.target_frames, len(eligible))
    if len(set(seeds)) > target:
        raise ValueError(f"{len(set(seeds))} trajectory seeds exceed budget {target}; increase --target-frames or --max-gap")
    remap = {int(old): new for new, old in enumerate(eligible)}
    selected, coverage = choose_frames([observations[i] for i in eligible], quality[eligible], target,
                                      [remap[i] for i in seeds], int(patch_ids.max()) + 1)
    selected = [int(eligible[i]) for i in selected]
    selected_set = set(selected)
    names = [images[i].name for i in selected]
    manifest = dict(version=2, method="quality_lidar_coverage", source_path=str(source),
                    image_directory=args.images, eval=args.eval, selected_count=len(names),
                    selected_images=names, excluded_images=sorted(reserved),
                    registered_count=len(all_images), candidate_count=len(images),
                    registered_names_sha256=digest(sorted(im.name for im in all_images)),
                    settings_sha256=digest(settings), actual_voxel_size=voxel,
                    trajectory_seed_count=len(set(seeds)),
                    lidar_patch_count=len(coverage), observed_patch_count=int(np.sum(coverage > 0)),
                    frames=[dict(name=im.name, selected=i in selected_set, sharpness=float(sharpness[i]),
                                 quality=float(quality[i]), visible_patches=len(observations[i][0])) for i, im in enumerate(images)],
                    notes=["Natural filename order within image parent directories defines temporal streams.",
                           "LiDAR visibility uses an approximate point z-buffer, not a watertight surface.",
                           "All distances use the input map units (expected metres)."])
    (output / "image_list.txt").write_text("".join(name + "\n" for name in names))
    atomic_json(output / "keyframes.json", manifest)
    print(f"Selected {len(names)}/{len(all_images)}; wrote {output / 'keyframes.json'}", flush=True)
    return output / "keyframes.json"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--images", default="images")
    p.add_argument("--point-cloud", type=Path, help="LiDAR PLY in the COLMAP world frame; default sparse model/points3D.ply")
    p.add_argument("--target-frames", type=int, default=600)
    p.add_argument("--max-gap", type=int, default=30, help="Seed each block of N candidate frames, per parent directory")
    p.add_argument("--voxel-size", type=float, default=.08)
    p.add_argument("--max-points", type=int, default=300_000)
    p.add_argument("--patch-size", type=float, default=.4)
    p.add_argument("--min-relative-sharpness", type=float, default=.25)
    p.add_argument("--eval", action="store_true", help="Exclude original lexical 1-in-8 test images before selection")
    p.add_argument("--exclude-list", type=Path, help="Additional held-out image names, one per line")
    args = p.parse_args(argv)
    if min(args.target_frames, args.max_gap) < 1 or args.max_points < 20:
        p.error("Positive target/gap and at least 20 cloud points required")
    if not np.isfinite([args.voxel_size, args.patch_size, args.min_relative_sharpness]).all() or not (0 < args.voxel_size <= args.patch_size) or not 0 < args.min_relative_sharpness <= 1:
        p.error("Require 0 < voxel-size <= patch-size and relative sharpness in (0,1]")
    try:
        return select(args)
    except (ValueError, FileNotFoundError) as exc:
        p.exit(2, f"Selection failed: {exc}\n")
