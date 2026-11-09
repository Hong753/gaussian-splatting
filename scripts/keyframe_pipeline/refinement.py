"""Refine selected COLMAP poses against image tracks and a fixed LiDAR map."""

import argparse
import json
from pathlib import Path
import struct

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .bundle import solve_bundle
from .common import (atomic_json, digest, file_stamp, image_path, load_cloud,
                     prepare_output, project, read_model)
from .features import build_tracks, candidate_pairs, extract_features, match_pair, surface_normals


def export_dataset(output, source, folder, cloud, cameras, all_images, selected,
                   features, tracks, R, C, X, manifest):
    """Publish atomically after validation; source data are never overwritten.

    All cameras are exported to preserve the original evaluation split. Only
    selected camera poses change. Feature tracks here are newly constructed;
    the initialization PLY remains the original LiDAR map.
    """
    output = Path(output)
    destination = output / "dataset"
    if destination.exists():
        raise ValueError(f"Refusing to overwrite {destination}")
    stage = output / "dataset.tmp"
    if stage.exists():
        raise ValueError(f"Incomplete export exists at {stage}; inspect/remove it before retrying")
    model = stage / "sparse/0"
    model.mkdir(parents=True)
    (stage / "images").symlink_to((source / folder).resolve(), target_is_directory=True)
    (model / "points3D.ply").symlink_to(cloud.resolve())
    selected_by_id = {im.id: i for i, im in enumerate(selected)}
    image_observations = {im.id: [] for im in all_images}
    point_tracks = []
    for k, track in enumerate(tracks):
        associations = []
        errors = []
        for i, feature in track["obs"]:
            im = selected[i]
            uv = features[i]["uv"][feature] / features[i]["scale"]
            index = len(image_observations[im.id])
            image_observations[im.id].append((*uv, k + 1))
            associations.append((im.id, index))
            predicted, _ = project(X[k:k + 1], R[i], C[i], cameras[im.camera_id].params)
            errors.append(float(np.linalg.norm(predicted[0] - uv)))
        point_tracks.append((associations, float(np.mean(errors))))
    with (model / "cameras.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cid, cam in sorted(cameras.items()):
            # SIMPLE_PINHOLE -> equivalent PINHOLE, supported by all repo readers.
            f.write(struct.pack("<iiQQ4d", cid, 1, cam.width, cam.height, *cam.params))
    with (model / "images.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(all_images)))
        for im in all_images:
            index = selected_by_id.get(im.id)
            rotation, center = (R[index], C[index]) if index is not None else (im.R, im.C)
            q = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
            f.write(struct.pack("<i7di", im.id, *q, *(-rotation @ center), im.camera_id))
            f.write(im.name.encode("utf8") + b"\0")
            observations = image_observations[im.id]
            f.write(struct.pack("<Q", len(observations)))
            for obs in observations:
                f.write(struct.pack("<ddq", *obs))
    with (model / "points3D.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(X)))
        for k, xyz in enumerate(X):
            associations, error = point_tracks[k]
            f.write(struct.pack("<QdddBBBdQ", k + 1, *xyz, 128, 128, 128, error, len(associations)))
            for association in associations:
                f.write(struct.pack("<ii", *association))
    updated_manifest = {**manifest, "source_path": str(destination.resolve()), "image_directory": "images",
                        "pose_refinement": str((output / "refinement_report.json").resolve())}
    atomic_json(stage / "keyframes.json", updated_manifest)
    (stage / "image_list.txt").write_text("".join(im.name + "\n" for im in selected))
    stage.rename(destination)
    return destination


def refine(args):
    cv2.setRNGSeed(0)
    source = args.source.expanduser().resolve()
    model, cameras, all_images, stamps = read_model(source)
    cloud = (args.point_cloud or model / "points3D.ply").expanduser().resolve()
    manifest = json.loads(args.keyframes.read_text())
    names = manifest["selected_images"]
    if not names or len(names) != len(set(names)):
        raise ValueError("Empty or duplicate keyframe names")
    unknown = set(names) - {im.name for im in all_images}
    if unknown:
        raise ValueError(f"Keyframes absent from the model: {sorted(unknown)[:5]}")
    if set(names) & set(manifest.get("excluded_images", [])):
        raise ValueError("Keyframes overlap the manifest's held-out images")
    selected = [im for im in all_images if im.name in set(names)]
    if len(selected) < 3:
        raise ValueError("At least three selected images are required")
    settings = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "output"}
    settings.update(version=1, model=stamps, cloud=file_stamp(cloud), manifest_sha256=digest(manifest),
                    image_files_sha256=digest([file_stamp(image_path(source, args.images, im.name)) for im in selected]))
    output = prepare_output(args.output.expanduser().resolve(), settings)
    report_file = output / "refinement_report.json"
    if (output / "dataset").exists():
        if report_file.exists() and json.loads(report_file.read_text()).get("accepted"):
            print(f"Reusing validated dataset: {output / 'dataset'}", flush=True)
            return output / "dataset"
        raise ValueError("Dataset exists without a successful refinement report")
    points, voxel = load_cloud(cloud, args.voxel_size, args.max_points)
    radius = max(args.surface_radius, voxel * 4)
    print(f"LiDAR: {len(points)} samples, voxel {voxel:.4g}; estimating local planes", flush=True)
    plane_cache = output / "planes.npz"
    if plane_cache.exists():
        with np.load(plane_cache, allow_pickle=False) as saved:
            normals, planar = saved["normals"], saved["planar"]
    else:
        normals, planar = surface_normals(points, radius)
        np.savez_compressed(plane_cache.with_suffix(".tmp.npz"), normals=normals, planar=planar)
        plane_cache.with_suffix(".tmp.npz").replace(plane_cache)
    cache = output / "features"
    cache.mkdir(exist_ok=True)
    features = []
    for i, im in enumerate(selected):
        path = cache / f"{im.id}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                features.append(dict(saved))
        else:
            data = extract_features(source, args.images, im, cameras[im.camera_id], points, normals, planar, args)
            features.append(data)
            np.savez_compressed(path.with_suffix(".tmp.npz"), **data)
            path.with_suffix(".tmp.npz").replace(path)
        if i % 25 == 0 or i == len(selected) - 1:
            print(f"Features {i + 1}/{len(selected)}; last frame {len(features[-1]['uv'])} surface-associated", flush=True)
    pairs = candidate_pairs(selected, args.spatial_neighbors, args.temporal_neighbors)
    pair_cache = output / "matches"
    pair_cache.mkdir(exist_ok=True)
    matched_pairs, match_count = [], 0
    for k, (i, j) in enumerate(pairs):
        path = pair_cache / f"{selected[i].id}_{selected[j].id}.npy"
        if path.exists():
            matches = np.load(path, allow_pickle=False)
        else:
            # Seed per pair so cache hits do not change later FLANN searches.
            cv2.setRNGSeed((selected[i].id * 100003 + selected[j].id) % (2**31 - 1))
            matches = match_pair(features[i], features[j], selected[i], selected[j], args)
            np.save(path.with_suffix(".tmp.npy"), matches)
            path.with_suffix(".tmp.npy").replace(path)
        if len(matches) >= args.min_pair_matches:
            matched_pairs.append((i, j, matches))
            match_count += len(matches)
        if k % 100 == 0 or k == len(pairs) - 1:
            print(f"Matched {k + 1}/{len(pairs)} pairs; {match_count} gated correspondences", flush=True)
    tracks = build_tracks(features, matched_pairs, args.max_tracks)
    diagnostics = dict(candidate_pairs=len(pairs), usable_pairs=len(matched_pairs), matches=match_count,
                       tracks=len(tracks), planar_lidar_samples=int(planar.sum()), actual_voxel_size=voxel,
                       features_per_image={im.name: len(f["uv"]) for im, f in zip(selected, features)})
    atomic_json(output / "matching_report.json", diagnostics)
    if len(tracks) < args.min_tracks:
        atomic_json(report_file, dict(accepted=False, rejection_reasons=["Insufficient three-view feature tracks"], **diagnostics))
        raise ValueError(f"Only {len(tracks)} three-view tracks (need {args.min_tracks}); inspect matching_report.json. No poses exported.")
    try:
        R, C, X, report = solve_bundle(selected, features, tracks, args)
    except ValueError as exc:
        atomic_json(report_file, dict(accepted=False, rejection_reasons=[str(exc)], matching=diagnostics))
        raise
    report.update(source=str(source), keyframes=str(args.keyframes.resolve()), matching=diagnostics)
    atomic_json(report_file, report)
    print(f"Held-out median: {report['validation_before']['median']:.3f} -> {report['validation_after']['median']:.3f} working-image px", flush=True)
    if not report["accepted"]:
        raise ValueError("Refinement rejected: " + "; ".join(report["rejection_reasons"]) + ". Original poses untouched; see refinement_report.json.")
    dataset = export_dataset(output, source, args.images, cloud, cameras, all_images,
                             selected, features, tracks, R, C, X, manifest)
    print(f"Validated COLMAP dataset: {dataset}\nTrain using {dataset / 'keyframes.json'}", flush=True)
    return dataset


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--keyframes", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--images", default="images")
    p.add_argument("--point-cloud", type=Path)
    p.add_argument("--voxel-size", type=float, default=.05)
    p.add_argument("--max-points", type=int, default=500_000)
    p.add_argument("--surface-radius", type=float, default=.3)
    p.add_argument("--feature-max-side", type=int, default=1600)
    p.add_argument("--max-features", type=int, default=3000)
    p.add_argument("--anchor-pixels", type=float, default=5.)
    p.add_argument("--ratio", type=float, default=.75)
    p.add_argument("--match-pixels", type=float, default=12.)
    p.add_argument("--match-distance", type=float, default=.15)
    p.add_argument("--spatial-neighbors", type=int, default=6)
    p.add_argument("--temporal-neighbors", type=int, default=2)
    p.add_argument("--min-pair-matches", type=int, default=15)
    p.add_argument("--max-tracks", type=int, default=8000)
    p.add_argument("--min-tracks", type=int, default=50)
    p.add_argument("--min-observations", type=int, default=25)
    p.add_argument("--pixel-sigma", type=float, default=1.5)
    p.add_argument("--surface-sigma", type=float, default=.015)
    p.add_argument("--landmark-sigma", type=float, default=.10)
    p.add_argument("--translation-prior", type=float, default=.03, help="Camera-center prior std in input map units (expected metres)")
    p.add_argument("--rotation-prior-deg", type=float, default=.2)
    p.add_argument("--max-translation", type=float, default=.10)
    p.add_argument("--max-rotation-deg", type=float, default=1.)
    p.add_argument("--max-landmark-shift", type=float, default=.15)
    p.add_argument("--baseline-steps", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--min-improvement-pixels", type=float, default=.01)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    numeric = [v for v in vars(args).values() if isinstance(v, (int, float))]
    if any(v <= 0 or not np.isfinite(v) for v in numeric) or not 0 < args.ratio < 1:
        p.error("Numeric parameters must be positive and ratio must be in (0,1)")
    if args.max_tracks < args.min_tracks or args.max_points < 20:
        p.error("Require max-tracks >= min-tracks and max-points >= 20")
    try:
        return refine(args)
    except (ValueError, FileNotFoundError) as exc:
        p.exit(2, f"Refinement failed: {exc}\n")
