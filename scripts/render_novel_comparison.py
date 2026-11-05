#!/usr/bin/env python3
"""Render two models on an automatically synthesized smooth novel-view path.

The path is derived from COLMAP poses with smooth seeded random deviations.
Camera-center bounds and distance to captured cameras are enforced, but this
is not collision checking against reconstructed surfaces.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import render_model_comparison as core


def rot_to_quat(r):
    """Rotation matrix to normalized [w,x,y,z] quaternion."""
    r = np.asarray(r, dtype=float)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError("Invalid camera rotation")
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-3) or np.linalg.det(r) < 0.99:
        raise ValueError("Camera rotation is not a proper rotation matrix")
    trace = np.trace(r)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = [s / 4, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = math.sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            q = [(r[2, 1] - r[1, 2]) / s, s / 4, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
        elif i == 1:
            s = math.sqrt(1 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, s / 4, (r[1, 2] + r[2, 1]) / s]
        else:
            s = math.sqrt(1 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, s / 4]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def quat_to_rot(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def choose_source_segment(cameras, window=180, max_angle_deg=45, seed=None):
    """Choose a continuous pose segment; seeded selection picks a random window."""
    if window < 4:
        raise ValueError("source window must contain at least 4 cameras")
    ordered = core.make_trajectory(cameras, 0, 1, None)
    centers, qs = [], []
    for c in ordered:
        rot = np.asarray(c.R, dtype=float)
        trans = np.asarray(c.T, dtype=float)
        if trans.shape != (3,) or not np.isfinite(trans).all():
            raise ValueError(f"Invalid translation for camera {c.image_name}")
        centers.append(-rot @ trans)
        qs.append(rot_to_quat(rot))
    centers, qs = np.asarray(centers), np.asarray(qs)
    if len(centers) < 4:
        raise ValueError("At least 4 registered cameras are needed")
    lengths = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    positive = lengths[lengths > 1e-8]
    median = float(np.median(positive)) if len(positive) else 0.0
    threshold = max(10 * median, 3 * float(np.percentile(lengths, 95)), 1e-6)
    dots = np.clip(np.abs(np.sum(qs[1:] * qs[:-1], axis=1)), 0, 1)
    degrees = np.degrees(2 * np.arccos(dots))
    breaks = np.flatnonzero((lengths > threshold) | (degrees > max_angle_deg)) + 1
    boundaries = np.r_[0, breaks, len(ordered)]
    spans = [(int(a), int(b)) for a, b in zip(boundaries[:-1], boundaries[1:])]
    a, b = max(spans, key=lambda item: item[1] - item[0])
    if b - a < 4:
        raise ValueError("No continuous camera segment found; check filename order and COLMAP poses")
    count = min(window, b - a)
    if seed is None:
        start = a + (b - a - count) // 2
    else:
        eligible = [(left, right) for left, right in spans if right - left >= count]
        total = sum(right - left - count + 1 for left, right in eligible)
        pick = int(np.random.default_rng(seed).integers(total))
        for begin, end in eligible:
            choices = end - begin - count + 1
            if pick < choices:
                a, b = begin, end
                start = begin + pick
                break
            pick -= choices
    return ordered[start:start+count], {"start_index": start, "end_index_exclusive": start+count,
        "continuity_distance_threshold": threshold, "max_angle_degrees": max_angle_deg,
        "continuous_segment_length": b-a}


def cubic_weights(t):
    return np.array([(1-t)**3, 3*t**3-6*t*t+4, -3*t**3+3*t*t+3*t+1, t**3]) / 6


def synthesize_path(source, frame_count, control_step=12, lateral_fraction=0.05):
    if frame_count < 2 or control_step < 1 or not math.isfinite(lateral_fraction) or lateral_fraction < 0:
        raise ValueError("Invalid path parameters")
    indices = list(range(0, len(source), control_step))
    if indices[-1] != len(source)-1:
        indices.append(len(source)-1)
    controls = [source[i] for i in indices]
    centers = np.array([-np.asarray(c.R) @ np.asarray(c.T) for c in controls])
    qs = np.array([rot_to_quat(c.R) for c in controls])
    for i in range(1, len(qs)):
        if np.dot(qs[i], qs[i-1]) < 0:
            qs[i] *= -1
    control_distances = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    nonzero = control_distances[control_distances > 1e-8]
    amplitude = float(np.median(nonzero) * lateral_fraction) if len(nonzero) else 0.0
    generated = []
    for frame, fraction in enumerate(np.linspace(0, 1, frame_count)):
        x = fraction * (len(controls)-1)
        i = min(int(math.floor(x)), len(controls)-1)
        t = x-i
        ids = np.array([max(0, i-1), i, min(i+1, len(controls)-1), min(i+2, len(controls)-1)])
        w = cubic_weights(t)
        center = w @ centers[ids]
        q = w @ qs[ids]
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-8:
            raise ValueError("Camera rotations cannot be interpolated reliably")
        rotation = quat_to_rot(q / q_norm)
        center = center + rotation[:, 0] * (amplitude * math.sin(2 * math.pi * fraction))
        # CameraInfo.R is camera-to-world rotation, while T is world-to-camera translation.
        translation = -rotation.T @ center
        fovx = float(w @ np.array([controls[j].FovX for j in ids]))
        fovy = float(w @ np.array([controls[j].FovY for j in ids]))
        generated.append(SimpleNamespace(image_name=f"novel_{frame:06d}", R=rotation, T=translation,
                         FovX=fovx, FovY=fovy, width=source[0].width, height=source[0].height))
    return generated, {"control_step": control_step, "control_images": [c.image_name for c in controls],
                       "lateral_fraction": lateral_fraction, "lateral_amplitude": amplitude,
                       "trajectory_type": "cubic B-spline positions and normalized quaternion blends with lateral offset"}


def _nearest_camera_distances(points, cameras, chunk_size=64):
    """Brute-force distances in bounded chunks; works without SciPy."""
    distances = []
    for start in range(0, len(points), chunk_size):
        delta = points[start:start+chunk_size, None, :] - cameras[None, :, :]
        distances.extend(np.sqrt(np.square(delta).sum(axis=2).min(axis=1)).tolist())
    return np.asarray(distances)


def _smooth_samples(values, count):
    samples = []
    for f in np.linspace(0, len(values)-1, count):
        i = min(int(math.floor(f)), len(values)-1)
        ids = [max(0, i-1), i, min(i+1, len(values)-1), min(i+2, len(values)-1)]
        samples.append(cubic_weights(f-i) @ values[ids])
    return np.asarray(samples)


def synthesize_random_path(source, all_cameras, frame_count, control_step=24,
                           deviation=2.0, max_distance_fraction=0.10, seed=42, view_degrees=12.0):
    """Add smooth random motion to a capture-guided path with camera-support bounds.

    The camera-position bounding box and nearest-camera radius are coverage
    heuristics, not collision checks and not a reconstruction of free space.
    """
    params = [deviation, max_distance_fraction, view_degrees]
    if not all(math.isfinite(x) and x >= 0 for x in params) or max_distance_fraction == 0:
        raise ValueError("deviation and view-degrees must be nonnegative; max-distance-fraction must be positive")
    base, info = synthesize_path(source, frame_count, control_step, 0)
    base_positions = np.array([-c.R @ c.T for c in base])
    camera_centers = np.array([-np.asarray(c.R) @ np.asarray(c.T) for c in all_cameras], dtype=float)
    if len(camera_centers) < 4 or not np.isfinite(camera_centers).all():
        raise ValueError("Invalid camera positions for workspace estimate")
    bounds_min, bounds_max = camera_centers.min(axis=0), camera_centers.max(axis=0)
    diag = float(np.linalg.norm(bounds_max-bounds_min))
    if diag < 1e-8:
        raise ValueError("Camera centers do not span a workspace; cannot generate a random path")
    control_indices = list(range(0, len(source), control_step))
    if control_indices[-1] != len(source)-1:
        control_indices.append(len(source)-1)
    controls = np.array([-np.asarray(source[i].R) @ np.asarray(source[i].T) for i in control_indices])
    spacing = np.linalg.norm(np.diff(controls, axis=0), axis=1)
    nonzero = spacing[spacing > 1e-8]
    typical_spacing = float(np.median(nonzero)) if len(nonzero) else diag * 0.01
    amplitude = min(deviation * typical_spacing, diag * max_distance_fraction)
    rng = np.random.default_rng(seed)
    random_controls = max(6, min(16, int(math.ceil(len(source) / 60)) + 3))
    directions = rng.normal(size=(random_controls, 3))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    directions *= rng.uniform(0.7, 1.0, size=(random_controls, 1))
    # Primary movement is sideways; less vertical and forward displacement.
    directions *= np.array([1.0, 0.3, 0.45])
    offsets = _smooth_samples(directions, frame_count) * amplitude
    angles = _smooth_samples(rng.uniform(-1, 1, size=(random_controls, 1)), frame_count)[:, 0]
    # Smoothly enter and leave the random movement.
    taper = np.sin(np.linspace(0, np.pi, frame_count)) ** 2
    offsets *= taper[:, None]
    angles *= taper * np.deg2rad(view_degrees)
    proposed = np.stack([c.R @ delta for c, delta in zip(base, offsets)])
    # Smoothly constrain each coordinate to the bounding box of all captured cameras.
    margins = np.where(proposed >= 0, bounds_max - base_positions, base_positions - bounds_min)
    margins = np.maximum(margins, 0)
    bounded = np.sign(proposed) * margins * np.tanh(np.abs(proposed) / np.maximum(margins, 1e-12))
    base_dist = _nearest_camera_distances(base_positions, camera_centers)
    distance_limit = max(diag * max_distance_fraction, float(base_dist.max()) + 1e-7)
    candidate_dist = _nearest_camera_distances(base_positions + bounded, camera_centers)
    scale = 1.0
    if np.any(candidate_dist > distance_limit):
        low, high = 0.0, 1.0
        for _ in range(18):
            mid = (low + high) / 2
            if np.all(_nearest_camera_distances(base_positions + mid * bounded, camera_centers) <= distance_limit):
                low = mid
            else:
                high = mid
        scale = low
    positions = base_positions + scale * bounded
    generated = []
    for i, (camera, position, angle) in enumerate(zip(base, positions, angles)):
        ca, sa = math.cos(float(angle)), math.sin(float(angle))
        local_yaw = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
        rotation = camera.R @ local_yaw
        generated.append(SimpleNamespace(image_name=f"novel_{i:06d}", R=rotation, T=-rotation.T @ position,
                         FovX=camera.FovX, FovY=camera.FovY, width=camera.width, height=camera.height))
    actual = np.linalg.norm(positions-base_positions, axis=1)
    info.update({"trajectory_type": "seeded smooth random offsets around a capture-guided spline",
                 "seed": seed, "deviation": deviation, "typical_control_spacing": typical_spacing,
                 "requested_offset_amplitude": amplitude, "max_actual_offset": float(actual.max()),
                 "mean_actual_offset": float(actual.mean()), "constraint_scale": scale,
                 "random_controls": random_controls, "view_degrees_limit": view_degrees,
                 "camera_bounds_min": bounds_min.tolist(), "camera_bounds_max": bounds_max.tolist(),
                 "nearest_camera_distance_limit": distance_limit,
                 "max_nearest_camera_distance": float(_nearest_camera_distances(positions, camera_centers).max()),
                 "collision_checked": False})
    return generated, info


def stack_two(left, right, output, crf):
    temp = output.with_name(output.stem + ".partial.mp4")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(left), "-i", str(right),
           "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]", "-an", "-c:v", "libx264",
           "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp)]
    try:
        subprocess.run(cmd, check=True)
        os.replace(temp, output)
    finally:
        temp.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--model-3dgs", required=True, type=Path)
    parser.add_argument("--model-scaffold", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--3dgs-aa", dest="aa_3dgs", choices=["on", "off"], required=True)
    parser.add_argument("--scaffold-aa", choices=["on", "off"], required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--duration", type=float, default=20, help="Video duration in seconds")
    parser.add_argument("--path-mode", choices=["random", "smooth"], default="random")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for trajectory selection and motion")
    parser.add_argument("--source-window", type=int, default=600, help="Number of nearby source cameras")
    parser.add_argument("--control-step", type=int, default=24)
    parser.add_argument("--deviation", type=float, default=2.0,
                        help="Offset magnitude in units of median control-camera spacing")
    parser.add_argument("--max-distance-fraction", type=float, default=0.10,
                        help="Approximate camera-coverage radius as fraction of camera workspace diagonal")
    parser.add_argument("--view-degrees", type=float, default=12.0, help="Maximum additional camera pan in degrees")
    parser.add_argument("--lateral-fraction", type=float, default=0.05,
                        help="Lateral excursion as fraction of median control-camera spacing; 0 disables")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not all(math.isfinite(x) and x > 0 for x in (args.fps, args.duration)) or not 0 <= args.crf <= 51:
        parser.error("fps and duration must be positive, and crf must be between 0 and 51")
    if args.iteration != -1 and args.iteration < 1:
        parser.error("--iteration must be positive or -1")
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg with libx264 is required")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    source = args.source.expanduser().resolve()
    if not source.is_dir():
        parser.error(f"Scene not found: {source}")
    output = args.output_dir.expanduser().resolve()
    outputs = [output / '3dgs.mp4', output / 'scaffold.mp4', output / 'side_by_side.mp4', output / 'trajectory.json']
    if not args.overwrite and any(p.exists() for p in outputs):
        parser.error(f"Outputs already exist in {output}; choose another folder or use --overwrite")
    try:
        specs = [core.model_spec(args.model_3dgs, "3dgs", args.iteration, source),
                 core.model_spec(args.model_scaffold, "scaffold", args.iteration, source)]
        if specs[0]['cfg'].get('white_background', False) != specs[1]['cfg'].get('white_background', False):
            raise ValueError("Models use different backgrounds")
        all_cameras = core.read_cameras(source)
        segment, selection = choose_source_segment(all_cameras, args.source_window,
                                                   seed=args.seed if args.path_mode == "random" else None)
        nframes = max(2, round(args.duration * args.fps))
        if args.path_mode == "random":
            cameras, synthesis = synthesize_random_path(segment, all_cameras, nframes, args.control_step,
                                  args.deviation, args.max_distance_fraction, args.seed, args.view_degrees)
        else:
            cameras, synthesis = synthesize_path(segment, nframes, args.control_step, args.lateral_fraction)
        width, height = core.video_dimensions(cameras, args.width)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    output.mkdir(parents=True, exist_ok=True)
    records = core.trajectory_record(cameras)
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    details = {"source": str(source), "frame_count": len(cameras), "fps": args.fps,
               "duration_seconds": len(cameras) / args.fps, "width": width, "height": height,
               "source_segment": selection, "synthesis": synthesis, "trajectory_sha256": digest,
               "side_by_side": {"left": "3dgs", "right": "scaffold"},
               "color_protocol": "base model appearance; no per-image exposure transforms",
               "models": {s['type']: {"path": str(s['path']), "iteration": s['iteration'],
                                      "antialiasing": aa == 'on'}
                          for s, aa in zip(specs, [args.aa_3dgs, args.scaffold_aa])}, "cameras": records}
    outputs[3].write_text(json.dumps(details, indent=2, allow_nan=False) + '\n')
    print(f"Synthesized {len(cameras)} views from {len(segment)} cameras "
          f"({segment[0].image_name} to {segment[-1].image_name}).", flush=True)
    if args.path_mode == "random":
        print(f"Seed {args.seed}; max offset: {synthesis['max_actual_offset']:.5g} COLMAP units; "
              f"nearest-camera limit: {synthesis['nearest_camera_distance_limit']:.5g}. "
              "Bounds are approximate; path is not collision checked.", flush=True)
    for spec, aa, destination in zip(specs, [args.aa_3dgs, args.scaffold_aa], outputs[:2]):
        core.write_video(destination, core.rendered_frames(spec, cameras, width, height, aa == 'on'),
                         width, height, args.fps, args.crf)
    stack_two(outputs[0], outputs[1], outputs[2], args.crf)
    for path in outputs:
        print(path)


if __name__ == '__main__':
    main()

