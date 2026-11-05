#!/usr/bin/env python3
"""Render two models on an automatically synthesized smooth novel-view path.

The path is derived from COLMAP poses and interpolated, with a small lateral
excursion. It is not a collision-checked camera path.
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


def choose_source_segment(cameras, window=180, max_angle_deg=45):
    """Choose the center of the longest pose sequence without large jumps."""
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
    start = a + (b - a - count) // 2
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
    parser.add_argument("--duration", type=float, default=10, help="Video duration in seconds")
    parser.add_argument("--source-window", type=int, default=180, help="Number of nearby source cameras")
    parser.add_argument("--control-step", type=int, default=12)
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
        segment, selection = choose_source_segment(core.read_cameras(source), args.source_window)
        cameras, synthesis = synthesize_path(segment, max(2, round(args.duration * args.fps)),
                                              args.control_step, args.lateral_fraction)
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
    for spec, aa, destination in zip(specs, [args.aa_3dgs, args.scaffold_aa], outputs[:2]):
        core.write_video(destination, core.rendered_frames(spec, cameras, width, height, aa == 'on'),
                         width, height, args.fps, args.crf)
    stack_two(outputs[0], outputs[1], outputs[2], args.crf)
    for path in outputs:
        print(path)


if __name__ == '__main__':
    main()
