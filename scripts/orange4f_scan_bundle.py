"""Register a learned RGB reconstruction to a metric scan and refine its camera poses."""

import math
import numpy as np
import pycolmap
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def scan_tree(path):
    vertices = PlyData.read(path, mmap="r")["vertex"].data
    if not {"x", "y", "z"}.issubset(vertices.dtype.names) or "f_dc_0" in vertices.dtype.names:
        raise ValueError(f"Expected a registered XYZ scan PLY: {path}")
    # Keep storage bounded for a raw, high-density scan, with deterministic sampling.
    rng = np.random.default_rng(7)
    subset = vertices[np.sort(rng.choice(len(vertices), min(4_000_000, len(vertices)), replace=False))]
    xyz = np.column_stack([subset[k] for k in "xyz"])
    xyz = xyz[np.isfinite(xyz).all(1)]
    if len(xyz) < 16:
        raise ValueError("Scan contains too few usable samples")
    return cKDTree(xyz)


def register(rec, reference):
    imgs = [rec.image(i) for i in sorted(rec.reg_image_ids())]
    learned = np.stack([i.projection_center() for i in imgs])
    prior = [reference[i.name] for i in imgs]
    reference_rot = np.stack([i["R"] for i in prior])
    reference_xyz = -(reference_rot @ np.array([i["T"] for i in prior])[..., None])[..., 0]
    rotations = Rotation.from_matrix(
        reference_rot @ np.stack([i.cam_from_world.rotation.matrix() for i in imgs])
    )
    mean = rotations.mean()
    for _ in range(3):
        angle = (mean.inv() * rotations).magnitude()
        mean = rotations[angle <= np.quantile(angle, 0.8)].mean()
    moved = mean.apply(learned)
    rng = np.random.default_rng(31)
    left, right = rng.integers(len(imgs), size=(2, 8192))
    d = np.linalg.norm(moved[left] - moved[right], axis=1)
    wide = d > max(1e-8, np.median(d))
    if not wide.any():
        raise ValueError("Camera centers do not span a trajectory")
    scale = np.median(np.linalg.norm(reference_xyz[left[wide]] - reference_xyz[right[wide]], axis=1) / d[wide])
    shift = np.median(reference_xyz - scale * moved, axis=0)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid metric scale")
    sim = pycolmap.Sim3d(float(scale), pycolmap.Rotation3d(mean.as_matrix()), shift)
    rec.transform(sim)
    return dict(
        transform=sim.matrix().tolist(),
        median_center_disagreement_m=float(
            np.median(np.linalg.norm(reference_xyz - (scale * moved + shift), axis=1))
        ),
    )


def local_surfaces(tree, xyz):
    dist, idx = tree.query(xyz, k=12, workers=4)
    cloud = tree.data[idx]
    center = cloud.mean(1)
    delta = cloud - center[:, None, :]
    eig, vec = np.linalg.eigh(delta.transpose(0, 2, 1) @ delta / 12)
    normal = vec[:, :, 0]
    offset = np.einsum("ij,ij->i", xyz - center, normal)
    usable = (
        (dist[:, 0] < 0.06)
        & (dist[:, -1] < 0.18)
        & (eig[:, 1] > 1e-7)
        & (eig[:, 0] < 0.08 * eig[:, 1])
        & (eig[:, 1] > 0.04 * eig[:, 2])
        & (np.abs(offset) < 0.05)
    )
    return xyz - offset[:, None] * normal, normal, usable


def refine(model, cloud, reference, output):
    rec = pycolmap.Reconstruction(str(model))
    rec.check()
    report = register(rec, reference)
    tree = scan_tree(cloud)
    images = list(rec.reg_image_ids())
    old = {i: rec.image(i).projection_center().copy() for i in images}
    report["rgb_reprojection_pre_ba_px"] = rec.compute_mean_reprojection_error()
    conf = pycolmap.BundleAdjustmentConfig()
    for i in images:
        conf.add_image(i)
    pids = np.asarray(sorted(rec.point3D_ids()), dtype=np.int64)
    held_ids, held_targets, held_normals, anchor_xyz = [], [], [], []
    coverage = {i: 0 for i in images}
    for start in range(0, len(pids), 8192):
        batch = pids[start : start + 8192]
        batch = [int(i) for i in batch if rec.point3D(int(i)).track.length() >= 3]
        if not batch:
            continue
        xyz = np.stack([rec.point3D(i).xyz for i in batch])
        projected, normals, good = local_surfaces(tree, xyz)
        for j in np.flatnonzero(good):
            pid = batch[j]
            if pid % 10 == 0:
                held_ids.append(pid)
                held_targets.append(projected[j])
                held_normals.append(normals[j])
                continue
            p = rec.point3D(pid)
            # Project only along the scan normal, keeping the learned tangential
            # coordinates as a fixed approximation during this BA.
            p.xyz = projected[j]
            conf.add_constant_point(pid)
            anchor_xyz.append(projected[j])
            for obs in p.track.elements:
                coverage[obs.image_id] += 1
    if (
        len(anchor_xyz) < 3
        or np.linalg.matrix_rank(np.asarray(anchor_xyz) - np.mean(anchor_xyz, axis=0), tol=0.001) < 2
    ):
        raise ValueError("Insufficient non-collinear, scan-supported tracks; check cloud and scan frame")

    def unused_error():
        if not held_ids:
            return None
        xyz = np.stack([rec.point3D(i).xyz for i in held_ids])
        return float(np.median(np.abs(np.einsum("ij,ij->i", xyz - held_targets, held_normals))))

    report.update(
        scan=str(cloud),
        scan_anchors=len(anchor_xyz),
        unused_scan_tracks=len(held_ids),
        unused_median_plane_distance_before_m=unused_error(),
        registered_cameras=len(images),
        cameras_with_scan_anchors=sum(x > 0 for x in coverage.values()),
    )
    opts = pycolmap.BundleAdjustmentOptions()
    opts.refine_focal_length = opts.refine_principal_point = opts.refine_extra_params = False
    opts.loss_function_type = pycolmap.LossFunctionType.SOFT_L1
    opts.solver_options.max_num_iterations = 35
    opts.solver_options.function_tolerance = 1e-6
    opts.solver_options.num_threads = 8
    print(
        f"LiDAR-anchored bundle adjustment: {conf.num_constant_points():,} scan anchors, "
        f"{len(images):,} variable poses, {len(pids):,} total landmarks",
        flush=True,
    )
    solver = pycolmap.create_default_bundle_adjuster(opts, conf, rec)
    result = solver.solve()
    if not result.IsSolutionUsable() or not math.isfinite(result.final_cost):
        raise RuntimeError(result.BriefReport())
    rec.update_point_3d_errors()
    rec.check()
    movement = np.array([np.linalg.norm(rec.image(i).projection_center() - old[i]) for i in images])
    if not np.isfinite(movement).all():
        raise ValueError("Non-finite refined cameras")
    report.update(
        solver=result.BriefReport(),
        unused_median_plane_distance_after_m=unused_error(),
        rgb_reprojection_post_ba_px=rec.compute_mean_reprojection_error(),
        median_camera_update_m=float(np.median(movement)),
        max_camera_update_m=float(movement.max()),
        direct_scan_tracks_per_camera={rec.image(i).name: n for i, n in coverage.items()},
    )
    output.mkdir()
    rec.write_binary(str(output))
    return report
