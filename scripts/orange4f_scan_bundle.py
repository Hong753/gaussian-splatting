"""Register a learned RGB reconstruction to a metric scan and refine its camera poses."""

import math
import numpy as np
import pycolmap
from scripts.orange4f_lidar_tracks import scan_xyz
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def scan_tree(path):
    vertices = scan_xyz(path)
    rng = np.random.default_rng(7)
    subset = vertices[np.sort(rng.choice(len(vertices), min(4_000_000, len(vertices)), replace=False))]
    xyz = np.column_stack([subset[k] for k in "xyz"]) if subset.dtype.names else subset
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
    scale = np.median(
        np.linalg.norm(reference_xyz[left[wide]] - reference_xyz[right[wide]], axis=1) / d[wide]
    )
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


def stored_anchors(rec, conf, z, reference):
    # Image IDs and point2D indices in the new SuperGlue model are unrelated
    # to those in the source. Append temporary observations by image name.
    image_by_name = {im.name: im for im in rec.images.values()}
    names, indices, pids, coords, pixels = (z[k] for k in ("names", "image", "pid", "xyz", "xy"))
    coverage = {im.image_id: 0 for im in image_by_name.values()}
    lengths, scan_ids, fixed_xyz = {}, {}, []
    held = []
    order = np.argsort(indices, kind="stable")
    sorted_indices = indices[order]
    for j, group in zip(*np.unique(sorted_indices, return_index=True)):
        # The samples are grouped by source image, irrespective of its mapper ID.
        name = str(names[j])
        image = image_by_name[name]
        source = reference[name]
        cam = rec.camera(image.camera_id)
        scale = np.array([cam.width / source["width"], cam.height / source["height"]])
        if not np.isclose(scale[0], scale[1], rtol=0.005):
            raise ValueError(f"Mapped camera and stored pixels have different aspect ratios: {name}")
        end = np.searchsorted(sorted_indices, j, side="right")
        which = order[group:end]
        xy = pixels[which] * scale
        xyz = coords[which]
        withheld = pids[which] % 10 == 0
        if withheld.any():
            held.append((image, cam, xyz[withheld], xy[withheld]))
        lengths[image.image_id] = image.num_points2D()
        for source_id, world, pixel in zip(pids[which][~withheld], xyz[~withheld], xy[~withheld]):
            feature = image.num_points2D()
            image.points2D.append(pycolmap.Point2D(pixel))
            element = pycolmap.TrackElement(image.image_id, feature)
            if source_id in scan_ids:
                rec.add_observation(scan_ids[source_id], element)
            else:
                pid = rec.add_point3D(world, pycolmap.Track([element]))
                conf.add_constant_point(pid)
                scan_ids[source_id] = pid
        fixed_xyz.append(xyz[~withheld])
        coverage[image.image_id] = int((~withheld).sum())
    fixed = np.concatenate(fixed_xyz)
    if len(fixed) < 3 or np.linalg.matrix_rank(fixed - fixed.mean(0), tol=0.001) < 2:
        raise ValueError("Insufficient non-collinear registered scan measurements")

    def held_error():
        errors = []
        for im, cam, xyz, xy in held:
            local = im.cam_from_world * xyz
            front = local[:, 2] > 0
            if front.any():
                uv = cam.img_from_cam(local[front, :2] / local[front, 2:])
                errors.extend(np.linalg.norm(uv - xy[front], axis=1))
        return float(np.median(errors)) if errors else None

    report = dict(
        mode="stored scan XYZ and image pixels",
        scan_anchors=len(scan_ids),
        scan_observations=sum(coverage.values()),
        withheld_observations=sum(len(h[2]) for h in held),
        withheld_median_reprojection_before_px=held_error(),
        cameras_with_scan_anchors=sum(n > 0 for n in coverage.values()),
    )
    return report, coverage, held_error, scan_ids.values(), lengths


def plane_anchors(rec, conf, cloud, images):
    tree = scan_tree(cloud)
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

    report = dict(
        mode="local scan planes",
        scan_anchors=len(anchor_xyz),
        unused_scan_tracks=len(held_ids),
        unused_median_plane_distance_before_m=unused_error(),
        cameras_with_scan_anchors=sum(x > 0 for x in coverage.values()),
    )
    return report, coverage, unused_error


def refine(model, cloud, reference, output, observations=None):
    rec = pycolmap.Reconstruction(str(model))
    rec.check()
    report = register(rec, reference)
    images = list(rec.reg_image_ids())
    old = {i: rec.image(i).projection_center().copy() for i in images}
    rec.update_point_3d_errors()
    report["rgb_reprojection_pre_ba_px"] = rec.compute_mean_reprojection_error()
    conf = pycolmap.BundleAdjustmentConfig()
    for i in images:
        conf.add_image(i)
    direct = observations is not None and len(observations["pid"]) > 0
    if direct:
        metrics, coverage, held_error, temp_ids, lengths = stored_anchors(rec, conf, observations, reference)
    else:
        metrics, coverage, held_error = plane_anchors(rec, conf, cloud, images)
    report.update(metrics, registered_cameras=len(images), scan=str(cloud))
    opts = pycolmap.BundleAdjustmentOptions()
    opts.refine_focal_length = opts.refine_principal_point = opts.refine_extra_params = False
    opts.loss_function_type = pycolmap.LossFunctionType.SOFT_L1
    opts.solver_options.max_num_iterations = 35
    opts.solver_options.function_tolerance = 1e-6
    opts.solver_options.num_threads = 8
    print(
        f"LiDAR-anchored bundle adjustment: {conf.num_constant_points():,} scan anchors, "
        f"{len(images):,} variable poses, {rec.num_points3D():,} total landmarks",
        flush=True,
    )
    solver = pycolmap.create_default_bundle_adjuster(opts, conf, rec)
    result = solver.solve()
    if not result.IsSolutionUsable() or not math.isfinite(result.final_cost):
        raise RuntimeError(result.BriefReport())
    del solver
    if direct:
        report["withheld_median_reprojection_after_px"] = held_error()
        for pid in temp_ids:
            rec.delete_point3D(pid)
        for iid, length in lengths.items():
            del rec.image(iid).points2D[length:]
    else:
        report["unused_median_plane_distance_after_m"] = held_error()
    rec.update_point_3d_errors()
    rec.check()
    movement = np.array([np.linalg.norm(rec.image(i).projection_center() - old[i]) for i in images])
    if not np.isfinite(movement).all():
        raise ValueError("Non-finite refined cameras")
    report.update(
        solver=result.BriefReport(),
        rgb_reprojection_post_ba_px=rec.compute_mean_reprojection_error(),
        median_camera_update_m=float(np.median(movement)),
        max_camera_update_m=float(movement.max()),
        direct_scan_tracks_per_camera={rec.image(i).name: n for i, n in coverage.items()},
    )
    output.mkdir()
    rec.write_binary(str(output))
    return report
