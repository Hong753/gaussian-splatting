"""Sparse robust BA with fixed LiDAR planes, fixed intrinsics and pose priors."""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation


def observation_arrays(tracks, features):
    camera, landmark, uv, heldout = [], [], [], []
    for k, track in enumerate(tracks):
        # Reserve one observation per track. It is never used by the optimizer.
        reserve = k % len(track["obs"])
        for j, (i, feature) in enumerate(track["obs"]):
            camera.append(i)
            landmark.append(k)
            uv.append(features[i]["uv"][feature])
            heldout.append(j == reserve)
    return np.array(camera), np.array(landmark), np.array(uv), np.array(heldout)


def solve_bundle(images, features, tracks, args):
    camera, landmark, observed, heldout = observation_arrays(tracks, features)
    R0, C0 = np.array([im.R for im in images]), np.array([im.C for im in images])
    K = np.array([f["K"] for f in features])
    anchors, normals, X0 = [], [], []
    for k, track in enumerate(tracks):
        # Do not initialize landmarks or their priors using the withheld pixel.
        training = [obs for j, obs in enumerate(track["obs"]) if j != k % len(track["obs"])]
        samples = np.array([features[i]["xyz"][feature] for i, feature in training])
        center = np.median(samples, axis=0)
        i, feature = training[int(np.argmin(np.linalg.norm(samples - center, axis=1)))]
        anchor, normal = features[i]["xyz"][feature], features[i]["normals"][feature]
        center -= normal * ((center - anchor) @ normal)
        anchors.append(anchor)
        normals.append(normal)
        X0.append(center)
    anchors, normals, X0 = np.array(anchors), np.array(normals), np.array(X0)
    counts = np.bincount(camera[~heldout], minlength=len(images))
    active = np.flatnonzero(counts >= args.min_observations)
    if not len(active):
        raise ValueError("No camera has enough training observations; poses have not been changed")
    # A completely unconstrained camera stays exactly at its original pose.
    active_index = np.full(len(images), -1, dtype=int)
    active_index[active] = np.arange(len(active))

    def predictions(R, C, X):
        local = np.einsum("nij,nj->ni", R[camera], X[landmark] - C[camera])
        return (local[:, :2] / np.maximum(local[:, 2:3], .01)) * K[camera, :2] + K[camera, 2:]

    def run(Xstart, optimize_poses, steps):
        ncam = len(active) if optimize_poses else 0
        offset = 6 * ncam
        nvar = offset + X0.size
        train_indices = np.flatnonzero(~heldout)

        def unpack(x):
            R, C = R0.copy(), C0.copy()
            if ncam:
                delta = x[:offset].reshape(-1, 6)
                R[active] = Rotation.from_rotvec(delta[:, :3]).as_matrix() @ R0[active]
                C[active] += delta[:, 3:]
            X = X0 + x[offset:].reshape(-1, 3)
            return R, C, X

        def residual(x):
            R, C, X = unpack(x)
            reprojection = (predictions(R, C, X)[~heldout] - observed[~heldout]).ravel() / args.pixel_sigma
            plane = np.sum((X - anchors) * normals, axis=1) / args.surface_sigma
            # Weak finite support of local plane patches prevents sliding by a tile.
            support = ((X - X0) / args.landmark_sigma).ravel()
            prior = np.empty(0)
            if ncam:
                delta = x[:offset].reshape(-1, 6)
                prior = (delta / np.array([np.deg2rad(args.rotation_prior_deg)] * 3 + [args.translation_prior] * 3)).ravel()
            return np.concatenate((reprojection, plane, support, prior))

        rows, cols = [], []

        def block(row_ids, column_ids):
            rows.extend(np.repeat(row_ids, len(column_ids)))
            cols.extend(np.tile(column_ids, len(row_ids)))

        for row, obs in enumerate(train_indices):
            block([2 * row, 2 * row + 1], offset + 3 * landmark[obs] + np.arange(3))
            index = active_index[camera[obs]]
            if ncam and index >= 0:
                block([2 * row, 2 * row + 1], 6 * index + np.arange(6))
        base = 2 * len(train_indices)
        for k in range(len(tracks)):
            block([base + k], offset + 3 * k + np.arange(3))
            for axis in range(3):
                block([base + len(tracks) + 3 * k + axis], [offset + 3 * k + axis])
        base += len(tracks) * 4
        if ncam:
            rows.extend(base + np.arange(offset))
            cols.extend(np.arange(offset))
        sparsity = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(base + offset, nvar)).tocsr()
        x = np.concatenate((np.zeros(offset), (Xstart - X0).ravel()))
        bounds = np.full(nvar, args.max_landmark_shift / np.sqrt(3))
        if ncam:
            bounds[:offset] = np.tile([np.deg2rad(args.max_rotation_deg) / np.sqrt(3)] * 3 +
                                      [args.max_translation / np.sqrt(3)] * 3, ncam)
        x = np.clip(x, -bounds + 1e-10, bounds - 1e-10)
        result = least_squares(residual, x, jac_sparsity=sparsity, method="trf",
                               tr_solver="lsmr", x_scale="jac", loss="soft_l1",
                               bounds=(-bounds, bounds), max_nfev=steps,
                               ftol=1e-5, xtol=1e-6, gtol=1e-6, verbose=1)
        return *unpack(result.x), result

    print(f"Fitting {len(tracks)} landmarks at original poses for the validation baseline", flush=True)
    _, _, Xbaseline, baseline_fit = run(X0, False, args.baseline_steps)
    before = np.linalg.norm(predictions(R0, C0, Xbaseline) - observed, axis=1)
    print(f"Refining {len(active)}/{len(images)} poses; {heldout.sum()} observations held out", flush=True)
    R, C, X, fit = run(Xbaseline, True, args.max_steps)
    after = np.linalg.norm(predictions(R, C, X) - observed, axis=1)
    rotation = np.rad2deg(Rotation.from_matrix(R @ R0.transpose(0, 2, 1)).magnitude())
    translation = np.linalg.norm(C - C0, axis=1)
    depth = np.einsum("nij,nj->ni", R[camera], X[landmark] - C[camera])[:, 2]

    def stats(values):
        return dict(count=len(values), median=float(np.median(values)),
                    p90=float(np.percentile(values, 90)), mean=float(np.mean(values))) if len(values) else None

    val_before, val_after = stats(before[heldout]), stats(after[heldout])
    reasons = []
    if val_after["median"] > val_before["median"] - args.min_improvement_pixels:
        reasons.append("Held-out median reprojection error did not improve enough")
    if val_after["p90"] > val_before["p90"] * 1.05 + .05:
        reasons.append("Held-out p90 reprojection error worsened")
    if np.any(depth <= .2) or not np.isfinite(after).all():
        reasons.append("Refined landmarks have invalid projections")
    per_image = []
    for i, im in enumerate(images):
        mask = heldout & (camera == i)
        b, a = stats(before[mask]), stats(after[mask])
        # Prevent a global average improvement from hiding a badly damaged room.
        if active_index[i] >= 0 and a and a["count"] >= 8 and a["median"] > b["median"] * 1.25 + .5:
            reasons.append(f"Held-out error increased substantially for {im.name}")
        per_image.append(dict(name=im.name, optimized=bool(active_index[i] >= 0),
                              training_observations=int(counts[i]), translation=float(translation[i]),
                              rotation_deg=float(rotation[i]), before=b, after=a))
    report = dict(accepted=not reasons, rejection_reasons=reasons, tracks=len(tracks),
                  optimized_cameras=len(active), selected_cameras=len(images),
                  train_before=stats(before[~heldout]), train_after=stats(after[~heldout]),
                  validation_before=val_before, validation_after=val_after,
                  max_translation=float(translation.max()), max_rotation_deg=float(rotation.max()),
                  solver=dict(status=int(fit.status), message=fit.message, nfev=int(fit.nfev),
                              initial_landmark_nfev=int(baseline_fit.nfev), optimality=float(fit.optimality)),
                  per_image=per_image,
                  notes=["Validation observations are withheld from BA, but feature matching/track construction uses all selected images.",
                         "Distances are in input map units; intrinsics and LiDAR points are fixed.",
                         "Planar/repetitive scenes may not constrain every pose direction; priors preserve the original gauge.",
                         "This validates reprojection consistency, not absolute pose accuracy or 3DGS quality."])
    return R, C, X, report
