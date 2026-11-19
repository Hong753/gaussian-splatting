"""Test pose changes against LiDAR-supported features in entirely withheld photos."""
import cv2
import numpy as np

from .common import project
from .features import extract_features, match_pair


def check_views(source, folder, cameras, checks, selected, features, points, normals, planar,
                rotations, centers, per_image, args):
    rows = []
    active = [i for i, row in enumerate(per_image) if row["optimized"]]
    for im in checks:
        ref = extract_features(source, folder, im, cameras[im.camera_id], points, normals, planar, args)
        nearby = sorted((i for i in active if im.R[2] @ selected[i].R[2] > .5),
                        key=lambda i: np.linalg.norm(im.C - selected[i].C))[:6]
        errors, ids = [], set()
        paired_views = 0
        for i in nearby:
            target, camera = features[i], selected[i]
            cv2.setRNGSeed((im.id * 100003 + camera.id) % (2**31 - 1))
            matches = match_pair(ref, target, im, camera, args)
            if len(matches) < args.min_pair_matches:
                continue
            left, right = matches.T
            # Both predictions use the same fixed world point from the withheld
            # photo and the same measured SIFT pixel in the training photo.
            xyz, observed = ref["xyz"][left], target["uv"][right]
            before, _ = project(xyz, camera.R, camera.C, target["K"])
            after, depth = project(xyz, rotations[i], centers[i], target["K"])
            old = np.linalg.norm((before - observed) / target["scale"], axis=1)
            new = np.linalg.norm((after - observed) / target["scale"], axis=1)
            new[depth <= .2] = np.inf
            errors.append(np.column_stack((old, new)))
            ids.update(left.tolist())
            paired_views += 1
        row = dict(image=im.name, matched_train_views=paired_views, distinct_features=len(ids),
                   count=sum(map(len, errors)), supported=False)
        if errors:
            uv = ref['uv'][sorted(ids)] / ref['scale']
            cam = cameras[im.camera_id]
            quadrants = (uv[:, 0] >= cam.width / 2).astype(int) + 2 * (uv[:, 1] >= cam.height / 2)
            row['occupied_quadrants'] = len(np.unique(quadrants))
            residuals = np.concatenate(errors)
            row['invalid_after'] = int((~np.isfinite(residuals[:, 1])).sum())
            if not row['invalid_after']:
                row.update(median=np.median(residuals, axis=0).tolist(),
                           p90=np.percentile(residuals, 90, axis=0).tolist(),
                           supported=(paired_views >= 2 and len(ids) >= args.min_pair_matches
                                      and row['occupied_quadrants'] >= 2))
        rows.append(row)
        print(f"Independent photo {len(rows)}/{len(checks)}: {im.name}, {row['count']} matches, "
              f"native-pixel medians {row.get('median')}", flush=True)
    supported = [r for r in rows if r["supported"]]
    reasons = []
    if any(r.get('invalid_after', 0) for r in rows):
        reasons.append("Pose changes put validation features behind the camera")
    if len(supported) < max(min(3, len(checks)), (len(checks) + 1) // 2):
        reasons.append("Insufficient independent photo coverage to judge refinement")
    delta = [r["median"][0] - r["median"][1] for r in supported]
    gain = float(np.median(delta)) if delta else None
    if gain is not None and gain < .1:
        reasons.append("Typical independent photo improved by less than 0.1 native pixel")
    medians = np.median([r["median"] for r in supported], axis=0).tolist() if supported else None
    if supported:
        tails = np.median([r["p90"] for r in supported], axis=0)
        if tails[1] > 1.05 * tails[0] + .1:
            reasons.append("Independent photo p90 errors worsened")
    return dict(per_image=rows, supported=len(supported), requested=len(rows),
                median_before_after_native_px=medians, median_gain_native_px=gain, improved=sum(d > 0 for d in delta),
                rejection_reasons=reasons,
                protocol="Matches gated at the original poses. LiDAR plane intersections from withheld photos "
                         "are projected into the same training images with original and adjusted poses. "
                         "Each supported withheld photo gets one vote.")
