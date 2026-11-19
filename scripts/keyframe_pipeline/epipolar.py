"""Cross-check observed RGB correspondences against poses without using a depth map."""
import numpy as np


def line_errors(ref, target, matches, ref_camera, target_rotation, target_center):
    """Distance in native reference pixels from observations to their epipolar lines."""
    left, right = matches.T
    K = target["K"]
    rays = np.column_stack(((target["uv"][right] - K[2:])/K[:2], np.ones(len(right)))) @ target_rotation
    # World-space epipolar plane through both camera centers and the target ray.
    n = np.cross(target_center - ref_camera.C, rays) @ ref_camera.R.T
    K = ref["K"]
    xy = (ref["uv"][left] - K[2:])/K[:2]
    scale = np.linalg.norm(n[:, :2] * ref["scale"]/K[:2], axis=1)
    numerator = np.abs(np.einsum("ni,ni->n", n[:, :2], xy) + n[:, 2])
    return np.divide(numerator, scale, out=np.full(len(matches), np.inf), where=scale > 1e-12)


def original_pose_errors(features, cameras, pairs, max_translation):
    rows = []
    for i, j, matches in pairs:
        a, b = cameras[i], cameras[j]
        if np.linalg.norm(a.C - b.C) <= 2 * max_translation:
            continue
        error = line_errors(features[i], features[j], matches, a, b.R, b.C)
        finite = error[np.isfinite(error)]
        if len(finite):
            rows.append(dict(reference=a.name, target=b.name, count=len(finite),
                             median=float(np.median(finite)), p90=float(np.percentile(finite, 90))))
    typical = float(np.median([row["median"] for row in rows])) if rows else None
    return dict(pairs=len(rows), measurements=sum(row["count"] for row in rows),
                median_of_pair_medians_native_px=typical, per_pair=rows)
