"""Fixed-camera RGB and an RGB-only track/triangulation path for rectified images."""


import numpy as np
from scipy import ndimage
from scipy.spatial import Delaunay, cKDTree

from .render import Camera


def load_views(path):
    with np.load(path) as source:
        rgb = source["rgb"].astype(np.float64)/255
        h, w = rgb.shape[1:3]
        cams = [Camera(c, r, k, w, h) for c, r, k in
                zip(source["centers"], source["rotations"], source["intrinsics"])]
    return rgb, cams


def disparity_observations(images, cams, ref, inverse_depth):
    """Return per-view patch correspondence depths in a fixed, rectified rig."""
    cam = cams[ref]
    for target in cams:
        if not (np.allclose(cam.rotation, target.rotation) and
                np.allclose(cam.intrinsics, target.intrinsics) and
                np.allclose(((target.center-cam.center) @ cam.rotation.T)[1:], 0)):
            raise ValueError("This correspondence path requires horizontal rectification.")
    gray = images @ [.25, .5, .25]
    a = gray[ref]
    yy, xx = np.indices(a.shape, dtype=float)
    box = lambda value: ndimage.uniform_filter(value, size=5, mode="nearest")
    avg_a = box(a)
    var_a = np.maximum(box(a*a)-avg_a*avg_a, 0)
    estimates, scores = [], []
    for j in range(len(cams)):
        if j == ref:
            continue
        offset = ((cams[j].center-cam.center) @ cam.rotation.T)[0]*cam.intrinsics[0]
        volume = []
        for d in inverse_depth:
            tx = xx-offset*d
            b = ndimage.map_coordinates(gray[j], [yy, tx], order=1, mode="nearest")
            avg_b = box(b)
            var_b = np.maximum(box(b*b)-avg_b*avg_b, 0)
            corr = (box(a*b)-avg_a*avg_b)/np.sqrt(np.maximum(var_a*var_b, 1e-10))
            valid = (tx >= 3) & (tx < cam.width-3) & (var_a > .00035) & (var_b > .00035)
            volume.append(np.where(valid, corr, -2.))
        volume = np.array(volume)
        peak = volume.argmax(0)
        score = np.take_along_axis(volume, peak[None], axis=0)[0]
        rival = np.max(np.where(np.abs(np.arange(len(volume))[:, None, None]-peak) > 5, volume, -2), axis=0)
        left = volume[np.maximum(peak-1, 0), yy.astype(int), xx.astype(int)]
        right = volume[np.minimum(peak+1, len(volume)-1), yy.astype(int), xx.astype(int)]
        delta = np.clip(.5*(left-right)/np.minimum(left-2*score+right, -1e-8), -.5, .5)
        d = inverse_depth[peak] + delta*(inverse_depth[1]-inverse_depth[0])
        valid = (score > .86) & (score-rival > .032) & (peak > 0) & (peak < len(volume)-1)
        estimates.append(d)
        scores.append(np.where(valid, score, -1))
    return np.stack(estimates), np.stack(scores), var_a


def reconstruct_tracks(images, cams, cell, dense, depth_range=(1.35, 6.8)):
    ref = min(range(len(cams)), key=lambda i: np.linalg.norm(cams[i].center -
              np.mean([camera.center for camera in cams], axis=0)))
    others = np.array([j for j in range(len(cams)) if j != ref])
    invs, scores, var = disparity_observations(
        images, cams, ref, np.linspace(1/depth_range[1], 1/depth_range[0], 151))
    # Choose the strongest compatible set of two or more independently matched views.
    agreeing = (np.abs(invs[:, None]-invs[None]) < .017) & (scores[:, None] > 0) & (scores[None] > 0)
    votes = (agreeing * scores[None].clip(0)).sum(1)
    best = votes.argmax(0)
    chosen = np.take_along_axis(agreeing, best[None,None], axis=0)[0]
    scale = np.array([np.linalg.norm(cams[j].center-cams[ref].center)**2 for j in others])
    weight = chosen * scale[:,None,None] * scores.clip(0)
    inv = (invs*weight).sum(0)/np.maximum(weight.sum(0), 1e-9)
    confidence = votes.max(0)
    intensity = images[ref] @ [.25,.50,.25]
    gy, gx = np.gradient(intensity)
    sm = lambda q: ndimage.uniform_filter(q, size=5)
    xx, xy, yy = sm(gx*gx), sm(gx*gy), sm(gy*gy)
    corner = (xx+yy-np.sqrt((xx-yy)**2+4*xy**2))*.5
    corner_peaks = corner >= .75*ndimage.maximum_filter(corner, size=5)
    eligible = (chosen.sum(0) >= 2) & (var > .00035)
    if not dense:
        eligible &= (corner > .00010) & corner_peaks
    if not eligible.any():
        raise ValueError("No tracks satisfy the fixed-pose matching gates.")
    score = confidence + .3*np.minimum(corner/.004, 1)
    score = np.where(eligible, score, -np.inf)
    h, w = score.shape
    selected = []
    for y in range(3, h-3, cell):
        for x in range(3, w-3, cell):
            patch = score[y:min(y+cell,h-3), x:min(x+cell,w-3)]
            row, col = np.unravel_index(np.argmax(patch), patch.shape)
            if np.isfinite(patch[row,col]):
                selected.append((x+col,y+row))
    coords = np.asarray(selected)
    x,y = coords.T
    cam = cams[ref]
    local_ray = cam.rays(coords)
    points = cam.center + local_ray / (inv[y,x] * (local_ray @ cam.rotation[2]))[:,None]
    spread = np.sqrt((((invs[:,y,x]-inv[y,x])**2)*weight[:,y,x]).sum(0) /
                     np.maximum(weight[:,y,x].sum(0),1e-9))
    tolerances = np.clip(.026 + spread/inv[y,x]**2, .03, .16)
    observations = np.concatenate([[np.ones(len(points), bool)], chosen[:,y,x]], axis=0).T
    return dict(points=points, xy=coords, ref=ref, obs=observations,
                cameras=np.r_[ref, others], tolerance=tolerances, score=confidence[y,x])


def triangulate(tracks, camera, max_px, max_edge):
    """Reference-image Delaunay filtered by 3D edges and trusted sightlines."""
    p = tracks["points"]
    faces = Delaunay(tracks["xy"]).simplices
    a,b,c = np.moveaxis(p[faces],1,0)
    edges = np.stack((np.linalg.norm(a-b,axis=1),np.linalg.norm(b-c,axis=1),
                      np.linalg.norm(c-a,axis=1)),axis=1)
    q = tracks["xy"][faces]
    p_edge = np.linalg.norm(q - np.roll(q,1,axis=1), axis=-1).max(1)
    double_area = np.linalg.norm(np.cross(b-a,c-a),axis=1)
    quality = np.sqrt(12)*double_area/np.maximum((edges**2).sum(1), 1e-9)
    accept = (p_edge <= max_px) & (edges.max(1) <= max_edge) & (quality > .08)
    possible = (p_edge <= max_px*1.32) & (edges.max(1) <= max_edge*1.16) & (quality > .08)
    return faces[accept], faces[possible & ~accept]


def compact(vertices, faces):
    """Remove orphan vertices and return old-to-new indices (-1 for orphans)."""
    used = np.unique(faces)
    mapping = np.full(len(vertices), -1, int)
    mapping[used] = np.arange(len(used))
    return vertices[used], mapping[faces], mapping


def nearest_track_faces(model, tracks):
    # Initialization points often lie on vertices. Ties get a stable face ID.
    triangles = model.vertices[model.faces]
    bary = np.array([[1,0,0],[0,1,0],[0,0,1],[.5,.5,0],[0,.5,.5],
                     [.5,0,.5],[1/3,1/3,1/3]])
    candidates = np.einsum("pv,fvc->fpc", bary, triangles).reshape(-1, 3)
    d, ids = cKDTree(candidates).query(tracks["points"])
    return np.where(d < .25, ids//len(bary), -1)
