"""Fitting-only CPU optimizer and reproducible local validation trials."""

from pathlib import Path
from time import perf_counter

import numpy as np

from .data import nearest_track_faces
from .model import (anchor_errors, area, fit_evidence, patch_correspondences, regularizers,
                    smooth_edges, split_face, subset_faces, warp_errors)
from .render import edge_neighbors, normals, OFFSETS, plane_hits, render, screen_edges

NAME = ("rgb","warp","free","anchor","bend","quality")


def mean(values):
    return float(np.mean(values)) if len(values) else 0.


def chunks(items, limit=256):
    for start in range(0,len(items),limit):
        yield items[start:start+limit]


def free_space(model, camera, coords, depth, epsilon, tau, scale):
    pred = render(model,camera,coords,tau)
    return (pred["weight"]*np.maximum(depth[:,None]-epsilon[:,None]-pred["depth"],0)).sum(1)/scale


def trusted_conflict(model, tracks, cams, threshold=.16, face=None):
    """A committed edit is vetoed if the exact hit blocks any trusted track."""
    visits = blocked = 0
    for j,i in enumerate(tracks["cameras"]):
        keep = tracks["obs"][:,j]
        xyz = tracks["points"][keep]
        cam = cams[i]
        for ids in chunks(np.arange(len(xyz))):
            observed = xyz[ids]
            pred = render(model,cam,cam.project(observed))
            hard = pred["hard_hit"]
            depth = pred["depth"][np.arange(len(ids)),hard]
            bad = (hard>=0) & (np.linalg.norm(observed-cam.center,axis=1)-depth >
                               np.maximum(threshold,tracks["tolerance"][keep][ids]))
            if face is not None:
                bad &= hard==face
            visits += len(ids)
            blocked += int(bad.sum())
    return blocked,visits


def support_faces(model, tracks, cams, threshold, minimum_anchors, maximum_unsupported_distance):
    a,b,c = np.moveaxis(model.vertices[model.faces],1,0)
    normal = normals(model.vertices,model.faces)
    xyz = tracks["points"]
    distance = np.abs(((xyz[:,None]-a)*normal).sum(-1))
    near = np.linalg.norm(xyz[:,None]-(a+b+c)/3,axis=-1)
    radius = .65*np.max(np.linalg.norm(model.vertices[model.faces]-(a+b+c)[:,None]/3,axis=-1),axis=1)+.09
    strong = (distance < .065) & (near < radius)
    visible = (strong.T.astype(int) @ tracks["obs"].astype(int)) > 0
    centers = np.asarray([cams[i].center for i in tracks["cameras"]])
    translated = np.linalg.norm(centers[:,None]-centers[None],axis=-1) > .20
    valid_views = (visible[:,None,:] & visible[:,:,None] & translated).any(axis=(1,2))
    centroid = (a+b+c)/3
    interior_support = (np.linalg.norm(xyz[:, None]-centroid,axis=-1) <
                        maximum_unsupported_distance) & (distance < .065)
    possible = (valid_views & (strong.sum(0) >= minimum_anchors) &
                interior_support.any(0) & ~model.transmitting)
    # Check each possible face in the same pass rather than letting an opaque
    # face mask conflicting evidence from another face along the same ray.
    bad = np.zeros(len(a),bool)
    hard = model.clone(hard=possible.copy())
    for j,i in enumerate(tracks["cameras"]):
        use = tracks["obs"][:,j]
        xyz, eps = tracks["points"][use],tracks["tolerance"][use]
        cam = cams[i]
        for ids in chunks(np.arange(len(xyz))):
            p = xyz[ids]
            d,_ = plane_hits(hard.vertices,hard.faces,cam.center,cam.rays(cam.project(p)))
            edges,_ = screen_edges(cam,cam.project(p),hard.vertices[hard.faces],.05)
            conflicts = (d>.05) & (edges.min(-1)>=-1e-8) & (
                np.linalg.norm(p-cam.center,axis=1)[:,None]-d > np.maximum(threshold,eps[ids,None]))
            bad |= conflicts.any(0)
    return possible & ~bad


def adam(param, grad, first, second, step, rate):
    first *= .9
    first += .1*grad
    second *= .99
    second += .01*grad*grad
    param -= rate * (first/(1-.9**step)) / (np.sqrt(second/(1-.99**step))+1e-8)


def draw(rng, cam, count, region):
    if region is None:
        lo,hi = np.array([0,0]), np.array([cam.width,cam.height])
    else:
        lo,hi = region(cam)
    x = rng.integers(lo[0],hi[0],count)
    y = rng.integers(lo[1],hi[1],count)
    return np.stack((x,y),axis=1)


def train(model, images, cams, tracks, cfg, steps, seed, logfile=None, assignments=None,
          region=None, active=None, fixed_geometry=False, fixed_color=False):
    starts, start = render.subsamples, perf_counter()
    rng = np.random.default_rng(seed)
    scale, cap = cfg["track_scale_m"], cfg["anchor_cap"]
    w = np.array([cfg["weights"][k] for k in NAME])
    assignments, free = fit_evidence(model,tracks,cams,assignments)
    smooth = smooth_edges(model)
    first = [np.zeros_like(p) for p in (model.vertices,model.opacity,model.colors)]
    second = [p.copy() for p in first]
    active = np.ones(len(model.vertices),bool) if active is None else np.asarray(active)
    opacity_active = active[model.faces].any(1) & ~model.hard
    log, totals = [],np.zeros(len(NAME),int)
    eps = cfg["geometry_probe_m"]
    initial = model.vertices.copy()
    for step in range(1,steps+1):
        i,j = rng.choice(len(cams),2,replace=False)
        camera = cams[i]
        pix = draw(rng,camera,cfg["batch_pixels"],region)
        # Keep subpixel samples identical across ± geometry probes.
        offsets = OFFSETS + rng.uniform(-.24,.24,(1,2))
        coords = (pix[:,None]+offsets).reshape(-1,2)
        rgb = images[i][pix[:,1],pix[:,0]]
        pair = patch_correspondences(model,images,cams,i,j,pix[:cfg["warp_patches"]],
                                     cfg["tau_px"]) if w[1] else None
        cam_id, xy, depth, tolerance = free[step%len(free)]
        indices = rng.choice(len(xy),min(cfg["free_rays"],len(xy)),replace=False)
        freedata = xy[indices],depth[indices],tolerance[indices]

        def objective(want_color=False):
            pred = render(model,camera,coords,cfg["tau_px"],color_basis=want_color)
            delta = pred["rgb"].reshape(-1,len(offsets),3).mean(1)-rgb
            robust = np.sqrt(delta**2 + .015**2)
            losses = [robust-.015,
                      warp_errors(model,images,cams,pair) if w[1] else (),
                      free_space(model,cams[cam_id],*freedata,cfg["tau_px"],scale) if w[2] else (),
                      anchor_errors(model,tracks["points"],assignments,scale,cap) if w[3] else ()]
            losses.extend(regularizers(model,smooth))
            values = np.array([mean(l) for l in losses])
            count = np.array([len(l) for l in losses])
            return values,count,pred["basis"],delta/robust

        values,count,basis,rgb_grad = objective(want_color=True)
        totals += count
        v_grad, a_grad = np.zeros_like(model.vertices),np.zeros_like(model.opacity)
        if not fixed_geometry:
            probes = cfg["geometry_probes"]
            normalizes = eps*2*probes
            for _ in range(probes):
                dv = rng.choice((-1.,1.),model.vertices.shape)*active[:,None]
                da = rng.choice((-1.,1.),model.opacity.shape)*opacity_active
                v, opacity = model.vertices.copy(),model.opacity.copy()
                pair_values = []
                for sign in (1,-1):
                    model.vertices[:] = v+sign*eps*dv
                    model.opacity[:] = (opacity+sign*eps*da).clip(.04,.998)
                    pair_values.append(objective()[0])
                model.vertices[:],model.opacity[:] = v,opacity
                directional = np.dot(w,pair_values[0]-pair_values[1])/normalizes
                v_grad += directional*dv
                a_grad += directional*da
            rate = cfg["geometry_lr_m"]*(1.-.74*(step-1)/max(steps-1,1))
            adam(model.vertices,v_grad,first[0],second[0],step,rate)
            displacement = model.vertices-initial
            length = np.linalg.norm(displacement,axis=1,keepdims=True)
            model.vertices[:] = initial+displacement*np.minimum(1,cfg["max_phase_displacement_m"]/
                                                                np.maximum(length,1e-9))
            adam(model.opacity,a_grad,first[1],second[1],step,cfg["opacity_lr"])
            model.opacity[:] = model.opacity.clip(.05,.995)

        if not fixed_color:
            rows,ids,coef = basis
            g = rgb_grad[rows//len(offsets)]/(len(coords)*3)
            gradient = np.stack([np.bincount(ids.ravel(),weights=(coef*g[:,c,None]).ravel(),
                                            minlength=model.colors.size//3) for c in range(3)],axis=1)
            adam(model.colors,gradient.reshape(model.colors.shape),first[2],second[2],step,cfg["color_lr"])
            model.colors[:] = model.colors.clip(0,1)
        if step == 1 or step == steps or step % cfg["log_every"] == 0:
            log.append(dict(step=step,total=float(values@w),loss=dict(zip(NAME,values.tolist())),
                            count=dict(zip(NAME,count.tolist())),weights=dict(zip(NAME,w.tolist())),
                            wall_s=perf_counter()-start))
    stats = dict(steps=steps, rendered_subsamples=render.subsamples-starts,
                 valid_totals=dict(zip(NAME,totals.tolist())), wall_s=perf_counter()-start,
                 active_geometry_scalars=int(active.sum()*3) if not fixed_geometry else 0,
                 valid_count_units={"rgb":"pixels","warp":"patches","free":"track rays",
                                    "anchor":"points including missing","bend":"edges","quality":"faces"},
                 curve=log)
    if logfile is not None:
        import json
        Path(logfile).write_text(json.dumps(stats,indent=2)+"\n")
    return stats


def edit_region(triangle, margin=5):
    def region(cam):
        pts = cam.project(triangle)
        dims = np.array([cam.width,cam.height])
        lo = np.minimum(np.maximum(np.floor(pts.min(0)).astype(int)-margin,0),dims-1)
        hi = np.clip(np.ceil(pts.max(0)).astype(int)+margin+1,lo+1,dims)
        return lo,hi
    return region


def compare_edit(mesh, face_id, fit, dev, tracks, cfg, seed, actions):
    """Equal local fitting schedules; validation never receives scan geometry."""
    images,cams = fit
    val_images,val_cams = dev
    face = mesh.faces[face_id]
    region = edit_region(mesh.vertices[face])
    assigned = nearest_track_faces(mesh,tracks)
    candidates, logs = {},{}
    for action in actions:
        trial = mesh.clone()
        active = np.isin(np.arange(len(mesh.vertices)),face)
        assoc = assigned.copy()
        fixed_geometry = action=="texture"
        if action == "harden":
            trial.hard[face_id] = True
        elif action == "reopen":
            trial.hard[face_id] = False
        elif action == "delete":
            keep = np.arange(len(trial.faces))!=face_id
            trial,remap = subset_faces(trial,keep)
            assoc = np.where(assoc>=0,remap[assoc.clip(0)],-1)
        elif action == "split":
            trial = split_face(trial,face_id)
            active = np.r_[np.zeros(len(active),bool),True]
        stats = train(trial,images,cams,tracks,cfg,cfg["local_steps"],seed,assignments=assoc,
                      region=region,active=active,fixed_geometry=fixed_geometry)
        val_loss, val_counts = [],[]
        for cam,image in zip(val_cams,val_images):
            lo,hi = region(cam)
            coords = np.stack(np.meshgrid(np.arange(lo[0],hi[0]),np.arange(lo[1],hi[1])),axis=-1)
            coords = coords.reshape(-1,2)
            samples = (coords[:,None]+OFFSETS).reshape(-1,2)
            pred = np.concatenate([render(trial,cam,b,cfg["tau_px"])["rgb"] for b in chunks(samples)])
            delta = pred.reshape(-1,len(OFFSETS),3).mean(1) - image[coords[:,1],coords[:,0]]
            val_loss.append(mean(np.sqrt(delta**2+.015**2)-.015))
            val_counts.append(len(coords))
        anchor = mean(anchor_errors(trial,tracks["points"],assoc,cfg["track_scale_m"],cfg["anchor_cap"]))
        blocked, visits = trusted_conflict(trial,tracks,cams,cfg["conflict_m"],
                                           face=face_id if action in ("harden","retain","reopen") else None)
        track_err = []
        local_track_count = 0
        for i,xy,d,e in fit_evidence(trial,tracks,cams,assoc)[1]:
            lo,hi = region(cams[i])
            take = ((xy>=lo)&(xy<hi)).all(1)
            local_track_count += int(take.sum())
            if take.any():
                track_err.extend(free_space(trial,cams[i],xy[take],d[take],e[take],
                                            cfg["tau_px"],cfg["track_scale_m"]))
        score = (mean(val_loss)+cfg["weights"]["anchor"]*anchor+
                 cfg["weights"]["free"]*mean(track_err))
        candidates[action] = trial
        logs[action] = dict(score=score,rgb=mean(val_loss),anchor=anchor,free=mean(track_err),
                            early_blockers=blocked,track_rays=visits,local_track_rays=local_track_count,
                            validation_pixels=val_counts,fit=stats)
    return candidates,logs


def coverage_trial(mesh, vertices, face, reference, fit, dev, tracks, cfg, seed):
    """Propose a nearby fitting-track face and compare with its absence."""
    from scipy.spatial import cKDTree

    distance, ids = cKDTree(mesh.vertices).query(vertices)
    new = distance > .12
    ids[new] = len(mesh.vertices)+np.arange(new.sum())
    added = mesh.clone(vertices=np.vstack((mesh.vertices,vertices[new])),
                       faces=np.vstack((mesh.faces,ids[face])),
                       uv=np.concatenate((mesh.uv,(reference.project(vertices[face])/
                                                   [reference.width-1,reference.height-1])[None])),
                       opacity=np.r_[mesh.opacity,.88],hard=np.r_[mesh.hard,False],
                       transmitting=np.r_[mesh.transmitting,False])
    if len(set(ids[face])) != 3 or np.any(np.all(np.sort(mesh.faces,axis=1)==np.sort(ids[face]),axis=1)):
        return None,None
    added.neighbors = edge_neighbors(added.faces)
    return compare_edit(added,len(added.faces)-1,fit,dev,tracks,cfg,seed,("retain","delete"))


def geometry_summary(mesh):
    weights = area(mesh)
    return dict(vertices=len(mesh.vertices),faces=len(mesh.faces),
                incident_vertices=len(np.unique(mesh.faces)),geometry_scalars=mesh.vertices.size,
                opacity_scalars=len(mesh.opacity),appearance_bytes=mesh.colors.nbytes,
                opaque_area_m2=float(weights[mesh.hard].sum()),
                transmitting_area_m2=float(weights[mesh.transmitting & ~mesh.hard].sum()),
                provisional_area_m2=float(weights[~(mesh.transmitting|mesh.hard)].sum()))
