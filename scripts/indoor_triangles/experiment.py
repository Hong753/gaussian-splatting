"""Fit the candidate and matched arms from fitting RGB and fixed cameras."""

import json
from pathlib import Path
import resource
from time import perf_counter

import numpy as np

from .data import compact, load_views, reconstruct_tracks, triangulate
from .model import area, connect, disconnect, initial_mesh, Mesh
from .render import edge_neighbors, render
from .train import (compare_edit, coverage_trial, geometry_summary,
                    support_faces, train)


def save_json(file, data):
    Path(file).write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")


def save_arm(folder, arm, mesh, log):
    file = folder/f"{arm}.npz"
    mesh.write(file)
    log.update(geometry_summary(mesh),checkpoint_bytes=file.stat().st_size,
               peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               peak_vram_bytes=None)
    save_json(folder/f"{arm}.json",log)
    print(f"{arm}  {len(mesh.faces)} faces  {log.get('wall_s',0):.2f}s",flush=True)


def train_branch(seed_mesh, label, folder, fit, tracks, cfg, seed, **options):
    mesh = seed_mesh.clone()
    log = train(mesh,*fit,tracks,cfg,cfg["steps"],seed,**options)
    save_arm(folder,label,mesh,log)
    return mesh


def initialize(fit, cfg, folder, dense, bg):
    start = perf_counter()
    cell = cfg["dense_cell_px"] if dense else cfg["sparse_cell_px"]
    tracks = reconstruct_tracks(*fit,cell,dense)
    return initialize_tracks(fit,cfg,folder,tracks,bg,start)


def initialize_tracks(fit, cfg, folder, tracks, bg, start=None):
    start = start if start is not None else perf_counter()
    rgb,cams = fit
    cam = cams[tracks["ref"]]
    good,pool = triangulate(tracks,cam,cfg["max_edge_px"],cfg["max_edge_m"])
    if not len(good):
        raise ValueError("No initial faces passed the triangulation gates.")
    v,f,_ = compact(tracks["points"],good)
    template = initial_mesh(v,f,cam,bg)
    soup = disconnect(template)
    topo1 = perf_counter()-start
    seed = cfg["seed"]
    warm = train(soup,*fit,tracks,cfg,cfg["warmup_steps"],seed)
    save_arm(folder,"warmup_soup",soup,warm)
    start = perf_counter()
    base = connect(soup,template,rgb[tracks["ref"]],cfg["atlas_hw"])
    topo2 = perf_counter()-start
    log = train(base,*fit,tracks,cfg,cfg["connected_steps"],seed+1)
    start = perf_counter()
    sample_start = render.subsamples
    supported = support_faces(base,tracks,cams,cfg["conflict_m"],cfg["hard_support_min_anchors"],
                              cfg["hard_support_max_centroid_m"])
    log.update(topology_s=topo1+topo2, support_test_s=perf_counter()-start,
               support_test_render_subsamples=render.subsamples-sample_start,
               harden_eligible=int(supported.sum()),track_points=len(tracks["points"]),
               track_observations=int(tracks["obs"].sum()),reserved_coverage_faces=len(pool))
    save_arm(folder,"shared_start",base,log)
    np.savez_compressed(folder/"fitting_tracks.npz",**tracks)
    np.save(folder/"supported_faces.npy",supported)
    return base,supported,tracks,pool


def selective(base, eligible, pool, fit, dev, tracks, cfg):
    start,samples = perf_counter(),render.subsamples
    seed = cfg["seed"]
    mesh = base.clone()
    edits = []
    candidates = np.flatnonzero(eligible)
    ordered = candidates[np.argsort(-area(mesh)[candidates])]
    # Evaluate the larger faces first. Other faces remain provisional.
    for ordinal,original in enumerate(ordered[:cfg["local_candidates"]]):
        # No deletions are applied in this loop, so original IDs remain stable.
        trials,log = compare_edit(mesh,original,fit,dev,tracks,cfg,seed+100+ordinal,
                                  ("retain","harden","delete"))
        chosen = "retain"
        if log["harden"]["early_blockers"] == 0 and log["harden"]["score"] <= (
                log["retain"]["score"]+cfg["edit_tolerance"]):
            chosen = "harden"
        mesh = trials[chosen]
        edits.append(dict(face=int(original),accepted=chosen,trials=log))
    if len(candidates):
        f = int(ordered[0])
        candidates,log = compare_edit(mesh,f,fit,dev,tracks,cfg,seed+117,("texture","split"))
        texture,split = log["texture"],log["split"]
        # A split needs separate held-out color improvement and in-fit parallax evidence.
        tw = texture["fit"]["curve"][-1]
        sw = split["fit"]["curve"][-1]
        parallax = tw["count"]["warp"] > 0 and sw["count"]["warp"] > 0 and (
            sw["loss"]["warp"] < .95*tw["loss"]["warp"])
        chosen = "split" if parallax and split["early_blockers"] == 0 and (
            split["score"] < texture["score"]-.001) else "texture"
        mesh = candidates[chosen]
        edits.append(dict(face=f,accepted=chosen,positive_parallax_gate=bool(parallax),trials=log))
    if len(pool):
        cam = fit[1][tracks["ref"]]
        xyz = tracks["points"][pool[0]]
        candidates,log = coverage_trial(mesh,xyz,np.arange(3),cam,fit,dev,tracks,cfg,seed+118)
        if log is not None:
            keep = (log["retain"]["score"] < log["delete"]["score"]-.001 and
                    log["retain"]["free"] <= log["delete"]["free"]+.003)
            chosen = "retain" if keep else "delete"
            if keep:
                mesh = candidates[chosen]
            edits.append(dict(action="coverage_proposal",accepted=chosen,trials=log))
    extra_s=perf_counter()-start
    extra_rays=render.subsamples-samples
    log = train(mesh,*fit,tracks,cfg,cfg["steps"],seed+2)
    log.update(edit_wall_s=extra_s,edit_render_subsamples=extra_rays,edits=edits)
    return mesh,log


def wrong_depth(base, fit, tracks, cfg, folder):
    cam = fit[1][tracks["ref"]]
    mesh = base.clone(vertices=cam.center+(base.vertices-cam.center)*.88)
    save_arm(folder,"wrong_depth_initial",mesh,{})
    train_cfg = cfg | {"geometry_lr_m":.009, "max_phase_displacement_m":.94}
    log = train(mesh,*fit,tracks,train_cfg,cfg["stress_steps"],cfg["seed"]+30)
    save_arm(folder,"wrong_depth_refit",mesh,log)


def false_closure(base, fit, dev, tracks, cfg, folder):
    from scipy.spatial import Delaunay
    cam = fit[1][tracks["ref"]]
    d = cam.local(tracks["points"])[:,2]
    far = d > np.quantile(d,.86)
    q = tracks["xy"][far]
    if len(q) < 3 or np.linalg.matrix_rank(q-q.mean(0)) < 2:
        raise ValueError("Fitting tracks do not provide a distinct distant stress region.")
    simplices = Delaunay(q).simplices
    a,b,c = np.moveaxis(q[simplices],1,0)
    extent = np.abs((b[:,0]-a[:,0])*(c[:,1]-a[:,1])-(c[:,0]-a[:,0])*(b[:,1]-a[:,1]))
    # Construct an intentionally wrong planar closure from the distant track hull.
    q = q[simplices[extent.argmax()]]
    ray = cam.rays(q)
    near = np.median(d[d<np.quantile(d,.65)])
    v = cam.center + ray*(near/(ray@cam.rotation[2]))[:,None]
    ids = np.arange(3)+len(base.vertices)
    faces = np.vstack((base.faces,ids))
    mesh = base.clone(vertices=np.vstack((base.vertices,v)),faces=faces,
                      uv=np.concatenate((base.uv,(q/[cam.width-1,cam.height-1])[None])),
                      opacity=np.r_[base.opacity,.95],hard=np.r_[base.hard,True],
                      transmitting=np.r_[base.transmitting,False],neighbors=edge_neighbors(faces))
    save_arm(folder,"false_closure_initial",mesh,{})
    f = len(faces)-1
    trial,log = compare_edit(mesh,f,fit,dev,tracks,cfg,cfg["seed"]+31,("retain","reopen","delete"))
    allowed = [k for k in log if k!="retain" or log[k]["early_blockers"]==0]
    chosen = min(allowed,key=lambda name: log[name]["score"])
    edit = trial[chosen]
    save_arm(folder,"false_closure_edited",edit,dict(accepted=chosen,trials=log,
             wall_s=sum(s["fit"]["wall_s"] for s in log.values())))


def run(data, folder, cfg, seed, dense_only=False, tracks_path=None, minimal=False):
    data,folder = Path(data),Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/"config.json").exists():
        raise FileExistsError(f"Use a fresh output directory: {folder}")
    cfg = cfg | {"seed":seed}
    save_json(folder/"config.json",cfg)
    fit,dev = load_views(data/"fit.npz"),load_views(data/"edit.npz")
    bg = np.array([.07,.08,.10])
    if tracks_path:
        with np.load(tracks_path) as points:
            loaded = {key: points[key] for key in points.files}
        base,eligible,tracks,pool = initialize_tracks(fit,cfg,folder,loaded,bg)
    else:
        base,eligible,tracks,pool = initialize(fit,cfg,folder,dense_only,bg)
    train_branch(base,"connected_soft",folder,fit,tracks,cfg,seed+2)
    hard = base.clone(hard=eligible)
    train_branch(hard,"connected_hard",folder,fit,tracks,cfg,seed+2)
    if dense_only or minimal:
        return
    global_mesh=base.clone(hard=~base.transmitting)
    train_branch(global_mesh,"global_hard",folder,fit,tracks,cfg,seed+2)
    private=disconnect(base).clone(neighbors=base.neighbors)
    train_branch(private,"private_soft",folder,fit,tracks,cfg,seed+2)
    thin_cfg=cfg | {"weights": cfg["weights"] | {"warp":0.,"free":0.,"anchor":0.}}
    train_branch(hard,"hard_no_warp_tracks",folder,fit,tracks,thin_cfg,seed+2)
    mesh,log=selective(base,eligible,pool,fit,dev,tracks,cfg)
    save_arm(folder,"selective",mesh,log)
    shader_cfg=cfg | {"steps":cfg["shader_steps"]}
    train_branch(mesh,"fixed_mesh_atlas",folder,fit,tracks,shader_cfg,seed+3,fixed_geometry=True)
    train_branch(mesh.clone(shader="gaussian"),"fixed_mesh_gaussian",folder,fit,tracks,
                 shader_cfg,seed+3,fixed_geometry=True)
    if tracks_path is None:
        wrong_depth(base,fit,tracks,cfg,folder)
        false_closure(hard,fit,dev,tracks,cfg,folder)
