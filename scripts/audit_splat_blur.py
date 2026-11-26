#!/usr/bin/env python3
"""Read-only native 3DGS scale/footprint ablation at saved training cameras.

Run from the repository root:
  python -m scripts.audit_splat_blur --model MODEL --iteration -1 --views 12
  python -m scripts.audit_splat_blur --model MODEL --image 00001.jpg --roi 0 .5 1 1
  python -m scripts.audit_splat_blur --model MODEL --history-only
All world thresholds are Gaussian standard deviations in scene coordinates.
No PLY, optimizer, camera calibration or training data is modified.
"""
import argparse
import csv
from copy import copy
from html import escape
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def quantiles(x):
    if not len(x):
        return {}
    return dict(zip(("p50", "p90", "p95", "p99", "max"),
                    map(float, np.percentile(x, [50, 90, 95, 99, 100]))))


def stats_from_ply(path, scale_limits, ratio_limits):
    from plyfile import PlyData
    v = PlyData.read(str(path), mmap="r")["vertex"].data
    s = np.stack([v["scale_"+str(i)] for i in range(3)], axis=1).astype(np.float64)
    logits = np.asarray(v["opacity"], dtype=np.float64)
    if not len(s) or not np.isfinite(s).all() or not np.isfinite(logits).all():
        raise ValueError(f"Empty or nonfinite Gaussian checkpoint: {path}")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        largest = np.exp(s.max(axis=1))
        ratio = np.exp(s.max(axis=1)-s.min(axis=1))
        alpha = 1/(1+np.exp(-np.clip(logits, -700, 700)))
    if not np.isfinite(largest).all() or not np.isfinite(ratio).all() or (largest <= 0).any():
        raise ValueError(f"Invalid activated scale or anisotropy in {path}")
    def counts(mask):
        return dict(count=int(mask.sum()), fraction=float(mask.mean()),
                    opacity_parameter_sum_fraction=float(alpha[mask].sum()/max(alpha.sum(), 1e-30)),
                    count_opacity_gt_01=int((mask & (alpha > .1)).sum()))
    groups = {f"scale_gt_{x:g}":counts(largest > x) for x in scale_limits}
    groups.update({f"ratio_gt_{x:g}":counts(ratio > x) for x in ratio_limits})
    return dict(ply=str(path), points=len(s), max_sigma=quantiles(largest),
                anisotropy=quantiles(ratio), opacity=quantiles(alpha),
                max_sigma_opacity_gt_01=quantiles(largest[alpha > .1]), groups=groups)


def history(model, step, all_steps, scales, ratios):
    candidates = sorted((int(p.parent.name.split("_")[-1]), p)
                        for p in (model/"point_cloud").glob("iteration_*/point_cloud.ply")
                        if p.parent.name.split("_")[-1].isdigit())
    results = []
    for n,p in candidates:
        if all_steps or n == step:
            print(f"Inspecting scales at iteration {n:,}", flush=True)
            results.append(dict(iteration=n, **stats_from_ply(p, scales, ratios)))
    return results


def densification_history(model):
    path = model/"densification.jsonl"
    if not path.is_file():
        return dict(available=False)
    events = []
    malformed = 0
    fields = ("cloned_parents", "split_parents", "pruned", "prune_screen_candidates",
              "prune_world_candidates", "prune_opacity_candidates", "eligible_clone_parents",
              "eligible_split_parents")
    with path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or "iteration" not in row:
                    raise ValueError("Missing iteration")
                events.append(row)
            except (ValueError, TypeError):
                malformed += 1
    totals = {k:sum(e[k] for e in events if isinstance(e.get(k),(int,float)))
              for k in fields if any(isinstance(e.get(k),(int,float)) for e in events)}
    return dict(available=True, events=len(events), malformed_lines=malformed,
                totals=totals, first=events[0] if events else None,
                last=events[-1] if events else None)


def select_views(names, requested, count):
    names = sorted(names)
    if requested:
        if len(set(requested)) != len(requested) or not set(requested) <= set(names):
            raise ValueError("Requested images must be unique exact names in the saved training split")
        return requested
    if not names or count < 1:
        raise ValueError("No training cameras or invalid view count")
    return [names[i] for i in np.linspace(0, len(names)-1, min(len(names),count), dtype=int)]


def variants(scales, ratios, radii):
    return [("baseline", None, None)] + [
        (f"{kind}_gt_{limit:g}", kind, limit)
        for kind, limits in (("scale",scales),("ratio",ratios),("radius",radii))
        for limit in limits]


def subset_model(model, keep):
    """Exact in-memory subset; shared calibration and exposure, no optimizer."""
    result = copy(model)
    for name in ("_xyz","_features_dc","_features_rest","_opacity","_scaling","_rotation"):
        setattr(result, name, getattr(model,name)[keep].contiguous())
    return result


def region_metrics(pred, gt, alpha, support):
    """RGB MSE/PSNR and first-difference error on pairs inside the same ROI."""
    import torch
    err = (pred-gt).square().mean(0)
    mse = float(err[support].mean())
    pred_sum = target_sum = diff_sum = 0.
    count = 0
    for axis in (1,2):
        p = pred.diff(dim=axis)
        t = gt.diff(dim=axis)
        valid = (support[1:] & support[:-1]) if axis == 1 else (support[:,1:] & support[:,:-1])
        if valid.any():
            pred_sum += float(p[:,valid].abs().sum())
            target_sum += float(t[:,valid].abs().sum())
            diff_sum += float((p[:,valid]-t[:,valid]).square().sum())
            count += int(valid.sum())*3
    return dict(psnr_db=-10*math.log10(max(mse,1e-30)),
                gradient_rmse=math.sqrt(diff_sum/count) if count else None,
                gradient_magnitude_ratio=pred_sum/target_sum if target_sum > 1e-12 else None,
                mean_alpha=float(alpha[support].mean()),
                low_alpha_fraction=float((alpha[support] < .5).float().mean()))


def save_image(t, path):
    from PIL import Image
    a = t.detach().clamp(0,1).mul(255).round().byte().cpu().numpy()
    if a.ndim == 3:
        a = a.transpose(1,2,0)
    Image.fromarray(a).save(path)


def save_report(output, report, rows):
    from scripts.view_comparisons import atomic_json
    atomic_json(output/"report.json", dict(report, rows=rows))
    if not rows:
        return
    with (output/"metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)
    metric_keys = ("psnr_db","gradient_rmse","gradient_magnitude_ratio","mean_alpha",
                   "low_alpha_fraction","removed_baseline_mass_fraction")
    summary = {}
    for name, rr in grouped.items():
        summary[name] = {k:float(np.mean([r[k] for r in rr if r[k] is not None]))
                         if any(r[k] is not None for r in rr) else None for k in metric_keys}
    atomic_json(output/"summary.json", summary)
    def fmt(v):
        return "n/a" if v is None else f"{v:.4f}"
    sections = ["<h2>Mean across selected views (equal view weight)</h2><table><tr><th>Variant</th>"+
                "".join(f"<th>{k}</th>" for k in metric_keys)+"</tr>"]
    for name, values in summary.items():
        sections.append(f"<tr><td>{escape(name)}</td>"+"".join(f"<td>{fmt(values[k])}</td>" for k in metric_keys)+"</tr>")
    sections.append("</table>")
    for name in dict.fromkeys(r["image"] for r in rows):
        rr = [r for r in rows if r["image"] == name]
        sections.append(f"<h2>{escape(name)}</h2><div class='cards'>")
        gt=escape(rr[0]["gt"], quote=True)
        sections.append(f"<article><b>Ground truth</b><a href='{gt}'><img src='{gt}' loading='lazy'></a></article>")
        for row in rr:
            sections.append(f"<article><b>{escape(row['variant'])}</b><p>PSNR {fmt(row['psnr_db'])}; "
                            f"gradient RMSE {fmt(row['gradient_rmse'])}; alpha {fmt(row['mean_alpha'])}; "
                            f"removed mass {fmt(row['removed_baseline_mass_fraction'])}</p>")
            for key,label in (("prediction","RGB"),("error","4x absolute error"),("alpha","Accumulated opacity"),
                              ("removed_mass","Removed original compositing mass")):
                if row[key]:
                    p=escape(row[key],quote=True)
                    sections.append(f"<small>{label}</small><a href='{p}'><img src='{p}' loading='lazy'></a>")
            sections.append("</article>")
        sections.append("</div>")
    (output/"index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Gaussian scale and blur audit</title>"
        "<style>body{font:15px system-ui;margin:24px;background:#17191d;color:#eee}table{border-collapse:collapse}"
        "td,th{border:1px solid #555;padding:7px}img{width:100%;height:auto;display:block}"
        ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:12px}"
        "article{background:#252931;padding:10px}a{color:#9dc8ff}</style>"
        "<h1>Gaussian scale and blur audit</h1>"
        "<p>Exact subset renders, with unchanged camera and rasterizer. White alpha means covered. "
        "Removed mass is the sum of original T_i alpha_i weights for removed splats, against black; "
        "it is not the opacity after removal. Gradient RMSE measures detail agreement, not just sharpness. "
        "All metric values use the selected ROI; images display the full frame.</p>"
        "<p><a href='report.json'>Report JSON</a> | <a href='metrics.csv'>Per-view metrics</a> | "
        "<a href='summary.json'>Summary JSON</a></p>"+"".join(sections))


def render_audit(args, spec, step, ply, output, report):
    import torch
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from scripts.render_views_worker import load_record_camera
    from scripts.view_comparisons import image_id
    from scripts import pose_options
    from scripts.occlusion_options import inference_settings
    cfg = spec["cfg"]
    if inference_settings(cfg)["occlusion_mode"] != "off":
        raise ValueError("This audit isolates native compositing; use a checkpoint with inference culling off")
    if cfg.get("train_test_exp"):
        raise ValueError("Half-image exposure-training checkpoints require a different metric protocol")
    if args.antialiasing is None and "antialiasing" not in cfg:
        raise ValueError("Legacy config lacks AA setting: specify --antialiasing or --no-antialiasing")
    aa = args.antialiasing if args.antialiasing is not None else cfg["antialiasing"]
    pipe = SimpleNamespace(antialiasing=bool(aa), convert_SHs_python=False,
                           compute_cov3D_python=False, debug=False, occlusion_mode="off")
    camera_args = SimpleNamespace(resolution=cfg.get("resolution",-1),data_device="cpu",
                                  train_test_exp=False,loss_mask_dir=cfg.get("loss_mask_dir",""),
                                  colmap_principal_point=cfg.get("colmap_principal_point",False))
    spec = pose_options.apply_overrides(spec,args)
    cache = pose_options.load_cache(spec,step)
    names = select_views(spec["splits"]["train"], args.image, args.views)
    model = GaussianModel(cfg.get("sh_degree",3))
    model.load_ply(str(ply))
    bg = model.get_xyz.new_full((3,),float(cfg.get("white_background",False)))
    black = torch.zeros_like(bg)
    with torch.no_grad():
        s = model.get_scaling
        largest = s.amax(dim=1)
        ratio = largest/s.amin(dim=1).clamp_min(1e-30)
        if not bool(torch.isfinite(largest).all() and torch.isfinite(ratio).all()):
            raise ValueError("Nonfinite activated Gaussian scales")
        del s
        vv = variants(args.scale,args.anisotropy,args.radius_px)
        report.update(antialiasing=bool(aa), resolution=camera_args.resolution,
                      camera_poses=pose_options.protocol(spec), selected_images=names,
                      roi=args.roi, exposure=False, occlusion="off", variants=[v[0] for v in vv])
        rows = []
        for index,name in enumerate(names):
            print(f"[{index+1}/{len(names)}] {name}",flush=True)
            cam=pose_options.apply_camera(load_record_camera(camera_args,spec["records"][name],index),cache)
            gt=cam.original_image[:3].to(bg.device).clamp(0,1)
            h,w=gt.shape[-2:]
            x0,y0,x1,y1=args.roi
            support=torch.zeros((h,w),dtype=torch.bool,device=bg.device)
            support[int(y0*h):math.ceil(y1*h),int(x0*w):math.ceil(x1*w)]=True
            if getattr(cam,"loss_mask",None) is not None:
                support &= cam.loss_mask[0].to(device=bg.device,dtype=torch.bool)
            if not support.any():
                raise ValueError(f"Empty ROI for {name}")
            folder=output/image_id(name)
            folder.mkdir(exist_ok=True)
            save_image(gt,folder/"gt.png")
            def rel(path):
                return str(path.relative_to(output))
            def draw(m):
                result=render(cam,m,pipe,bg)
                pred=result["render"]
                radii=result["radii"]
                return pred,radii
            pred0,radii0=draw(model)
            if not torch.isfinite(pred0).all():
                raise ValueError(f"Nonfinite baseline for {name}")
            ones=model.get_xyz.new_ones((len(largest),3))
            alpha0=render(cam,model,pipe,black,override_color=ones)["render"][0].clone()
            del ones
            base_mass=float(alpha0[support].sum())
            def exclusion(kind,limit):
                if kind is None:
                    return torch.zeros_like(largest,dtype=torch.bool)
                return {"scale":largest,"ratio":ratio,"radius":radii0}[kind] > limit
            # RGB tags give up to three independent compositing-mass maps in a
            # single baseline rasterization; tags affect color only, not alpha.
            for begin in range(0,len(vv),3):
                batch=vv[begin:begin+3]
                masks=[exclusion(k,l) for _,k,l in batch]
                tags=model.get_xyz.new_zeros((len(largest),3))
                for channel,mask in enumerate(masks):
                    tags[:,channel]=mask
                mass=render(cam,model,pipe,black,override_color=tags)["render"]
                del tags
                for channel,((label,kind,limit),removed) in enumerate(zip(batch,masks)):
                    if kind is None or not removed.any():
                        pred,alpha=pred0,alpha0
                    else:
                        filtered=subset_model(model,~removed)
                        pred,_=draw(filtered)
                        ones=filtered.get_xyz.new_ones((len(filtered.get_xyz),3))
                        alpha=render(cam,filtered,pipe,black,override_color=ones)["render"][0].clone()
                        del filtered,ones
                    if not torch.isfinite(pred).all() or not torch.isfinite(alpha).all():
                        raise ValueError(f"Nonfinite render: {name} {label}")
                    m=region_metrics(pred,gt,alpha,support)
                    if not torch.isfinite(mass[channel]).all():
                        raise ValueError("Nonfinite compositing tag map")
                    removed_mass=float(mass[channel][support].sum())/base_mass if base_mass>1e-12 else None
                    files={key:folder/f"{label}_{key}.png" for key in ("prediction","error","alpha","removed_mass")}
                    for key,tensor in (("prediction",pred),("error",(pred-gt).abs()*4),
                                       ("alpha",alpha),("removed_mass",mass[channel])):
                        save_image(tensor,files[key])
                    rows.append(dict(image=name,variant=label,threshold_kind=kind,threshold=limit,
                                     removed_count=int(removed.sum()),removed_raster_count=int((removed & (radii0>0)).sum()),
                                     baseline_radius_p99_px=float(torch.quantile(radii0[radii0>0].float(),.99)) if (radii0>0).any() else None,
                                     roi_pixels=int(support.sum()), **m,
                                     removed_baseline_mass_fraction=removed_mass,
                                     gt=rel(folder/"gt.png"), **{k:rel(v) for k,v in files.items()}))
                    print(f"  {label}: PSNR={m['psnr_db']:.2f} gradient RMSE={m['gradient_rmse']} "
                          f"alpha={m['mean_alpha']:.3f} removed mass={removed_mass}",flush=True)
                    del pred,alpha
                del mass,masks
            del cam,gt,pred0,radii0,alpha0,support
            save_report(output,report,rows)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--iteration",type=int,default=-1)
    p.add_argument("--output",type=Path)
    p.add_argument("--image",action="append",help="Exact saved training image name; repeat as needed")
    p.add_argument("--views",type=int,default=12)
    p.add_argument("--roi",nargs=4,type=float,default=[0,0,1,1],metavar=("X0","Y0","X1","Y1"),
                   help="Fractional rectangle at saved training resolution; identical for selected views")
    p.add_argument("--scale",nargs="*",type=float,default=[.5,.2,.1])
    p.add_argument("--anisotropy",nargs="*",type=float,default=[20.])
    p.add_argument("--radius-px",nargs="*",type=float,default=[100.,200.])
    p.add_argument("--antialiasing",action=argparse.BooleanOptionalAction,default=None)
    p.add_argument("--camera-poses",choices=("original","refined"),default="original")
    p.add_argument("--history",action="store_true",help="Inspect all saved PLY checkpoints")
    p.add_argument("--history-only",action="store_true",help="CPU scale and densification statistics; no rendering")
    args=p.parse_args()
    for limits in (args.scale,args.anisotropy,args.radius_px):
        if len(set(limits))!=len(limits) or any(not math.isfinite(x) or x<=0 for x in limits):
            p.error("Thresholds must be finite, positive, and unique within each family")
    x0,y0,x1,y1=args.roi
    if not (0<=x0<x1<=1 and 0<=y0<y1<=1) or args.views<1:
        p.error("Invalid ROI or view count")
    from scripts.eval_all_frames import choose_iteration,read_config
    model=args.model.expanduser().resolve()
    cfg=read_config(model/"cfg_args")
    if cfg.get("model_type","3dgs")!="3dgs":
        p.error("Native 3DGS checkpoints only")
    step,ply=choose_iteration(model,args.iteration)
    output=(args.output or model/"scale_audit"/f"iteration_{step}").expanduser().resolve()
    # Never overwrite an existing audit or a checkpoint directory.
    if output==model or output==ply.parent or (model/"point_cloud") in output.parents:
        p.error("Choose a separate audit output directory")
    output.mkdir(parents=True,exist_ok=False)
    report=dict(model=str(model),iteration=step,checkpoint=str(ply),
                scale_units="Gaussian standard deviation in scene coordinates",
                statistic_note="Opacity parameter sums are not compositing contribution. Radius means native raster bounding radius, not physical scale.",
                history=history(model,step,args.history or args.history_only,args.scale,args.anisotropy),
                densification=densification_history(model))
    save_report(output,report,[])
    if not args.history_only:
        from scripts.view_comparisons import model_info
        render_audit(args,model_info(model),step,ply,output,report)
    print(f"Audit: {output}",flush=True)


if __name__=="__main__":
    main()
