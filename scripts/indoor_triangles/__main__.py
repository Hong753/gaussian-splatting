"""Commands for the indoor triangle reference experiment."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .data import load_views
from .evaluate import compare_arm, export, ref_view, render_view, surface_context
from .experiment import run, save_json
from .model import Mesh
from .sanity import run_checks
from .synthetic import generate

DEFAULT=Path(__file__).with_name("synthetic_config.json")


def config(path):
    return json.loads(Path(path).read_text())


def evaluate(data,run_folder,output,reference_mesh=None,reference_masks=None,first_hit=False,
             rgb_only=False,checkpoint_name=None):
    data,run_folder,output=Path(data),Path(run_folder),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    cfg=config(run_folder/"config.json")
    images,cameras=load_views(data/"final.npz")
    if rgb_only:
        refs = None
    elif reference_mesh:
        from .room import reference_views
        vertices,faces,refs=reference_views(reference_mesh,cameras,reference_masks)
    else:
        with np.load(data/"reference.npz") as source:
            reference=tuple(source[k] for k in ("vertices","faces","kinds"))
        vertices,faces,kinds=reference
        refs=[ref_view(reference,cam) for cam in cameras]
        vertices,faces=vertices,faces[kinds!=4]
    target=None if refs is None else surface_context(vertices,faces,cameras,[r[1] for r in refs],cfg)
    table=[]
    checkpoints = sorted(run_folder.glob("*.npz")) if checkpoint_name is None else [run_folder/checkpoint_name]
    for ckpt in checkpoints:
        if ckpt.name=="fitting_tracks.npz":
            continue
        result=compare_arm(ckpt,images,cameras,refs,output,cfg,target,first_hit=first_hit)
        save_json(output/f"{ckpt.stem}.json",result)
        print(ckpt.stem,flush=True)
        views=result["views"]
        doorway=sum(v.get("doorway",{}).get("reference_pixels",0) for v in views)
        curve=config(run_folder/f"{ckpt.stem}.json")
        extra=curve.get("edit_wall_s",0)
        for mode,s in result["surface"].items():
            table.append(dict(arm=ckpt.stem,export=mode,faces=s["exported_faces"],
                              accuracy_m=s.get("accuracy_m"),completeness_m=s.get("completeness_m"),
                              chamfer_l1_m=s.get("chamfer_l1_m"),fscore_5cm=s.get("fscore"),
                              normal_consistency=s.get("normal_consistency"),
                              psnr_db=float(np.mean([v["psnr_db"] for v in views])),
                              ssim=float(np.mean([v["ssim"] for v in views])),
                              opaque_fraction=float(np.mean([v["opaque_subsample_fraction"] for v in views])),
                              mesh_hit_fraction=float(np.mean([v["mesh_hit_fraction"] for v in views])),
                              early_doorway_fraction=sum(v["doorway"]["early"] for v in views)/doorway if doorway else None,
                              missed_doorway_fraction=sum(v["doorway"]["missed"] for v in views)/doorway if doorway else None,
                              appearance_bytes=curve["appearance_bytes"],
                              train_s=curve.get("wall_s",0),extra_edit_s=extra,export_s=result["export_s"]))
    if not table:
        raise ValueError("No trained checkpoints found.")
    with (output/"paired_results.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,list(table[0]))
        writer.writeheader()
        writer.writerows(table)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    p=sub.add_parser("make-synthetic")
    p.add_argument("--out",type=Path,required=True)
    p=sub.add_parser("check")
    p.add_argument("--out",type=Path,required=True)
    p=sub.add_parser("fit")
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--config",type=Path,default=DEFAULT)
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--dense",action="store_true")
    mode=p.add_mutually_exclusive_group()
    mode.add_argument("--minimal",action="store_true")
    mode.add_argument("--compare-kernels",action="store_true",
                      help="Fit smoothstep and elastic coverage from a shared connected checkpoint.")
    mode.add_argument("--hybrid",action="store_true",
                      help="Fit and selectively harden an indoor mesh. Defaults to elastic coverage.")
    p.add_argument("--kernel",choices=("smoothstep","elastic"),
                   help="Coverage kernel for provisional triangles.")
    p.add_argument("--tracks",type=Path)
    p=sub.add_parser("evaluate")
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    reference=p.add_mutually_exclusive_group()
    reference.add_argument("--reference-mesh",type=Path)
    reference.add_argument("--rgb-only",action="store_true",
                           help="Score held-out RGB and render mesh diagnostics without reference geometry.")
    p.add_argument("--reference-masks",type=Path)
    p.add_argument("--first-hit",action="store_true",
                   help="Render all nontransmitting faces opaque for a common RGB comparison.")
    p=sub.add_parser("export")
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--threshold",type=float,default=.90)
    p=sub.add_parser("render")
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--views",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--config",type=Path,default=DEFAULT)
    p=sub.add_parser("score-external")
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--mesh",type=Path,required=True)
    p.add_argument("--reference-mesh",type=Path,required=True)
    p.add_argument("--reference-masks",type=Path)
    p.add_argument("--renders",type=Path)
    p.add_argument("--config",type=Path,default=DEFAULT)
    p.add_argument("--out",type=Path,required=True)
    p=sub.add_parser("fit-diagnostics")
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--fit",type=Path,required=True)
    p.add_argument("--config",type=Path,default=DEFAULT)
    p.add_argument("--out",type=Path,required=True)
    p=sub.add_parser("lpips")
    p.add_argument("--eval",type=Path,required=True)
    p=sub.add_parser("prepare-room")
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--images",type=Path,required=True)
    p.add_argument("--split",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--width",type=int,default=240)
    p.add_argument("--reference-view",required=True)
    p.add_argument("--pose-provenance",required=True)
    p.add_argument("--cell",type=int,default=18)
    p.add_argument("--gpu-features",action="store_true")
    from .orange4f import REFERENCE_VIEW, SOURCE
    p=sub.add_parser("orange4f",help="Fit and evaluate a local chart from the Orange4F COLMAP export.")
    p.add_argument("--source",type=Path,default=SOURCE)
    p.add_argument("--images",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--config",type=Path,default=DEFAULT.with_name("room_config.json"))
    p.add_argument("--reference-view",default=REFERENCE_VIEW)
    p.add_argument("--reference-mesh",type=Path)
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--fit-views",type=int,default=18)
    p.add_argument("--edit-views",type=int,default=3)
    p.add_argument("--final-views",type=int,default=4)
    p.add_argument("--window",type=int,default=160)
    p.add_argument("--radius",type=float,default=2.5)
    p.add_argument("--width",type=int,default=384)
    p.add_argument("--cell",type=int,default=20)
    p.add_argument("--gpu-features",action="store_true")
    p.add_argument("--kernel",choices=("smoothstep","elastic"),default="elastic")
    args=parser.parse_args()
    if args.action=="make-synthetic":
        manifest=generate(args.out)
        save_json(args.out/"split_manifest.json",manifest)
    elif args.action=="check":
        save_json(args.out,run_checks())
    elif args.action=="fit":
        if args.dense and args.tracks:
            parser.error("--dense and externally triangulated --tracks are mutually exclusive.")
        if args.compare_kernels and args.kernel:
            parser.error("--compare-kernels fits both kernels and does not take --kernel.")
        cfg = config(args.config)
        if args.kernel:
            cfg["kernel"] = args.kernel
        run(args.data,args.out,cfg,args.seed,args.dense,args.tracks,args.minimal,
            kernel_comparison=args.compare_kernels,hybrid=args.hybrid)
    elif args.action=="evaluate":
        evaluate(args.data,args.run,args.out,args.reference_mesh,args.reference_masks,args.first_hit,
                 rgb_only=args.rgb_only)
    elif args.action=="export":
        export(args.checkpoint,args.out,args.threshold)
    elif args.action=="render":
        from .evaluate import as_image
        _,cams=load_views(args.views)
        mesh=Mesh.read(args.checkpoint)
        args.out.mkdir(parents=True,exist_ok=True)
        for i,cam in enumerate(cams):
            r=render_view(mesh,cam,config(args.config)["tau_px"])
            as_image(r["rgb"].mean(2)).save(args.out/f"{i:03}.png")
            np.savez_compressed(args.out/f"{i:03}.npz",**r)
    elif args.action=="score-external":
        from .evaluate import score_external
        score_external(args.data,args.reference_mesh,args.mesh,config(args.config),args.out,
                       args.renders,args.reference_masks)
    elif args.action=="fit-diagnostics":
        from .evaluate import fitting_diagnostics
        fitting_diagnostics(args.checkpoint,args.fit,config(args.config),args.out)
    elif args.action=="orange4f":
        from .orange4f import prepare_orange4f, write_report
        cfg = config(args.config) | {"kernel": args.kernel}
        prepare_orange4f(args)
        data,out = args.out/"input",args.out/"run"
        run(data,out,cfg,args.seed,tracks_path=data/"fitting_tracks.npz",hybrid=True)
        evaluate(data,out,args.out/"eval",args.reference_mesh,rgb_only=args.reference_mesh is None,
                 checkpoint_name="hybrid.npz")
        write_report(args.out)
    elif args.action=="lpips":
        from .perceptual import score_folder
        score_folder(args.eval)
    else:
        from .room import prepare
        prepare(args)


if __name__=="__main__":
    main()
