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


def evaluate(data,run_folder,output,reference_mesh=None,reference_masks=None):
    data,run_folder,output=Path(data),Path(run_folder),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    cfg=config(run_folder/"config.json")
    images,cameras=load_views(data/"final.npz")
    if reference_mesh:
        from .room import reference_views
        vertices,faces,refs=reference_views(reference_mesh,cameras,reference_masks)
    else:
        with np.load(data/"reference.npz") as source:
            reference=tuple(source[k] for k in ("vertices","faces","kinds"))
        vertices,faces,kinds=reference
        refs=[ref_view(reference,cam) for cam in cameras]
        vertices,faces=vertices,faces[kinds!=4]
    target=surface_context(vertices,faces,cameras,[r[1] for r in refs],cfg)
    table=[]
    for ckpt in sorted(run_folder.glob("*.npz")):
        if ckpt.name=="fitting_tracks.npz":
            continue
        result=compare_arm(ckpt,images,cameras,refs,output,cfg,target)
        save_json(output/f"{ckpt.stem}.json",result)
        print(ckpt.stem,flush=True)
        views=result["views"]
        doorway=sum(v["doorway"]["reference_pixels"] for v in views)
        curve=config(run_folder/f"{ckpt.stem}.json")
        extra=curve.get("edit_wall_s",0)
        for mode,s in result["surface"].items():
            table.append(dict(arm=ckpt.stem,export=mode,faces=s["exported_faces"],
                              accuracy_m=s["accuracy_m"],completeness_m=s["completeness_m"],
                              chamfer_l1_m=s["chamfer_l1_m"],fscore_5cm=s["fscore"],
                              normal_consistency=s["normal_consistency"],
                              psnr_db=float(np.mean([v["psnr_db"] for v in views])),
                              ssim=float(np.mean([v["ssim"] for v in views])),
                              opaque_fraction=float(np.mean([v["opaque_subsample_fraction"] for v in views])),
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
    p.add_argument("--minimal",action="store_true")
    p.add_argument("--tracks",type=Path)
    p=sub.add_parser("evaluate")
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--reference-mesh",type=Path)
    p.add_argument("--reference-masks",type=Path)
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
    args=parser.parse_args()
    if args.action=="make-synthetic":
        manifest=generate(args.out)
        save_json(args.out/"split_manifest.json",manifest)
    elif args.action=="check":
        save_json(args.out,run_checks())
    elif args.action=="fit":
        if args.dense and args.tracks:
            parser.error("--dense and externally triangulated --tracks are mutually exclusive.")
        run(args.data,args.out,config(args.config),args.seed,args.dense,args.tracks,args.minimal)
    elif args.action=="evaluate":
        evaluate(args.data,args.run,args.out,args.reference_mesh,args.reference_masks)
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
    elif args.action=="lpips":
        from .perceptual import score_folder
        score_folder(args.eval)
    else:
        from .room import prepare
        prepare(args)


if __name__=="__main__":
    main()
