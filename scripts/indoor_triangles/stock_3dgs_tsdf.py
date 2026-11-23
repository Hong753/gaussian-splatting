"""Extract a mesh from the pinned stock 3DGS renderer using fitting RGB cameras."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--repo",type=Path,required=True)
    cli.add_argument("--model",type=Path,required=True)
    cli.add_argument("--source",type=Path,required=True)
    cli.add_argument("--out",type=Path,required=True)
    cli.add_argument("--iteration",type=int,default=30000)
    cli.add_argument("--voxel-m",type=float,default=.025)
    cli.add_argument("--trunc-m",type=float,default=.075)
    cli.add_argument("--max-z-m",type=float,default=12.)
    cli.add_argument("--minimum-alpha",type=float,default=.55)
    cli.add_argument("--antialiasing",action="store_true")
    args=cli.parse_args()
    sys.path.insert(0,str(args.repo.resolve()))

    import numpy as np
    import open3d as o3d
    import torch
    from arguments import ModelParams, PipelineParams
    from gaussian_renderer import render
    from scene import GaussianModel, Scene

    start=perf_counter()
    parser=argparse.ArgumentParser()
    source=ModelParams(parser)
    renderer=PipelineParams(parser)
    flags=["-s",str(args.source.resolve()),"-m",str(args.model.resolve()),"-r","1"]
    if args.antialiasing:
        flags.append("--antialiasing")
    parsed=parser.parse_args(flags)
    dataset=source.extract(parsed)
    pipe=renderer.extract(parsed)
    gaussians=GaussianModel(dataset.sh_degree)
    scene=Scene(dataset,gaussians,load_iteration=args.iteration,shuffle=False)
    volume=o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_m,sdf_trunc=args.trunc_m,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    trace=[]
    with torch.no_grad():
        white=torch.ones_like(gaussians.get_xyz)
        bg=torch.zeros(3,device=white.device,dtype=white.dtype)
        for view in scene.getTrainCameras():
            pred=render(view,gaussians,pipe,bg)
            color=(pred["render"].clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
            alpha=render(view,gaussians,pipe,bg,override_color=white)["render"][0].cpu().numpy()
            inv=pred["depth"].squeeze().cpu().numpy()
            # The rasterizer's inverse depth is sum(T_i * alpha_i / z_i).
            valid=(alpha>=args.minimum_alpha)&(inv>1e-7)&np.isfinite(inv)
            z=np.divide(alpha,inv,out=np.zeros_like(inv),where=valid)
            z[z>args.max_z_m]=0
            h,w=z.shape
            fx=w/(2*np.tan(view.FoVx/2))
            fy=h/(2*np.tan(view.FoVy/2))
            k=o3d.camera.PinholeCameraIntrinsic(w,h,fx,fy,(w-1)/2,(h-1)/2)
            rt=view.world_view_transform.T.cpu().double().numpy()
            data=o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(color)),o3d.geometry.Image(z.astype(np.float32)),
                depth_scale=1.,depth_trunc=args.max_z_m,convert_rgb_to_intensity=False)
            volume.integrate(data,k,rt)
            trace.append({"image":view.image_name,"valid_depth_pixels":int(np.count_nonzero(z))})
    model=volume.extract_triangle_mesh()
    if not len(model.triangles):
        raise ValueError("The fitted radiance field produced an empty TSDF mesh.")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(args.out),model):
        raise OSError(f"Mesh write failed: {args.out}")
    output={"parameters":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            "wall_s":perf_counter()-start,"vertices":len(model.vertices),"faces":len(model.triangles),
            "peak_vram_bytes":torch.cuda.max_memory_allocated(),"views":trace}
    args.out.with_suffix(".extract.json").write_text(json.dumps(output,indent=2)+"\n")


if __name__=="__main__":
    main()
