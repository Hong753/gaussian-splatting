#!/usr/bin/env python3
"""Fresh regional sharp-stride3 run: full RGB supervision, frozen LiDAR XYZ."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.keyframes import registered_names,select_keyframes
from scripts.keyframe_pipeline.common import read_model
from scripts import run_native_lidar as native,run_soft_surface_long as periodic
from utils.checkpoint_snapshots import atomic_json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--region',type=Path,required=True)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--iterations',type=int,default=30000)
    args=parser.parse_args(argv)
    region=args.region.expanduser().resolve()
    meta=json.loads((region/'selection.json').read_text())
    if meta.get('schema')!=1 or Path(meta['region'])!=region:
        raise ValueError('Use the existing original regional export at its recorded path')
    if (region/'colmap_rerun.json').exists():
        raise ValueError('This restart uses original poses, not the rerun COLMAP export')
    names=registered_names(region)
    selected_region=[row['name'] for row in meta['selection']]
    if len(selected_region)!=len(set(selected_region)) or set(names)!=set(selected_region):
        raise ValueError('Regional registered cameras no longer match the selected image subset')
    if len(names)<4 or args.iterations<15000:
        raise ValueError('Need at least four regional images and the full 15000-step native pruning/reset phase')
    model=(args.model or region/'runs/stride3_fullimage_fixed_lidar_native_lr_resets_v1').expanduser().resolve()
    if model.exists():
        raise FileExistsError(f'Use a fresh model directory; retained existing output: {model}')
    # Same scorer as the original full-scene stride3 experiment. Group ALL
    # region-selected frames before selecting training winners; do not stride
    # again over the old global stride3 winners, and do not read ROI masks.
    manifest=select_keyframes(region,'images',model,3,False)
    chosen=json.loads(manifest.read_text())['selected_images']
    command=['--source',str(region),'--model',str(model),'--keyframes',str(manifest),
        '--point-cloud',str(region/'input.ply'),'--gpu',args.gpu,
        '--iterations',str(args.iterations),'--checkpoint-every','10000',
        '--expected-frames',str(len(names)),'--resolution','1',
        '--freeze-lidar-positions','--pruning','--opacity-resets',
        '--feature-lr','0.0025','--opacity-lr','0.05','--scaling-lr','0.005','--rotation-lr','0.001',
        '--densify-from-iter','500','--densify-until-iter','15000','--densification-interval','100',
        '--percent-dense','0.01','--densify-grad-threshold','0.0002',
        '--prune-world-scale-ratio','0.1','--prune-screen-size','20','--prune-size-from-iter','3000']
    training=native.parser().parse_args(command)
    plan=native.build_plan(training)
    _,_,_,pose_files=read_model(region)
    inputs=[manifest,region/'selection.json',region/'input.ply',*[Path(stamp[0]) for stamp in pose_files]]
    result=periodic.prepare_experiment(training,plan,'native_lidar',inputs)
    protocol=dict(region=str(region),model=str(model),gpu=args.gpu,iterations=args.iterations,
        source_images=len(names),train_images=len(chosen),heldout_images=len(names)-len(chosen),
        selection='Sharpest full-resolution grayscale Laplacian variance per consecutive group of 3 region-selected images in natural filename order; partial final group included',
        split='New regional stride3 winners train; all other region-selected images held out. Old global membership is not reused.',
        camera_poses='Original regional poses, fixed',
        lidar='Same regional crop+halo input.ply; surviving centers remain exact initialization coordinates',
        geometry='No XYZ optimization or cloning/splitting; native opacity/size pruning enabled',
        learning_rates=dict(xyz=0.,sh_dc=.0025,sh_rest=.000125,opacity=.05,scale=.005,rotation=.001),
        opacity_resets=plan['opacity_reset_schedule'],
        loss='Full-image native L1 + SSIM, lambda_dssim=0.2; no ROI mask',
        evaluation='Full-image PSNR on every regional image; train/heldout split from the new manifest; no dimming',
        antialiasing=True,culling=False,scale_cap=False,
        previews='GT/render/error at step 1, every 500 steps and final; parallel full-image evaluation every 10000 steps',
        caveat='The LiDAR cloud is still cropped to the selected region plus halo. Pixels outside that geometry have no guaranteed coverage. Full-image PSNR is not directly comparable with prior ROI PSNR.',
        lr_reference='https://github.com/graphdeco-inria/gaussian-splatting#running',
        unchanged_inputs='Original RGB files, original poses, regional LiDAR, old masks and old keyframe manifest')
    atomic_json(model/'region_full_image.json',protocol)
    print(json.dumps(protocol,indent=2),flush=True)
    return result


if __name__=='__main__': raise SystemExit(main())
