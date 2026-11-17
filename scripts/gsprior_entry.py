#!/usr/bin/env python3
"""Train/evaluate the isolated upstream GSPrior, never import the native renderer."""
import argparse
import gc
import json
import os
from pathlib import Path
import runpy
import sys
import time


def evaluate_checkpoint(config, iteration):
    import torch
    from arguments import ModelParams, PipelineParams
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    import experiment_adapter as adapter
    from types import SimpleNamespace
    p=argparse.ArgumentParser();mp=ModelParams(p);pp=PipelineParams(p)
    args=p.parse_args(['-s',config['source'],'-m',config['model'],'-r','1','--data_device','cpu'])
    dataset=mp.extract(args);pipe=pp.extract(args);pipe.antialiasing=False
    model=GaussianModel(dataset.sh_degree)
    scene=Scene(dataset,model,load_iteration=iteration,shuffle=False)
    scene._previewer=SimpleNamespace(frames=[])
    scene._preview_render=lambda c,m,p,b,**kw:render(c,m,p,b,return_plane=False,return_depth_normal=False)
    result=adapter.evaluate_all(scene,pipe,render,torch.zeros(3,device='cuda'),iteration)
    del model,scene
    gc.collect();torch.cuda.empty_cache()
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('train','watch','eval'))
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--iteration',type=int,default=30000)
    p.add_argument('--parent-pid',type=int)
    args=p.parse_args(argv)
    model=args.model.resolve();repo=args.repo.resolve()
    config=json.loads((model/'gsprior_run.json').read_text())
    os.environ['CUDA_VISIBLE_DEVICES']=config['gpu']
    os.environ['GSPRIOR_RUN_CONFIG']=str(model/'gsprior_run.json')
    # GSPrior and native 3DGS have identically named modules and incompatible
    # rasterizer signatures. Only the dedicated checkout is on the import path.
    project=Path(__file__).resolve().parents[1]
    sys.path[:]=[str(repo)]+[s for s in sys.path if s and Path(s).resolve() not in (project,project/'scripts')]
    os.chdir(repo)
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('GSPrior requires a working CUDA PyTorch environment')
    from utils.checkpoint_snapshots import atomic_json, digest
    import experiment_adapter as adapter
    state=model/'training_state.json'
    if args.mode=='train':
        if (model/'cfg_args').exists(): raise FileExistsError('Model already started; use a fresh suite version')
        atomic_json(state,dict(state='running',pid=os.getpid()))
        try:
            for name, expected in config['input_fingerprints'].items():
                if digest(Path(name)) != expected: raise ValueError(f'Prepared input changed: {name}')
            adapter.smoke()
            sys.argv=['train.py','-s',config['source'],'-m',str(model),'-r','1','--data_device','cpu',
                '--iterations',str(config['iterations']),'--tsdf_iter',*map(str,config['tsdf_iterations']),
                '--tsdf_trunc',*map(str,config['grid']['truncation']),
                '--test_iterations','-1','--save_iterations',*map(str,config['checkpoints']),
                '--checkpoint_iterations',*map(str,config['checkpoints'])]
            runpy.run_path(str(repo/'train.py'),run_name='__main__')
        except BaseException as exc:
            atomic_json(state,dict(state='failed',error=str(exc),pid=os.getpid()));raise
        atomic_json(state,dict(state='trained',pid=os.getpid()))
    elif args.mode=='eval':
        evaluate_checkpoint(config,args.iteration)
    else:
        torch.cuda.set_per_process_memory_fraction(.20)
        for iteration in config['checkpoints']:
            ready=model/'point_cloud'/f'iteration_{iteration}'/'ready.json'
            result=model/'psnr_all_frames'/f'iteration_{iteration}'/'complete.json'
            if result.exists(): continue
            while not ready.exists():
                status=json.loads(state.read_text()) if state.exists() else {}
                if status.get('state')=='failed': raise RuntimeError(status['error'])
                if status.get('state')=='trained': raise RuntimeError(f'Missing completed checkpoint {iteration}')
                if args.parent_pid:
                    try: os.kill(args.parent_pid,0)
                    except ProcessLookupError: return 0
                time.sleep(10)
            while True:
                try:
                    evaluate_checkpoint(config,iteration)
                    break
                except torch.cuda.OutOfMemoryError:
                    gc.collect();torch.cuda.empty_cache()
                    status=json.loads(state.read_text()) if state.exists() else {}
                    if status.get('state') in ('trained','failed'): raise
                    print(f'Evaluation {iteration}: waiting for VRAM; retrying in 30s',flush=True)
                    time.sleep(30)
    return 0


if __name__=='__main__': raise SystemExit(main())
