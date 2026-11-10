#!/usr/bin/env python3
"""Check actual CUDA and xFormers execution before loading DA3 or scene data."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def require_cuda(torch, gpu):
    if not torch.cuda.is_available():
        raise RuntimeError(
            f'DA3 cannot initialize CUDA on requested GPU {gpu}. '
            f'Python={sys.executable}, torch={torch.__version__}, wheel CUDA={torch.version.cuda}. '
            'For the CUDA-12.4 driver host, activate da3-depth and install the pinned '
            'requirements-da3.txt, then run scripts/check_da3_environment.py --gpu '+str(gpu)+'. '
            'Training must wait until depth preparation succeeds.')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', default='3')
    p.add_argument('--imports-only', action='store_true',
                   help='Validate the actual DA3 API import chain without requiring a GPU')
    args = p.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    print(f'Python: {sys.executable}; torch: {torch.__version__}; wheel CUDA: {torch.version.cuda}', flush=True)
    if not args.imports_only:
        require_cuda(torch, args.gpu)
    import torchvision
    import xformers
    from xformers.ops import memory_efficient_attention
    from depth_anything_3.api import DepthAnything3
    if args.imports_only:
        print(f'DA3 API imports passed; torchvision={torchvision.__version__}; '
              f'xformers={xformers.__version__}', flush=True)
        return 0
    with torch.inference_mode():
        # Exercise CUDA kernels and the matching xFormers extension, not only imports.
        q = torch.randn(1, 64, 4, 64, dtype=torch.float16, device='cuda')
        output = memory_efficient_attention(q, q, q)
        if not torch.isfinite(output).all():
            raise RuntimeError('xFormers CUDA check produced nonfinite output')
        torch.cuda.synchronize()
    print(f'DA3 CUDA check passed on GPU {args.gpu}: {torch.cuda.get_device_name(0)}; '
          f'torchvision={torchvision.__version__}; xformers={xformers.__version__}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
