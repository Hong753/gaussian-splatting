#!/usr/bin/env python3
"""Native 3DGS+AA with soft surface regularization and mesh-free inference."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_occlusion


def parser():
    p = run_occlusion.parser()
    p.description = __doc__
    p.set_defaults(soft_surface=True, inference_occlusion='off', gpu='0',
        position_warmup_steps=7000, position_ramp_steps=2000, position_lr_scale=.02)
    return p


def main(argv=None):
    return run_occlusion.main(argv, argument_parser=parser())


if __name__ == '__main__':
    raise SystemExit(main())
