#!/usr/bin/env python3
"""Train native full-scene Orange4F 3DGS with inverse-square ray gradient weights."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_lidar as native
from scripts import run_soft_surface_long as periodic
from scripts.orange4f_detail import SUITE, experiment_lock, select_gpu
from scripts.prepare_orange4f import native_run_arguments
from scripts.refresh_orange4f_dashboard import refresh

RUN = "3dgs_ray_depth_gradients_v1"


def prepare(suite, gpu=None):
    inputs = json.loads((suite / "suite.json").read_text())
    model = suite / RUN
    with experiment_lock(model):
        if (periodic.directory(model) / "plan.json").is_file():
            plan = periodic.load_plan(model)
            if (
                not plan["plan"].get("ray_depth_gradients")
                or Path(plan["source"]) != Path(inputs["source"]).resolve()
            ):
                raise ValueError(f"Existing plan does not match the Orange4F ray-depth run at {model}.")
            if gpu is not None and gpu != plan["gpu"]:
                raise ValueError(f"This plan uses GPU {plan['gpu']}.")
        else:
            gpu = select_gpu(gpu)
            command = native_run_arguments(
                inputs["source"], model, inputs["selection"], inputs["cloud"]["cached"], gpu, inputs["frames"]
            )
            native.prepare(command + ["--ray-depth-gradients"])
        refresh(suite)
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=SUITE)
    parser.add_argument("--gpu", help="Physical CUDA device. Defaults to an idle GPU.")
    parser.add_argument(
        "--prepare-only", action="store_true", help="Save the native plan and publish dashboard links"
    )
    args = parser.parse_args(argv)
    model = prepare(args.suite.expanduser().resolve(), args.gpu)
    if not args.prepare_only:
        launcher = str(ROOT / "scripts/run_prepared_reconstruction.sh")
        os.execvp("bash", ["bash", launcher, str(model)])


if __name__ == "__main__":
    main()
