#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

here = os.path.dirname(os.path.abspath(__file__))
glm = os.environ.get("ABSGS_GLM_PATH", os.path.join(here, "..", "diff-gaussian-rasterization", "third_party", "glm"))
if not os.path.isfile(os.path.join(glm, "glm", "glm.hpp")):
    raise RuntimeError(f"GLM headers not found under {glm}; set ABSGS_GLM_PATH to a folder containing glm/glm.hpp")

setup(
    name="diff_gaussian_rasterization_abs",
    packages=['diff_gaussian_rasterization_abs'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization_abs._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.abspath(glm)]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

