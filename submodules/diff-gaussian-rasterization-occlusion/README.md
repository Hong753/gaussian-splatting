# Native AA rasterizer with fixed per-pixel depth clipping

Derived from the repository's vendored Graphdeco AA rasterizer, with the AbsGS
gradient accumulation removed. Original license and copyright are retained.
This package is separate from the baseline and AbsGS installed extensions.

Each splat fragment uses the native Gaussian center camera-z depth. A fragment
with alpha >= 1/255 is skipped when its depth exceeds the supplied finite pixel
cutoff. The cutoff contains geometric calibration margins and is fixed during
backpropagation. This is the native 3DGS depth approximation, not a volumetric
Gaussian/ray intersection or a learned opacity threshold.

Forward and backward apply the identical hard test, before transmittance updates.
Unknown pixels (infinity) only affect their own rays. AA covariance filtering and
opacity compensation are preserved. Output integer flags per Gaussian are:

- bit 1: at least one fragment actually contributed;
- bit 2: at least one encountered fragment was blocked by the fixed surface.

Native opacity early termination still applies: flags describe visited work,
not every theoretical fragment. `flags == 2` is fully blocked encountered work;
`flags == 3` is partially blocked. `flags == 0` does not prove geometric occlusion.

Use `scripts/smoke_ray_occlusion.py` on the GPU host. It checks AA-on/off parity
against the installed native renderer with infinite depth, forward/backward
wall behavior, a one-pixel opening, and masked Adam state. The viewer must not
be treated as validated until these real CUDA checks pass.
