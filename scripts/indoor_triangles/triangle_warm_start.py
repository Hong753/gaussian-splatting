"""Initialize a trainable TriangleModel from an upstream render snapshot."""

from pathlib import Path

import torch
from torch import nn


def load_training_snapshot(model, snapshot, camera_extent, device="cuda"):
    snapshot = Path(snapshot)
    state = torch.load(snapshot / "point_cloud_state_dict.pt", map_location="cpu", weights_only=True)
    vertices = state["triangles_points"]
    if vertices.ndim != 3 or vertices.shape[1:] != (3, 3) or len(vertices) == 0:
        raise ValueError(f"Expected a nonempty triangle model at {snapshot}")
    count = len(vertices)
    fields = {"triangles_points": (count, 3, 3), "opacity": (count, 1), "sigma": (count, 1),
              "features_dc": (count, 1, 3), "features_rest": (count, (model.max_sh_degree + 1)**2 - 1, 3)}
    for key, shape in fields.items():
        tensor = state[key]
        if tensor.shape != shape:
            raise ValueError(f"Unexpected {key} shape {tuple(tensor.shape)} at {snapshot}, expected {shape}")
        setattr(model, "_" + key, nn.Parameter(tensor.detach().to(device=device, dtype=torch.float32)))
    model.active_sh_degree = int(state["active_sh_degree"])
    if not 0 <= model.active_sh_degree <= model.max_sh_degree:
        raise ValueError(f"Invalid active SH degree at {snapshot}")
    model.nb_points = 3
    model._number_of_points = count
    model._num_points_per_triangle = torch.full((count,), 3, dtype=torch.int32, device=device)
    model._cumsum_of_points_per_triangle = torch.arange(count, dtype=torch.int32, device=device) * 3
    model._mask = nn.Parameter(model._opacity.new_ones((count, 1)))
    model.spatial_lr_scale = camera_extent
    model.large = bool((vertices.amax(dim=(0, 1)) - vertices.amin(dim=(0, 1))).max() > 300)
    for key in ("max_scaling", "max_radii2D", "max_density_factor", "triangle_area", "image_size", "importance_score"):
        setattr(model, key, model._opacity.new_zeros((count,)))
    print(f"Warm started {count:,} triangles from {snapshot}. Initializing fresh optimizer and densification statistics.",
          flush=True)
