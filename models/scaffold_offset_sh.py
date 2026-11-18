"""Independent SH color tables for the native and pinned upstream Scaffold trainers."""
from functools import partial
from pathlib import Path

import torch
from torch import nn


class OffsetSHColors(nn.Module):
    __constants__ = ["degree"]

    def __init__(self, anchors, offsets, degree, reference):
        super().__init__()
        self.degree = degree
        self.coefficients = nn.Parameter(reference.new_zeros((anchors, offsets, 3, (degree + 1) ** 2)))

    def forward(self, parents: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.coefficients[parents, offsets]


class OffsetSHModelMixin:
    """Use the trainer's anchor optimizer and checkpoint layout for the color table."""

    def __init__(self, *args, offset_sh_degree, **kwargs):
        super().__init__(*args, **kwargs)
        self.offset_sh_degree = offset_sh_degree
        self._init_offset_colors(0)

    def _init_offset_colors(self, count):
        reference = next(self.mlp_cov.parameters())
        self.mlp_color = OffsetSHColors(count, self.n_offsets, self.offset_sh_degree, reference)

    def create_from_pcd(self, *args, **kwargs):
        super().create_from_pcd(*args, **kwargs)
        self._init_offset_colors(len(self.get_anchor))

    def training_setup(self, training_args):
        super().training_setup(training_args)
        color_group = next(g for g in self.optimizer.param_groups if g["name"] == "mlp_color")
        color_group.update(name="offset_sh", lr=0.0025)

    def cat_tensors_to_optimizer(self, tensors_dict):
        # New voxels start at neutral gray. Existing rows and Adam moments retain their order.
        table = self.mlp_color.coefficients
        extension = table.new_zeros((len(tensors_dict["anchor"]), *table.shape[1:]))
        params = super().cat_tensors_to_optimizer(dict(tensors_dict, offset_sh=extension))
        self.mlp_color.coefficients = params["offset_sh"]
        return params

    def _prune_anchor_optimizer(self, mask):
        params = super()._prune_anchor_optimizer(mask)
        self.mlp_color.coefficients = params["offset_sh"]
        return params

    def restore(self, state, training_args):
        # Native optimizer checkpoint, possibly from a different anchor population.
        if not isinstance(state, dict):
            return super().restore(state, training_args)
        table = state["modules"]["mlp_color"].get("coefficients")
        expected = (len(state["tensors"]["_anchor"]), self.n_offsets, 3, (self.offset_sh_degree + 1) ** 2)
        if table is None or tuple(table.shape) != expected:
            raise ValueError("SH checkpoint color table does not match the anchor count, offsets or degree")
        self._init_offset_colors(expected[0])
        super().restore(state, training_args)

    def save_mlp_checkpoints(self, path, mode="split"):
        if mode != "split":
            return super().save_mlp_checkpoints(path, mode)
        # Preserve the public checkpoint filenames. The color module contains the
        # lookup table itself, so all three inference files remain independently loadable.
        modules = {"opacity_mlp": self.mlp_opacity, "cov_mlp": self.mlp_cov, "color_mlp": self.mlp_color}
        if self.use_feat_bank:
            modules["feature_bank_mlp"] = self.mlp_feature_bank
        if self.appearance_dim:
            modules["embedding_appearance"] = self.embedding_appearance
        for name, module in modules.items():
            torch.jit.script(module).save(str(Path(path) / f"{name}.pt"))

    def load_mlp_checkpoints(self, path, mode="split"):
        if mode != "split":
            self._init_offset_colors(len(self.get_anchor))
        super().load_mlp_checkpoints(path, mode)
        table = getattr(self.mlp_color, "coefficients", None)
        if table is None or tuple(table.shape) != (
            len(self.get_anchor), self.n_offsets, 3, (self.offset_sh_degree + 1) ** 2
        ):
            raise ValueError("Saved SH color table does not match its PLY or configured degree")


def scaffold_model_class(base, degree):
    if degree == -1:
        return base
    if degree not in (0, 1, 2, 3):
        raise ValueError("Scaffold offset SH degree must be -1 (MLP) or 0 through 3")

    class OffsetSHModel(OffsetSHModelMixin, base):
        pass

    return partial(OffsetSHModel, offset_sh_degree=degree)
