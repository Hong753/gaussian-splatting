"""Exact candidate-only operations for Proxy-GS's no-grad anchor growth."""


def selected_offset_positions(anchors, offsets, scaling, candidate_mask):
    import torch
    flat = candidate_mask.nonzero(as_tuple=True)[0]
    anchor_ids = torch.div(flat, offsets.shape[1], rounding_mode="floor")
    offset_ids = flat.remainder(offsets.shape[1])
    return anchors[anchor_ids] + offsets[anchor_ids, offset_ids] * scaling[anchor_ids, :3]


def pooled_candidate_features(anchor_features, n_offsets, candidate_mask,
                              inverse_indices, keep_groups, chunk_size=65536):
    """Match repeat/mask/scatter-max without allocating N*offsets*features.

    Each candidate belongs to a unique voxel group in inverse_indices. Only
    retained groups need output features; candidate features are gathered in
    chunks. Called under the trainer's torch.no_grad() densification block.
    """
    import torch
    if chunk_size < 1:
        raise ValueError("Feature pooling chunk size must be positive")
    count = int(keep_groups.sum().item())
    result = anchor_features.new_full((count, anchor_features.shape[1]), -torch.inf)
    if count == 0:
        return result
    group_ids = keep_groups.long().cumsum(0) - 1
    group_ids[~keep_groups] = -1
    selected = candidate_mask.nonzero(as_tuple=True)[0]
    if selected.numel() != inverse_indices.numel():
        raise ValueError("Candidate offsets and voxel assignments differ in length")
    for start in range(0, selected.numel(), chunk_size):
        end = min(start + chunk_size, selected.numel())
        groups = group_ids[inverse_indices[start:end]]
        valid = groups >= 0
        anchor_ids = torch.div(selected[start:end][valid], n_offsets, rounding_mode="floor")
        features = anchor_features.index_select(0, anchor_ids)
        result.scatter_reduce_(0, groups[valid, None].expand(-1, features.shape[1]),
                               features, reduce="amax", include_self=True)
    return result


def occupied_voxel_mask(existing, candidates):
    """Exact integer-row membership with O(N+M), not O(N*M), storage."""
    import torch
    if existing.shape[0] == 0 or candidates.shape[0] == 0:
        return torch.zeros(candidates.shape[0], dtype=torch.bool, device=candidates.device)
    unique, inverse = torch.unique(torch.cat((existing, candidates), dim=0), dim=0,
                                   return_inverse=True)
    occupied = torch.zeros(unique.shape[0], dtype=torch.bool, device=existing.device)
    occupied[inverse[:existing.shape[0]]] = True
    return occupied[inverse[existing.shape[0]:]]
