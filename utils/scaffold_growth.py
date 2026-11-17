"""Linear-storage anchor candidate selection (no candidate-by-scene matrix)."""
import torch


def new_voxel_mask(existing, candidates):
    if candidates.shape[0] == 0:
        return torch.zeros(0, dtype=torch.bool, device=candidates.device)
    _, inverse = torch.unique(torch.cat((existing, candidates)), dim=0, return_inverse=True)
    occupied = torch.zeros(len(existing) + len(candidates), dtype=torch.bool, device=existing.device)
    occupied[inverse[:len(existing)]] = True
    return ~occupied[inverse[len(existing):]]


def candidate_offsets(anchors, offsets, scales, candidate_mask):
    ids = candidate_mask.nonzero(as_tuple=True)[0]
    parents = ids // offsets.shape[1]
    xyz = anchors[parents] + offsets.reshape(-1, 3)[ids] * scales[parents, :3]
    return xyz, parents
