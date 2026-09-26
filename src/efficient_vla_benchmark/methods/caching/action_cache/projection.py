"""Data-independent, exactly balanced sparse ternary projection."""

import torch
import torch.nn.functional as F


def projection(width, dimension, density, seed, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    half = max(1, int(width * density / 2))
    if 2 * half > width:
        raise ValueError("Projection density exceeds source width")
    matrix = torch.zeros(dimension, width, dtype=torch.float32)
    for row in matrix:
        ids = torch.randperm(width, generator=generator)[: 2 * half]
        row[ids[:half]] = 1
        row[ids[half:]] = -1
    return matrix.to(device)


def key_for(condition, matrix):
    value = condition[0].reshape(-1).float()
    if value.numel() != matrix.shape[1]:
        raise ValueError("Unexpected cognition key geometry")
    return F.normalize(matrix @ value, dim=0)
