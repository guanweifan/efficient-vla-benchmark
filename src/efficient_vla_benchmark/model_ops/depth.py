"""Metric-depth geometry aligned with the selected policy's image coordinates."""

import math

import torch
from torch.nn import functional as F


def depth_patch_signals(depth, device, crop_area=1.0, edge_threshold=0.03):
    values = torch.as_tensor(depth, dtype=torch.float32, device=device)[None, None]
    if values.ndim != 4 or not 0 < crop_area <= 1:
        raise ValueError("Expected metric HxW depth and a valid center crop")
    # Nearest resampling preserves measured surfaces instead of blending their depths.
    values = F.interpolate(values, size=(224, 224), mode="nearest-exact")
    if crop_area < 1:
        width = math.sqrt(crop_area)
        coordinates = torch.linspace(-width, width, 224, device=device)
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        grid = torch.stack((xx, yy), -1)[None]
        values = F.grid_sample(values, grid, mode="nearest", align_corners=True)
    gradient = torch.zeros_like(values)
    vertical = (values[..., 1:, :] - values[..., :-1, :]).abs()
    horizontal = (values[..., :, 1:] - values[..., :, :-1]).abs()
    gradient[..., 1:, :] = torch.maximum(gradient[..., 1:, :], vertical)
    gradient[..., :-1, :] = torch.maximum(gradient[..., :-1, :], vertical)
    gradient[..., :, 1:] = torch.maximum(gradient[..., :, 1:], horizontal)
    gradient[..., :, :-1] = torch.maximum(gradient[..., :, :-1], horizontal)
    return (
        F.avg_pool2d(values, 14).flatten(),
        F.max_pool2d(gradient, 14).flatten() > edge_threshold,
    )
