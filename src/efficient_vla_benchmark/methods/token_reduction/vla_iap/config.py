"""IAP v1 equations and explicit, untuned values for unspecified paper parameters."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class VLAIAPConfig:
    enabled: bool = True
    temperature: float = 0.01
    history_decay: float = 0.7
    mask_std_scale: float = 0.5
    background_std_scale: float = -0.5
    iou_threshold: float = 0.05
    semantic_radius: float = 3.0
    edge_weight: float = 1.0
    geometry_threshold: float = 1.0
    semantic_pool_kernel: int = 3
    closing_kernel: int = 3
    gaussian_kernel: int = 3
    gaussian_sigma: float = 1.0
    epsilon: float = 1e-6

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown VLA-IAP parameters")
        cfg = cls(**parameters)
        if type(cfg.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        kernels = ("semantic_pool_kernel", "closing_kernel", "gaussian_kernel")
        for key in kernels:
            value = getattr(cfg, key)
            if type(value) is not int or value not in (1, 3, 5, 7):
                raise ValueError(f"{key} must be an odd kernel in [1, 7]")
        for f in fields(cls):
            if f.name in ("enabled", *kernels):
                continue
            value = getattr(cfg, f.name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{f.name} must be finite")
        if cfg.temperature <= 0 or cfg.gaussian_sigma <= 0 or cfg.epsilon <= 0:
            raise ValueError("Temperature, sigma and epsilon must be positive")
        if not 0 <= cfg.history_decay < 1 or not 0 <= cfg.iou_threshold <= 1:
            raise ValueError("Invalid temporal decay or IoU threshold")
        if (
            cfg.background_std_scale >= 0
            or min(cfg.mask_std_scale, cfg.semantic_radius, cfg.edge_weight, cfg.geometry_threshold)
            < 0
        ):
            raise ValueError("Invalid mask, background, radius or geometry threshold")
        return cfg
