"""Explicit primary-view DepthCache scope and untuned threshold defaults."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class DepthCacheConfig:
    enabled: bool = True
    warmup_queries: int = 5
    depth_clusters: int = 3
    merge_window: int = 5
    min_merge_ratio: float = 0.0
    max_merge_ratio: float = 0.7
    edge_threshold_m: float = 0.03
    static_threshold_m: float = 0.01
    dynamic_fraction: float = 0.2
    attention_layer: int = 31

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {field.name for field in fields(cls)}:
            raise ValueError("Unknown DepthCache primary-core parameter")
        config = cls(**parameters)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        for name in ("warmup_queries", "depth_clusters", "merge_window"):
            value = getattr(config, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be positive")
        if type(config.attention_layer) is not int or not 0 <= config.attention_layer < 32:
            raise ValueError("attention_layer must be in [0, 31]")
        for name in ("min_merge_ratio", "max_merge_ratio", "dynamic_fraction"):
            value = getattr(config, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if config.min_merge_ratio > config.max_merge_ratio:
            raise ValueError("Merge-ratio range is reversed")
        for name in ("edge_threshold_m", "static_threshold_m"):
            value = getattr(config, name)
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        return config
