"""Released dual-score pruning, with explicit retention rather than drop ratios."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class VLAPrunerConfig:
    enabled: bool = True
    keep_ratio: float = 0.25
    pruning_layer: int = 3
    history_layer: int = 15
    temporal: bool = True
    history_size: int = 3
    decay: float = 0.8

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown VLA-Pruner parameters")
        cfg = cls(**parameters)
        for name in ("enabled", "temporal"):
            if type(getattr(cfg, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("keep_ratio", "decay"):
            v = getattr(cfg, name)
            if type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 1:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        for name in ("pruning_layer", "history_layer"):
            if type(getattr(cfg, name)) is not int or getattr(cfg, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(cfg.history_size) is not int or cfg.history_size < 1:
            raise ValueError("history_size must be a positive integer")
        return cfg
