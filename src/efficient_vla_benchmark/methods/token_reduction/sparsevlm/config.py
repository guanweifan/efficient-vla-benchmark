"""Original SparseVLM rank adaptation and recycling, not SparseVLM+."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class SparseVLMConfig:
    enabled: bool = True
    pruning_layers: tuple[int, ...] = (2, 6, 15)
    rank_scale: float = 0.25
    recycle_ratio: float = 0.3
    cluster_ratio: float = 0.1
    neighbors: int = 3
    rank_rtol: float = 1e-5

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {field.name for field in fields(cls)}:
            raise ValueError("Unknown SparseVLM parameter")
        parameters = dict(parameters)
        if "pruning_layers" in parameters:
            value = parameters["pruning_layers"]
            if not isinstance(value, (list, tuple)):
                raise ValueError("pruning_layers must be a sequence")
            parameters["pruning_layers"] = tuple(value)
        config = cls(**parameters)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if (
            not config.pruning_layers
            or any(type(x) is not int or not 0 <= x < 31 for x in config.pruning_layers)
            or tuple(sorted(set(config.pruning_layers))) != config.pruning_layers
        ):
            raise ValueError("pruning_layers must be unique increasing indices in [0, 30]")
        for name in ("rank_scale", "recycle_ratio", "cluster_ratio", "rank_rtol"):
            value = getattr(config, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if type(config.neighbors) is not int or config.neighbors < 1:
            raise ValueError("neighbors must be positive")
        return config
