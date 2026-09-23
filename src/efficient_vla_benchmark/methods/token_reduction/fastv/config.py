"""FastV HF-code variant: last-query attention at an early decoder layer."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class FastVConfig:
    enabled: bool = True
    pruning_layer: int = 2
    keep_ratio: float = 0.5

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {field.name for field in fields(cls)}:
            raise ValueError("Unknown FastV parameter")
        config = cls(**parameters)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if type(config.pruning_layer) is not int or not 1 <= config.pruning_layer <= 30:
            raise ValueError("pruning_layer must be in [1, 30]")
        if (
            type(config.keep_ratio) not in (float, int)
            or not math.isfinite(config.keep_ratio)
            or not 0 < config.keep_ratio <= 1
        ):
            raise ValueError("keep_ratio must be in (0, 1]")
        return config
