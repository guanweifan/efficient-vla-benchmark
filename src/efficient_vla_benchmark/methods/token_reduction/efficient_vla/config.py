"""EfficientVLA visual-token core; no layer pruning or diffusion caching."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class EfficientVLAConfig:
    enabled: bool = True
    pruning_layer: int = 1
    tokens_per_view: int = 128
    key_tokens_per_view: int = 8
    task_fraction: float = 0.5

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown EfficientVLA parameter")
        config = cls(**parameters)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if type(config.pruning_layer) is not int or not 0 <= config.pruning_layer < 31:
            raise ValueError("pruning_layer must be in [0, 30]")
        for name in ("tokens_per_view", "key_tokens_per_view"):
            value = getattr(config, name)
            if type(value) is not int or not 1 <= value <= 256:
                raise ValueError(f"{name} must be in [1, 256]")
        if config.key_tokens_per_view > config.tokens_per_view:
            raise ValueError("Key budget exceeds total budget")
        if (
            type(config.task_fraction) not in (int, float)
            or not math.isfinite(config.task_fraction)
            or not 0 <= config.task_fraction <= 1
        ):
            raise ValueError("task_fraction must be in [0, 1]")
        return config
