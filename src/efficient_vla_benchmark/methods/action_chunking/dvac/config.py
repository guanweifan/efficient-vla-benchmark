"""DVAC portable parameters; no tensor runtime at discovery."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class DVACConfig:
    enabled: bool = True
    alpha: float = 2.0
    tail_steps: int = 5
    window_size: int = 5
    min_horizon: int = 1
    max_horizon: int = 16

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown DVAC parameter")
        config = cls(**parameters)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if (
            type(config.alpha) not in (int, float)
            or not math.isfinite(config.alpha)
            or config.alpha < 0
        ):
            raise ValueError("alpha must be finite and nonnegative")
        for name in ("tail_steps", "window_size", "min_horizon", "max_horizon"):
            if type(getattr(config, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if config.tail_steps < 2 or config.window_size < 1:
            raise ValueError("Use at least two clean predictions and one historical query")
        if not 1 <= config.min_horizon <= config.max_horizon:
            raise ValueError("Invalid execution bounds")
        return config
