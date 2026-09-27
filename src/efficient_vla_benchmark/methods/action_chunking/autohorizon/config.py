"""Portable AutoHorizon parameters."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class AutoHorizonConfig:
    enabled: bool = True
    denoising_step: int = 3
    hold_threshold: float = 0.3
    entropy_quantile: float = 0.9
    run_length: int = 1
    max_horizon: int = 16

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown AutoHorizon parameter")
        config = cls(**parameters)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        for name in ("denoising_step", "run_length", "max_horizon"):
            if type(getattr(config, name)) is not int or getattr(config, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("hold_threshold", "entropy_quantile"):
            value = getattr(config, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if config.hold_threshold <= 0 or not 0 <= config.entropy_quantile <= 1:
            raise ValueError("Use a positive plateau threshold and a quantile in [0, 1]")
        return config
