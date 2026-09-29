"""Falcon partial-denoising policy; budgets never alter the native action horizon."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class FalconConfig:
    enabled: bool = True
    distance_threshold: float = 0.01
    exploration_rate: float = 0.1
    temperature: float = 1.0
    min_noise_level: int = 10
    buffer_size: int = 20
    reference_horizon: int = 8
    selection_seed: int = 0

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {field.name for field in fields(cls)}:
            raise ValueError("Unknown Falcon parameter")
        config = cls(**parameters)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        for name in ("distance_threshold", "exploration_rate", "temperature"):
            value = getattr(config, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Invalid {name}")
        if config.distance_threshold <= 0 or config.temperature < 1e-6:
            raise ValueError("Threshold and temperature must be positive")
        if not 0 <= config.exploration_rate <= 1:
            raise ValueError("Exploration probability must be in [0, 1]")
        for name in ("min_noise_level", "buffer_size", "reference_horizon", "selection_seed"):
            if type(getattr(config, name)) is not int:
                raise ValueError(f"{name} must be integer")
        if config.min_noise_level < 0 or not 1 <= config.buffer_size <= 128:
            raise ValueError("Invalid noise floor or buffer capacity")
        if config.reference_horizon < 1 or not 0 <= config.selection_seed < 2**32:
            raise ValueError("Invalid executed horizon or selection seed")
        return config
