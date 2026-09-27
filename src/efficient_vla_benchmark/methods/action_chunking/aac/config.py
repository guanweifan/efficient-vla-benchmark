"""Portable AAC search parameters; no model or tensor imports."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class AACConfig:
    enabled: bool = True
    num_samples: int = 20
    movement_threshold: float = 3.0
    covariance_regularization: float = 1e-6
    gripper_movement_weight: float = 0.2

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown AAC parameter")
        config = cls(**parameters)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if type(config.num_samples) is not int or not 4 <= config.num_samples <= 40:
            raise ValueError("Use 4..40 samples for empirical three-dimensional covariance")
        for name in ("movement_threshold", "covariance_regularization", "gripper_movement_weight"):
            value = getattr(config, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if config.covariance_regularization == 0:
            raise ValueError("Covariance regularization must be positive")
        return config
