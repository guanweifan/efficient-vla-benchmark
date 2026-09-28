"""A3 official-code parameters; portable and independent of tensor runtimes."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class A3OfficialConfig:
    enabled: bool = True
    num_samples: int = 8
    num_denoising_steps: int = 10
    action_horizon: int = 16
    verification_chunk_size: int = 16
    confidence_alpha: float = 0.8
    confidence_epsilon: float = 1e-8
    delta_thresholds: tuple[float, ...] = (0.41, 0.35, 0.5, 0.5, 0.5, 0.5)
    threshold_scale: float = 1.0

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown A3 official-code parameter")
        parameters = dict(parameters)
        if "delta_thresholds" in parameters:
            if not isinstance(parameters["delta_thresholds"], (list, tuple)):
                raise ValueError("delta_thresholds must contain six positive values")
            parameters["delta_thresholds"] = tuple(parameters["delta_thresholds"])
        config = cls(**parameters)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if type(config.num_samples) is not int or not 2 <= config.num_samples <= 32:
            raise ValueError("A3 requires 2..32 independent candidates")
        if type(config.num_denoising_steps) is not int or not 2 <= config.num_denoising_steps <= 10:
            raise ValueError("num_denoising_steps must be an integer from 2 to 10")
        if type(config.action_horizon) is not int or not 2 <= config.action_horizon <= 16:
            raise ValueError("action_horizon must be 2..16 real native predictions")
        size = config.verification_chunk_size
        if type(size) is not int or size < 1 or config.action_horizon % size:
            raise ValueError("verification_chunk_size must divide action_horizon")
        if len(config.delta_thresholds) != 6:
            raise ValueError("delta_thresholds must contain six positive values")
        values = (
            config.confidence_alpha,
            config.confidence_epsilon,
            config.threshold_scale,
            *config.delta_thresholds,
        )
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Confidence and threshold parameters must be finite and positive")
        return config
