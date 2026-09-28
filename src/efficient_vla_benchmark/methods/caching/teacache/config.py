"""Official timestep-embedding variant with independent polynomial calibration."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class TeaCacheConfig:
    enabled: bool = True
    threshold: float = 0.25
    warmup_steps: int = 2
    polynomial_degree: int = 4
    calibration_samples: int = 32
    calibration_seed: int = 0

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {field.name for field in fields(cls)}:
            raise ValueError("Unknown TeaCache parameter")
        config = cls(**parameters)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if (
            type(config.threshold) not in (int, float)
            or not math.isfinite(config.threshold)
            or config.threshold < 0
        ):
            raise ValueError("threshold must be finite and nonnegative")
        for name in (
            "warmup_steps",
            "polynomial_degree",
            "calibration_samples",
            "calibration_seed",
        ):
            if type(getattr(config, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if config.warmup_steps < 1 or not 1 <= config.polynomial_degree <= 4:
            raise ValueError("Require positive warmup and polynomial degree in [1, 4]")
        if config.calibration_samples < 4 or config.calibration_samples % 4:
            raise ValueError("Calibration must balance all four training suites")
        if not 0 <= config.calibration_seed < 2**32:
            raise ValueError("Invalid calibration seed")
        return config

    def calibration_parameters(self):
        return dict(
            samples=self.calibration_samples,
            seed=self.calibration_seed,
            polynomial_degree=self.polynomial_degree,
        )
