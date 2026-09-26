from dataclasses import dataclass, fields
from math import isfinite


@dataclass(frozen=True)
class EcoVLAConfig:
    enabled: bool = True
    keep_ratio: float = 0.75
    temporal_inertia: float = 0.7
    history_momentum: float = 0.9
    trigger_quantile: float = 0.05
    history_window: int = 16
    calibration_samples: int = 32
    calibration_seed: int = 0

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {field.name for field in fields(cls)}:
            raise ValueError("Unknown EcoVLA parameter")
        config = cls(**parameters)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        for name in ("keep_ratio", "temporal_inertia", "history_momentum", "trigger_quantile"):
            value = getattr(config, name)
            if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite in [0, 1]")
        if config.keep_ratio == 0 or config.temporal_inertia == 1 or config.history_momentum == 1:
            raise ValueError("Retention must be positive and temporal weights less than one")
        for name in ("history_window", "calibration_samples"):
            if type(getattr(config, name)) is not int or getattr(config, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if config.calibration_samples % 4:
            raise ValueError("Calibration samples must be divisible by four")
        if type(config.calibration_seed) is not int or config.calibration_seed < 0:
            raise ValueError("Calibration seed must be a nonnegative integer")
        return config

    def calibration_parameters(self):
        return {"samples": self.calibration_samples, "seed": self.calibration_seed}
