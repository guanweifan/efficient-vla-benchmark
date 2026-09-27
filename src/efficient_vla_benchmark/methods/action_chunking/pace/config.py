"""Policy-agnostic phase-boundary selection parameters."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class PACEConfig:
    enabled: bool = True
    smoothing_window: int = 3
    minimum_spacing: int = 2
    calibration_percentile: float = 5.0
    max_horizon: int | None = None

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown PACE parameter")
        config = cls(**parameters)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        for name in ("smoothing_window", "minimum_spacing"):
            if type(getattr(config, name)) is not int or getattr(config, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if config.smoothing_window % 2 != 1:
            raise ValueError("Use an odd centered smoothing window")
        if config.max_horizon is not None and (
            type(config.max_horizon) is not int or config.max_horizon < 1
        ):
            raise ValueError("max_horizon must be null or a positive integer")
        if (
            type(config.calibration_percentile) not in (int, float)
            or not math.isfinite(config.calibration_percentile)
            or not 0 <= config.calibration_percentile <= 100
        ):
            raise ValueError("calibration_percentile must be in [0, 100]")
        return config

    def calibration_parameters(self):
        return dict(
            smoothing_window=self.smoothing_window,
            minimum_spacing=self.minimum_spacing,
            action_profile="libero_xyz_delta_norm",
            smoothing="centered_boxcar_edge_padding",
            valley="scipy_negative_profile_peaks_with_topographic_prominence",
            calibration_profile="complete_training_trajectory",
        )
