"""Portable SmoothQuant settings, without runtime dependencies."""

import math
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class SmoothQuantConfig:
    alpha: float = 0.85
    calibration_samples: int = 32
    calibration_seed: int = 0

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown SmoothQuant parameter")
        c = cls(**parameters)
        if type(c.alpha) not in (int, float) or not math.isfinite(c.alpha) or not 0 <= c.alpha <= 1:
            raise ValueError("SmoothQuant alpha must be finite in [0, 1]")
        if (
            type(c.calibration_samples) is not int
            or not 4 <= c.calibration_samples <= 512
            or c.calibration_samples % 4
        ):
            raise ValueError("Calibration count must be a multiple of four in [4, 512]")
        if type(c.calibration_seed) is not int or c.calibration_seed < 0:
            raise ValueError("Invalid calibration seed")
        return c

    def calibration_parameters(self):
        return asdict(self)
