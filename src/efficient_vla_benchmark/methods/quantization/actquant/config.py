"""Offline calibration and author GGUF allocation; no resource paths in configs."""

import math
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class ActQuantConfig:
    calibration_samples: int = 200
    calibration_seed: int = 0
    alpha: float = 0.5
    target_bpw: float = 4.5
    base_type: str = "IQ2_XS"
    max_type: str = "Q8_0"

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown ActQuant parameter")
        c = cls(**parameters)
        if (
            type(c.calibration_samples) is not int
            or not 8 <= c.calibration_samples <= 512
            or c.calibration_samples % 4
        ):
            raise ValueError("ActQuant calibration needs a multiple of four in [8,512]")
        if type(c.calibration_seed) is not int or c.calibration_seed < 0:
            raise ValueError("Invalid ActQuant seed")
        if type(c.alpha) not in (int, float) or not math.isfinite(c.alpha) or not 0 < c.alpha < 1:
            raise ValueError("ActQuant requires both action and NLL Fisher terms")
        if (
            type(c.target_bpw) not in (int, float)
            or not math.isfinite(c.target_bpw)
            or not 2.3125 <= c.target_bpw < 8.5
        ):
            raise ValueError("Invalid ActQuant block-weight budget")
        if c.base_type != "IQ2_XS" or c.max_type != "Q8_0":
            raise ValueError("This port fixes the author IQ2_XS-to-Q8_0 allocation menu")
        return c

    def calibration_parameters(self):
        return {
            k: v
            for k, v in asdict(self).items()
            if k in ("calibration_samples", "calibration_seed", "alpha")
        }


@dataclass(frozen=True)
class ActQuantFloatingConfig:
    @classmethod
    def parse(cls, parameters):
        if parameters:
            raise ValueError("The floating runtime control has no tunable parameters")
        return cls()
