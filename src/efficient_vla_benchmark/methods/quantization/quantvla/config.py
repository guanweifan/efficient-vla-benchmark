"""Calibration identity for the CogACT port of the pinned official QuantVLA code."""

import math
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class QuantVLAConfig:
    block_size: int = 64
    enable_permutation: bool = False
    energy_smoothing: float = 0.15
    activation_percentile: float = 99.9
    calibration_samples: int = 32
    attention_calibration_samples: int = 128
    calibration_seed: int = 0
    alpha_min: float = 0.7
    alpha_max: float = 1.4
    alpha_neutral: float = 0.02
    beta_log_clip: float = 0.3
    beta_neutral: float = 0.03

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown QuantVLA parameter")
        c = cls(**parameters)
        if type(c.block_size) is not int or c.block_size not in (16, 32, 64, 128):
            raise ValueError("Unsupported block size")
        if type(c.enable_permutation) is not bool:
            raise ValueError("enable_permutation must be boolean")
        for name in ("calibration_samples", "attention_calibration_samples"):
            n = getattr(c, name)
            if type(n) is not int or not 4 <= n <= 512 or n % 4:
                raise ValueError("Calibration counts must be multiples of four in [4,512]")
        if type(c.calibration_seed) is not int or c.calibration_seed < 0:
            raise ValueError("Invalid calibration seed")
        for name in (
            "energy_smoothing",
            "activation_percentile",
            "alpha_min",
            "alpha_max",
            "alpha_neutral",
            "beta_log_clip",
            "beta_neutral",
        ):
            x = getattr(c, name)
            if type(x) not in (int, float) or not math.isfinite(x):
                raise ValueError("QuantVLA scalar must be finite")
        if not 0 <= c.energy_smoothing <= 1 or not 0 < c.activation_percentile <= 100:
            raise ValueError("Invalid transform/percentile parameter")
        if not 0 < c.alpha_min <= 1 <= c.alpha_max or not 0 <= c.alpha_neutral < 1:
            raise ValueError("Invalid ATM range")
        if not 0 <= c.beta_neutral <= c.beta_log_clip <= 1:
            raise ValueError("Invalid OHB range")
        return c

    def calibration_parameters(self):
        return asdict(self)
