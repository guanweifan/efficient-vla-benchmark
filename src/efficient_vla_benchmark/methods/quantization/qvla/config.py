"""Pinned public-code QVLA calibration and channel budget."""

import math
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class QVLAConfig:
    target_avg_bits: float = 4.0
    candidate_bits: tuple = (0, 2, 4, 8, 16)
    calibration_samples: int = 32
    calibration_seed: int = 0
    percdamp: float = 0.01

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown QVLA parameter")
        values = dict(parameters)
        if "candidate_bits" in values:
            if not isinstance(values["candidate_bits"], (list, tuple)):
                raise ValueError("candidate_bits must be a sequence")
            values["candidate_bits"] = tuple(values["candidate_bits"])
        c = cls(**values)
        if (
            any(type(b) is not int or b not in (0, 2, 4, 8, 16) for b in c.candidate_bits)
            or len(set(c.candidate_bits)) != len(c.candidate_bits)
            or not c.candidate_bits
            or max(c.candidate_bits) != 16
            or len(c.candidate_bits) < 2
        ):
            raise ValueError("QVLA needs distinct supported candidates including native 16-bit")
        if (
            type(c.target_avg_bits) not in (int, float)
            or not math.isfinite(c.target_avg_bits)
            or not min(c.candidate_bits) <= c.target_avg_bits < 16
        ):
            raise ValueError("QVLA budget must require an active quantized allocation")
        if (
            type(c.calibration_samples) is not int
            or not 4 <= c.calibration_samples <= 512
            or c.calibration_samples % 4
        ):
            raise ValueError("Calibration samples must be a multiple of four in [4,512]")
        if type(c.calibration_seed) is not int or c.calibration_seed < 0:
            raise ValueError("Invalid seed")
        if (
            type(c.percdamp) not in (int, float)
            or not math.isfinite(c.percdamp)
            or not 0 < c.percdamp <= 1
        ):
            raise ValueError("Invalid Hessian damping")
        return c

    def calibration_parameters(self):
        result = asdict(self)
        result["candidate_bits"] = list(self.candidate_bits)
        return result
