"""Fixed W4g128 scope with portable offline calibration controls."""

import math
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class GPTQConfig:
    calibration_samples: int = 32
    calibration_seed: int = 0
    damp_percent: float = 0.01
    block_size: int = 128

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown GPTQ parameter")
        c = cls(**parameters)
        if (
            type(c.calibration_samples) is not int
            or not 4 <= c.calibration_samples <= 512
            or c.calibration_samples % 4
        ):
            raise ValueError("GPTQ calibration count must be a multiple of four in [4, 512]")
        if type(c.calibration_seed) is not int or c.calibration_seed < 0:
            raise ValueError("Invalid GPTQ calibration seed")
        if (
            type(c.damp_percent) not in (int, float)
            or not math.isfinite(c.damp_percent)
            or not 0 < c.damp_percent <= 1
        ):
            raise ValueError("Invalid GPTQ damping")
        if type(c.block_size) is not int or c.block_size not in (32, 64, 128):
            raise ValueError("GPTQ update block must be 32, 64 or 128")
        return c

    def calibration_parameters(self):
        return dict(
            **asdict(self),
            bits=4,
            group_size=128,
            symmetric=True,
            act_order=False,
            true_sequential=True,
            static_groups=False,
        )
