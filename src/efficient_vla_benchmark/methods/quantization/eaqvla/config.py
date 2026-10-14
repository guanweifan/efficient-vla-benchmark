"""Portable calibration choices where EaqVLA leaves details unspecified."""

import math
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class EaqVLAConfig:
    vision_bits: int = 4
    language_bits: int = 8
    calibration_samples: int = 40
    calibration_tokens: int = 256
    calibration_seed: int = 0
    gptq_block_size: int = 128
    damp_percent: float = 0.01

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown EaqVLA parameter")
        c = cls(**parameters)
        if (
            type(c.vision_bits) is not int
            or type(c.language_bits) is not int
            or c.vision_bits not in (4, 8)
            or c.language_bits not in (4, 8)
        ):
            raise ValueError("Unsupported modular precision")
        if (
            type(c.calibration_samples) is not int
            or not 4 <= c.calibration_samples <= 512
            or c.calibration_samples % 4
        ):
            raise ValueError("Calibration count must be a positive multiple of four")
        if type(c.calibration_tokens) is not int or not 16 <= c.calibration_tokens <= 1024:
            raise ValueError("Invalid captured token budget")
        if type(c.calibration_seed) is not int or c.calibration_seed < 0:
            raise ValueError("Invalid seed")
        if type(c.gptq_block_size) is not int or c.gptq_block_size not in (64, 128, 256):
            raise ValueError("Invalid GPTQ block size")
        if (
            type(c.damp_percent) not in (float, int)
            or not math.isfinite(c.damp_percent)
            or not 0 < c.damp_percent <= 1
        ):
            raise ValueError("Invalid dampening")
        return c

    def calibration_parameters(self):
        return asdict(self)
