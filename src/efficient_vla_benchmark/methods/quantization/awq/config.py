import math
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class AWQConfig:
    kernel_precision: str = "auto"
    bits: int = 4
    group_size: int = 128
    calibration_samples: int = 32
    calibration_seed: int = 0
    scale_grid: int = 20
    clip_grid: int = 20
    clip_max_shrink: float = 0.5
    clip_sample_tokens: int = 512

    def calibration_parameters(self):
        """Runtime execution precision does not change offline artifact identity."""
        values = asdict(self)
        values.pop("kernel_precision")
        return values

    @classmethod
    def parse(cls, parameters):
        allowed = {field.name for field in fields(cls)}
        if set(parameters) - allowed:
            raise ValueError(f"Unknown AWQ parameters: {sorted(set(parameters) - allowed)}")
        config = cls(**parameters)
        if not isinstance(config.kernel_precision, str) or config.kernel_precision not in {
            "auto",
            "bfloat16",
            "float16",
        }:
            raise ValueError("kernel_precision must be auto, bfloat16 or float16")
        if type(config.bits) is not int or config.bits != 4:
            raise ValueError("The pinned AWQ kernel supports four-bit weights")
        if type(config.group_size) is not int or config.group_size not in (32, 64, 128):
            raise ValueError("AWQ group size must be 32, 64 or 128")
        if (
            type(config.calibration_samples) is not int
            or not 4 <= config.calibration_samples <= 256
            or config.calibration_samples % 4
        ):
            raise ValueError("calibration_samples must be a multiple of four between 4 and 256")
        if type(config.calibration_seed) is not int or config.calibration_seed < 0:
            raise ValueError("calibration_seed must be a nonnegative integer")
        for name in ("scale_grid", "clip_grid", "clip_sample_tokens"):
            value = getattr(config, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            type(config.clip_max_shrink) not in (int, float)
            or not math.isfinite(config.clip_max_shrink)
            or not 0 < config.clip_max_shrink < 1
        ):
            raise ValueError("clip_max_shrink must be finite and between zero and one")
        if int(config.clip_max_shrink * config.clip_grid) < 1:
            raise ValueError("Clipping configuration must evaluate at least one candidate")
        return config
