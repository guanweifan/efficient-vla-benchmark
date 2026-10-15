import math
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class DAMPAConfig:
    runtime: str = "tile_float"
    calibration_samples: int = 512
    calibration_seed: int = 0
    probe_steps: int = 16
    damping: float = 0.0003
    translation_weight: float = 1.8
    rotation_weight: float = 0.15
    retention_ratio: float = 0.3
    exclude_final_blocks: int = 2
    activation_percentile: float = 99.9

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown DA-MPA parameter")
        c = cls(**parameters)
        if c.runtime not in ("tile_float", "tinychat_fp16"):
            raise ValueError("Unsupported DA-MPA runtime")
        for name in (
            "calibration_samples",
            "calibration_seed",
            "probe_steps",
            "exclude_final_blocks",
        ):
            if type(getattr(c, name)) is not int:
                raise ValueError("DA-MPA counts must be integers")
        if (
            not 4 <= c.calibration_samples <= 512
            or c.calibration_samples % 4
            or c.calibration_seed < 0
            or not 4 <= c.probe_steps <= c.calibration_samples
            or c.probe_steps % 4
            or not 0 <= c.exclude_final_blocks < 12
        ):
            raise ValueError("Invalid DA-MPA calibration/probe count")
        for name in (
            "damping",
            "translation_weight",
            "rotation_weight",
            "retention_ratio",
            "activation_percentile",
        ):
            value = getattr(c, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("DA-MPA numerical parameters must be finite positive")
        if c.retention_ratio >= 1 or c.activation_percentile > 100:
            raise ValueError("DA-MPA core cannot be disabled")
        return c

    def calibration_parameters(self):
        d = asdict(self)
        # Allocation may select another retention from the SAME frozen scores.
        d.pop("retention_ratio")
        # Native-only calibration does not execute the selected packed engine.
        # Preserve its schema-1 identity; every engine still needs fresh Dev/timing.
        d.pop("runtime")
        return d
