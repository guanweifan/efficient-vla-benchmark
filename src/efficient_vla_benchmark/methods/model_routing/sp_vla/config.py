import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class SPVLAConfig:
    enabled: bool = True
    scheduling: bool = True
    token_pruning: bool = False
    z_prune_threshold: float = 0.5
    max_prune_fraction: float = 0.9
    canny_low: int = 100
    canny_high: int = 200
    history_size: int = 6
    min_model_actions: int = 4
    gate_steps: int = 1
    max_z_xy_ratio: float = 0.4
    max_z: float = 0.3
    ridge: float = 0.0001
    translation_clip: float = 0.9

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown SP-VLA scheduling parameter")
        config = cls(**parameters)
        for name in ("enabled", "scheduling", "token_pruning"):
            if type(getattr(config, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("z_prune_threshold", "max_prune_fraction"):
            value = getattr(config, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must be finite in [0, 1)")
        if any(type(v) is not int for v in (config.canny_low, config.canny_high)) or not (
            0 <= config.canny_low < config.canny_high <= 255
        ):
            raise ValueError("Canny thresholds must be ordered integers in [0, 255]")
        for name in ("history_size", "min_model_actions", "gate_steps"):
            if type(getattr(config, name)) is not int or getattr(config, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 2 <= config.min_model_actions <= config.history_size:
            raise ValueError("Need at least two model actions within the history window")
        if config.gate_steps > config.history_size:
            raise ValueError("gate_steps exceeds history_size")
        for name in ("max_z_xy_ratio", "max_z", "ridge", "translation_clip"):
            value = getattr(config, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        return config
