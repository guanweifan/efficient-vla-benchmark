import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class SPVLAConfig:
    enabled: bool = True
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
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
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
