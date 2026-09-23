"""Independent ASR deletion instrument; no composition with another paper."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class ASRConfig:
    enabled: bool = True
    keep_ratio: float = 0.1
    gate_layers: tuple[int, ...] = (15,)
    actuation_window_ms: float = 400.0

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {field.name for field in fields(cls)}:
            raise ValueError("Unknown ASR deletion parameter")
        parameters = dict(parameters)
        if "gate_layers" in parameters:
            value = parameters["gate_layers"]
            if not isinstance(value, (tuple, list)):
                raise ValueError("gate_layers must be a sequence")
            parameters["gate_layers"] = tuple(value)
        config = cls(**parameters)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if (
            type(config.keep_ratio) not in (float, int)
            or not math.isfinite(config.keep_ratio)
            or not 0 < config.keep_ratio <= 1
        ):
            raise ValueError("keep_ratio must be in (0, 1]")
        if (
            not config.gate_layers
            or any(type(x) is not int or not 0 <= x < 32 for x in config.gate_layers)
            or tuple(sorted(set(config.gate_layers))) != config.gate_layers
        ):
            raise ValueError("gate_layers must be unique increasing decoder indices")
        if (
            type(config.actuation_window_ms) not in (int, float)
            or not math.isfinite(config.actuation_window_ms)
            or config.actuation_window_ms <= 0
        ):
            raise ValueError("actuation_window_ms must be positive")
        return config
