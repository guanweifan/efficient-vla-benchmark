"""Portable ILSA budgets; the selected backend validates its actual depth."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class ILSAConfig:
    enabled: bool = True
    pruning_layer: int = 3
    keep_ratio: float = 0.65
    recycle_ratio: float = 0.1

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {field.name for field in fields(cls)}:
            raise ValueError("Unknown ETA-VLA ILSA parameter")
        config = cls(**parameters)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if type(config.pruning_layer) is not int or config.pruning_layer < 0:
            raise ValueError("pruning_layer must be a nonnegative index")
        for name in ("keep_ratio", "recycle_ratio"):
            value = getattr(config, name)
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"Invalid {name}")
        if config.recycle_ratio == 1:
            raise ValueError("Both semantic selection and diversity recycling must remain active")
        return config
