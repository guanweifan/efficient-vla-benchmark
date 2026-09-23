"""DivPrune's visual retention budget."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class DivPruneConfig:
    enabled: bool = True
    keep_ratio: float = 0.5

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {field.name for field in fields(cls)}:
            raise ValueError("Unknown DivPrune parameter")
        config = cls(**parameters)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if (
            type(config.keep_ratio) not in (float, int)
            or not math.isfinite(config.keep_ratio)
            or not 0 < config.keep_ratio <= 1
        ):
            raise ValueError("keep_ratio must be in (0, 1]")
        return config
