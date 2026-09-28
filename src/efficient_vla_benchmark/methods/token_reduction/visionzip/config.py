"""Training-free VisionZip budgets, independent of native feature dimensions."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class VisionZipConfig:
    enabled: bool = True
    keep_ratio: float = 0.5
    dominant_ratio: float = 0.84375

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {field.name for field in fields(cls)}:
            raise ValueError("Unknown VisionZip parameter")
        config = cls(**parameters)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        for name in ("keep_ratio", "dominant_ratio"):
            value = getattr(config, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"Invalid {name}")
        if config.dominant_ratio == 1:
            raise ValueError("VisionZip must preserve a contextual-token contribution")
        return config
