import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class VoteConfig:
    enabled: bool = True
    similarity_threshold: float = 0.5
    max_history: int = 3

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown VOTE ensemble parameter")
        config = cls(**parameters)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if type(config.max_history) is not int or config.max_history < 1:
            raise ValueError("max_history must be a positive integer")
        value = config.similarity_threshold
        if type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value < 1:
            raise ValueError("similarity_threshold must be in [-1, 1)")
        return config
