import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class VoteConfig:
    enabled: bool = True
    similarity_threshold: float = 0.5
    history_chunks: int = 1

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown VOTE ensemble parameter")
        config = cls(**parameters)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if type(config.history_chunks) is not int or config.history_chunks < 1:
            raise ValueError("history_chunks must be a positive integer")
        value = config.similarity_threshold
        if type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value < 1:
            raise ValueError("similarity_threshold must be in [-1, 1)")
        return config
