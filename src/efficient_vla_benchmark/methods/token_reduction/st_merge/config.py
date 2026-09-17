import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class STMergeConfig:
    enabled: bool = True
    merging_layer: int = 1
    tokens_per_view: int = 128
    queues: int = 4
    spatial_sigma: float = 2.0

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown ST-Merge parameter")
        config = cls(**parameters)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if type(config.merging_layer) is not int or not 0 <= config.merging_layer <= 20:
            raise ValueError("merging_layer must precede both native vision readouts")
        if type(config.queues) is not int or not 2 <= config.queues <= 16:
            raise ValueError("queues must be in [2, 16]")
        if type(config.tokens_per_view) is not int or not (
            math.ceil(256 / config.queues) <= config.tokens_per_view <= 256
        ):
            raise ValueError("Token budget must retain the anchor queue")
        if (
            type(config.spatial_sigma) not in (int, float)
            or not math.isfinite(config.spatial_sigma)
            or config.spatial_sigma <= 0
        ):
            raise ValueError("spatial_sigma must be finite and positive")
        return config
