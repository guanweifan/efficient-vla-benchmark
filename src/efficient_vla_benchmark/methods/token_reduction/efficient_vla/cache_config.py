"""Independent DiT caching core, not token/layer pruning or step reduction."""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class DiffusionCacheConfig:
    enabled: bool = True
    cache_interval: int = 5

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {field.name for field in fields(cls)}:
            raise ValueError("Unknown EfficientVLA diffusion-cache parameter")
        config = cls(**parameters)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if type(config.cache_interval) is not int or not 1 <= config.cache_interval <= 100:
            raise ValueError("cache_interval must be an integer in [1, 100]")
        return config
