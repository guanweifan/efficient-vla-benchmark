"""DP-Cache stable-window output reuse; sampler transitions stay native."""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class DPCacheConfig:
    enabled: bool = True
    start_step: int = 2
    end_step: int = 7
    cache_interval: int = 2

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown DP-Cache parameter")
        config = cls(**parameters)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        for name in ("start_step", "end_step", "cache_interval"):
            if type(getattr(config, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 0 <= config.start_step <= config.end_step:
            raise ValueError("Invalid inclusive stable window")
        if config.cache_interval < 1:
            raise ValueError("cache_interval must be positive")
        return config
