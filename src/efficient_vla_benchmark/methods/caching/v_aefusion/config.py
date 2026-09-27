"""Fixed early stale-conditioning window with a mandatory fresh tail."""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class VAEFusionConfig:
    enabled: bool = True
    stale_steps: int = 5

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown V-AEFusion parameter")
        config = cls(**parameters)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if type(config.stale_steps) is not int or config.stale_steps < 0:
            raise ValueError("stale_steps must be a nonnegative integer")
        return config
