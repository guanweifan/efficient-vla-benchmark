from dataclasses import dataclass, fields


@dataclass(frozen=True)
class C3acheConfig:
    enabled: bool = True
    refresh_interval: int = 4
    cache_steps: int = 7
    live_prefix_blocks: int = 1

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown C3ache parameter")
        c = cls(**parameters)
        if type(c.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        for name in ("refresh_interval", "cache_steps", "live_prefix_blocks"):
            if type(getattr(c, name)) is not int or getattr(c, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        return c
