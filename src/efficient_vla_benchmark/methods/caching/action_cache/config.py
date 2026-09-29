from dataclasses import dataclass, fields


@dataclass(frozen=True)
class ActionCacheConfig:
    enabled: bool = True
    hit_threshold: float = 0.90
    refinement_steps: int = 2
    key_dim: int = 500
    projection_density: float = 0.01
    projection_seed: int = 42

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown ActionCache parameter")
        c = cls(**parameters)
        if type(c.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if type(c.hit_threshold) not in (int, float) or not -1 <= c.hit_threshold <= 1:
            raise ValueError("hit_threshold must be in [-1, 1]")
        for name in ("refinement_steps", "key_dim", "projection_seed"):
            if type(getattr(c, name)) is not int or getattr(c, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if c.key_dim < 1:
            raise ValueError("This binding requires positive key_dim")
        if type(c.projection_density) not in (int, float) or not 0 < c.projection_density <= 1:
            raise ValueError("projection_density must be in (0, 1]")
        return c
