from dataclasses import dataclass, fields


@dataclass(frozen=True)
class TemporalReuseConfig:
    enabled: bool = True
    update_ratio: float = 0.5

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown temporal visual reuse parameter")
        c = cls(**parameters)
        if type(c.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if type(c.update_ratio) not in (int, float) or not 0 < c.update_ratio <= 1:
            raise ValueError("update_ratio must be in (0, 1]")
        return c
