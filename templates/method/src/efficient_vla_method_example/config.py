import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    scale: float = 0.5

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {"scale"}:
            raise ValueError("Unknown example parameter")
        value = parameters.get("scale", 0.5)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("scale must be finite")
        return cls(float(value))
