"""Paper-default probe and domain-level sensitivity; controls are explicit."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class ProbeFlowConfig:
    enabled: bool = True
    epsilon: float = 0.008
    probe_time: float = 0.5
    max_steps: int = 10
    step_increment: int = 2
    diagnostic_fixed_steps: int = 0

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown ProbeFlow parameter")
        config = cls(**parameters)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        for name in ("epsilon", "probe_time"):
            value = getattr(config, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if config.epsilon <= 0 or not 0 < config.probe_time < 1:
            raise ValueError("Positive sensitivity and an interior probe time are required")
        for name in ("max_steps", "step_increment", "diagnostic_fixed_steps"):
            if type(getattr(config, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if config.max_steps < 2 or config.step_increment < 1:
            raise ValueError("ProbeFlow requires at least two integration steps")
        if config.diagnostic_fixed_steps not in (0, *range(2, config.max_steps + 1)):
            raise ValueError("Invalid diagnostic-only fixed Euler budget")
        return config
