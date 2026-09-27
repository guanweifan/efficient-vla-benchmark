"""Portable, bounded high-order solver configuration."""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class DPMSolverPPConfig:
    enabled: bool = True
    num_steps: int = 6
    order: int = 2
    spacing: str = "time_uniform"

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {field.name for field in fields(cls)}:
            raise ValueError("Unknown DPM-Solver++ parameter")
        config = cls(**parameters)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if type(config.order) is not int or config.order not in (2, 3):
            raise ValueError("Primary solver order must be 2 or 3")
        minimum = 3 if config.order == 2 else 5
        if type(config.num_steps) is not int or not minimum <= config.num_steps <= 10:
            raise ValueError("Budget must activate high-order updates and not exceed native ten")
        if config.spacing not in ("time_uniform", "log_snr"):
            raise ValueError("Unknown timestep spacing")
        return config
