"""Portable KeyStone parameters, without importing a tensor runtime."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class KeyStoneConfig:
    enabled: bool = True
    num_samples: int = 4
    num_clusters: int = 2
    unimodal_tau: float = 0.3
    selection_horizon: int = 8

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {field.name for field in fields(cls)}:
            raise ValueError("Unknown KeyStone parameter")
        config = cls(**parameters)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        for name in ("num_samples", "num_clusters", "selection_horizon"):
            if type(getattr(config, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 3 <= config.num_samples <= 32:
            raise ValueError("Use 3..32 candidates; K=1/2 is not meaningful consensus")
        if not 2 <= config.num_clusters < config.num_samples:
            raise ValueError("Clusters must be between 2 and K-1")
        if config.selection_horizon < 1:
            raise ValueError("selection_horizon must be positive")
        if (
            type(config.unimodal_tau) not in (int, float)
            or not math.isfinite(config.unimodal_tau)
            or config.unimodal_tau < 0
        ):
            raise ValueError("unimodal_tau must be finite and nonnegative")
        return config
