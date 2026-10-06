"""SparseVLM+ official V2.0 budgets and text-visual priority heads."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class SparseVLMConfig:
    enabled: bool = True
    pruning_layers: tuple[int, ...] = (2, 6, 15)
    budget_profile: int = 192
    budget_scale: float = 1.0
    priority_heads: int = 14

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {field.name for field in fields(cls)}:
            raise ValueError(
                "Unknown SparseVLM v2 parameter; v1 rank/recycling settings are retired"
            )
        values = dict(parameters)
        if "pruning_layers" in values:
            if not isinstance(values["pruning_layers"], (list, tuple)):
                raise ValueError("pruning_layers must be a sequence")
            values["pruning_layers"] = tuple(values["pruning_layers"])
        config = cls(**values)
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if (
            len(config.pruning_layers) != 3
            or any(type(x) is not int or not 0 <= x < 31 for x in config.pruning_layers)
            or tuple(sorted(set(config.pruning_layers))) != config.pruning_layers
        ):
            raise ValueError("pruning_layers must contain three increasing indices in [0, 30]")
        if type(config.budget_profile) is not int or config.budget_profile not in (
            64,
            96,
            128,
            192,
        ):
            raise ValueError("budget_profile must be an official 64/96/128/192 profile")
        if (
            type(config.budget_scale) not in (float, int)
            or not math.isfinite(config.budget_scale)
            or not 0 < config.budget_scale <= 4
        ):
            raise ValueError("budget_scale must be in (0, 4]")
        if type(config.priority_heads) is not int or config.priority_heads < 1:
            raise ValueError("priority_heads must be a positive integer")
        return config
