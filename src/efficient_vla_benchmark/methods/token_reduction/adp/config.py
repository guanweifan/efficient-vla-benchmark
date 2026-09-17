"""ADP's released embedding-stage preset and observed-motion gate."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class ADPConfig:
    enabled: bool = True
    keep_ratio: float = 0.75
    view_weights: tuple[float, ...] = (0.4, 0.6)
    dynamic: bool = True
    decision: str = "adjacent"
    window: int = 3
    delta: str = "net"
    retain_previous_state: bool = False
    max_consecutive_pruned: int = 3

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown ADP parameters")
        values = dict(parameters)
        if "view_weights" in values:
            if not isinstance(values["view_weights"], (list, tuple)):
                raise ValueError("view_weights must be a sequence")
            values["view_weights"] = tuple(values["view_weights"])
        config = cls(**values)
        for name in ("enabled", "dynamic", "retain_previous_state"):
            if type(getattr(config, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if (
            type(config.keep_ratio) not in (int, float)
            or not math.isfinite(config.keep_ratio)
            or not 0 < config.keep_ratio <= 1
        ):
            raise ValueError("keep_ratio must be in (0, 1]")
        if (
            not config.view_weights
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or v < 0
                for v in config.view_weights
            )
            or sum(config.view_weights) <= 0
        ):
            raise ValueError("view_weights must be finite nonnegative weights with positive sum")
        if type(config.window) is not int or config.window < 3:
            raise ValueError("window must be an integer >= 3")
        if type(config.max_consecutive_pruned) is not int or config.max_consecutive_pruned < 0:
            raise ValueError("max_consecutive_pruned must be a nonnegative integer (0 disables)")
        if config.decision not in ("adjacent", "avg") or config.delta not in ("net", "sum"):
            raise ValueError("Unsupported motion gate rule")
        return config
