"""TIES revised-paper parameters; no learned selector or per-task settings."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class TIESConfig:
    enabled: bool = True
    keep_ratio: float = 0.5
    pruning_layer: int = 2
    ranking_layers: int = 16
    ranking_topk: int = 30
    refresh_steps: int = 24
    mode: str = "soft"
    uniform_seed: int = 0
    calibration_samples: int = 100
    calibration_seed: int = 0

    @classmethod
    def parse(cls, values):
        if set(values) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown TIES parameter")
        c = cls(**values)
        if type(c.enabled) is not bool or c.mode not in ("hard", "soft"):
            raise ValueError("Invalid TIES enabled flag or mode")
        if (
            type(c.keep_ratio) not in (int, float)
            or not math.isfinite(c.keep_ratio)
            or not 0 < c.keep_ratio <= 1
        ):
            raise ValueError("keep_ratio must be in (0,1]")
        for name in (
            "pruning_layer",
            "ranking_layers",
            "ranking_topk",
            "refresh_steps",
            "calibration_samples",
        ):
            if type(getattr(c, name)) is not int or getattr(c, name) < 1:
                raise ValueError(f"{name} must be positive integer")
        if c.ranking_layers < 2 or c.ranking_topk < 2 or c.calibration_samples % 4:
            raise ValueError("TIES needs adjacent layers, rank pairs and balanced training frames")
        for name in ("uniform_seed", "calibration_seed"):
            if type(getattr(c, name)) is not int or not 0 <= getattr(c, name) < 2**32:
                raise ValueError(f"Invalid {name}")
        return c

    def calibration_parameters(self):
        return dict(
            samples=self.calibration_samples,
            seed=self.calibration_seed,
            ranking_layers=self.ranking_layers,
            ranking_topk=self.ranking_topk,
        )
