"""Paper-based TEAM-VLA parameters; one preset across all four suites."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class TEAMVLAConfig:
    enabled: bool = True
    context_ratio: float = 0.25
    expansion_kernel: int = 3
    density_threshold: float = 1.0
    merging_layer: int = 15
    merge_tokens: int = 80
    rms_epsilon: float = 1e-6
    seed: int = 0

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown TEAM-VLA parameters")
        cfg = cls(**parameters)
        if type(cfg.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        for key in ("context_ratio", "density_threshold", "rms_epsilon"):
            value = getattr(cfg, key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if not 0 <= cfg.context_ratio <= 1 or cfg.density_threshold < 0 or cfg.rms_epsilon <= 0:
            raise ValueError("Invalid sampling, density or RMS normalization parameter")
        for key in ("expansion_kernel", "merging_layer", "merge_tokens", "seed"):
            if type(getattr(cfg, key)) is not int:
                raise ValueError(f"{key} must be an integer")
        if cfg.expansion_kernel not in (1, 3, 5, 7) or cfg.merging_layer < 0:
            raise ValueError("Invalid expansion kernel or merging layer")
        if not 1 <= cfg.merge_tokens <= 512 or not 0 <= cfg.seed < 2**32:
            raise ValueError("Invalid merging budget or seed")
        return cfg
