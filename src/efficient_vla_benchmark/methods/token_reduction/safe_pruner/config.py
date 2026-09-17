"""Explicit parameters corresponding to the released SAFE-Pruner OFT preset."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class SAFEPrunerConfig:
    enabled: bool = True
    keep_ratio: float = 0.25
    pruning_layer: int = 3
    reference_layers: tuple[int, ...] = (0, 31)
    forecast_weight: float = 1.0
    similarity_threshold: float = 0.92

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown SAFE-Pruner parameters")
        values = dict(parameters)
        if "reference_layers" in values:
            if not isinstance(values["reference_layers"], (list, tuple)):
                raise ValueError("reference_layers must be a sequence")
            values["reference_layers"] = tuple(values["reference_layers"])
        cfg = cls(**values)
        if type(cfg.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        for name in ("keep_ratio", "forecast_weight", "similarity_threshold"):
            v = getattr(cfg, name)
            if type(v) not in (int, float) or not math.isfinite(v):
                raise ValueError(f"{name} must be a finite number")
        if not 0 < cfg.keep_ratio <= 1 or cfg.forecast_weight < 0:
            raise ValueError("Invalid retention or forecast weight")
        if not -1 <= cfg.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be in [-1, 1]")
        if type(cfg.pruning_layer) is not int or cfg.pruning_layer < 2:
            raise ValueError("pruning_layer must be >= 2, after the layer-zero gate")
        layers = cfg.reference_layers
        if len(layers) < 2 or any(type(i) is not int or i < 0 for i in layers):
            raise ValueError("reference_layers requires at least two nonnegative indices")
        if layers[0] != 0 or tuple(sorted(set(layers))) != layers or layers[-1] < cfg.pruning_layer:
            raise ValueError("reference_layers must start at zero and include a deep layer")
        return cfg
