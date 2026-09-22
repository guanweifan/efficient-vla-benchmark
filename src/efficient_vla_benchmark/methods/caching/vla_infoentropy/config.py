import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class InfoEntropyConfig:
    enabled: bool = True
    attention_backend: str = "eager"
    static_budget_per_view: int = 150
    similarity_threshold: float = 0.996
    low_quota: int = 40
    high_quota: int = 60
    transition_steps: int = 100
    reuse_layers: tuple[int, ...] = (2, 6, 9, 11)
    growth_factor: float = 0.55

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown VLA-InfoEntropy parameter")
        values = dict(parameters)
        if "reuse_layers" in values:
            if not isinstance(values["reuse_layers"], (list, tuple)):
                raise ValueError("reuse_layers must be a sequence")
            values["reuse_layers"] = tuple(values["reuse_layers"])
        config = cls(**values)
        if type(config.enabled) is not bool or config.attention_backend not in ("sdpa", "eager"):
            raise ValueError("Invalid enabled/attention_backend")
        for name in ("static_budget_per_view", "low_quota", "high_quota"):
            if type(getattr(config, name)) is not int or not 0 <= getattr(config, name) <= 256:
                raise ValueError(f"{name} must be in [0, 256]")
        if config.low_quota > config.high_quota:
            raise ValueError("low_quota exceeds high_quota")
        if type(config.transition_steps) is not int or config.transition_steps < 1:
            raise ValueError("transition_steps must be positive")
        layers = config.reuse_layers
        if (
            not layers
            or any(type(i) is not int or not 0 <= i < 32 for i in layers)
            or tuple(sorted(set(layers))) != layers
        ):
            raise ValueError("reuse_layers must be sorted unique native Llama layer indices")
        for name, low in (("similarity_threshold", -1), ("growth_factor", 0)):
            value = getattr(config, name)
            if type(value) not in (float, int) or not math.isfinite(value) or not low <= value <= 1:
                raise ValueError(f"Invalid {name}")
        return config
