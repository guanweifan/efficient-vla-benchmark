import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class VLACacheConfig:
    attention_backend: str = "eager"
    static_budget_per_view: int = 150
    similarity_threshold: float = 0.996
    protected_tokens_per_view: int = 100
    task_attention_layer: int = 15
    reuse_layers: tuple[int, ...] = (2, 6, 9, 11)
    weighted_growth: bool = True
    growth_factor: float = 0.55

    @classmethod
    def parse(cls, parameters):
        allowed = {field.name for field in fields(cls)}
        if set(parameters) - allowed:
            raise ValueError(f"Unknown VLA-Cache parameters: {sorted(set(parameters) - allowed)}")
        values = dict(parameters)
        if "reuse_layers" in values:
            if not isinstance(values["reuse_layers"], (list, tuple)):
                raise ValueError("reuse_layers must be a sequence")
            values["reuse_layers"] = tuple(values["reuse_layers"])
        config = cls(**values)
        if not isinstance(config.attention_backend, str) or config.attention_backend not in {
            "eager",
            "sdpa",
        }:
            raise ValueError("attention_backend must be eager or sdpa")
        for name in ("static_budget_per_view", "protected_tokens_per_view"):
            value = getattr(config, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(config.task_attention_layer) is not int or config.task_attention_layer < 0:
            raise ValueError("task_attention_layer must be a nonnegative layer index")
        layers = config.reuse_layers
        if not layers or any(type(i) is not int or i < 0 for i in layers):
            raise ValueError("reuse_layers must contain nonnegative layer indices")
        if tuple(sorted(set(layers))) != layers:
            raise ValueError("reuse_layers must be sorted and unique")
        if type(config.weighted_growth) is not bool:
            raise ValueError("weighted_growth must be boolean")
        for name, low in (("similarity_threshold", -1), ("growth_factor", 0)):
            value = getattr(config, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= 1:
                raise ValueError(f"{name} has an invalid range")
        return config
