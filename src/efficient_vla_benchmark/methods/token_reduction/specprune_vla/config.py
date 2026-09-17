"""Pinned-reference defaults, with explicit internal ablations."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class SpecPruneConfig:
    static_pruning: bool = True
    dynamic_pruning: bool = True
    action_controller: bool = True
    static_ratio: float = 0.8
    dynamic_keep_ratio: float = 0.9
    importance_beta: float = 0.2
    min_visual_tokens: int = 60
    importance_layers: tuple[int, ...] = (14, 19, 24)
    pruning_layers: tuple[int, ...] = (10, 15, 20, 25)
    global_layers: tuple[int, ...] = (14, 30)
    primary_similarity: float = 0.986
    wrist_similarity: float = 0.98
    primary_static_budget: int = 236
    wrist_static_budget: int = 230
    primary_precise_static_budget: int = 240
    wrist_precise_static_budget: int = 236
    velocity_threshold: float = 0.35
    rotation_threshold: float = 0.2
    xy_threshold: float = 0.1
    z_threshold: float = 0.1
    exit_z_threshold: float = 0.2
    exit_rotation_threshold: float = 0.3

    @classmethod
    def parse(cls, parameters: dict):
        allowed = {field.name for field in fields(cls)}
        if set(parameters) - allowed:
            raise ValueError(f"Unknown SpecPrune parameters: {sorted(set(parameters) - allowed)}")
        values = dict(parameters)
        for name in ("importance_layers", "pruning_layers", "global_layers"):
            if name in values:
                if not isinstance(values[name], (list, tuple)):
                    raise ValueError(f"{name} must be a sequence of layer indices")
                values[name] = tuple(values[name])
        config = cls(**values)
        for name in ("static_pruning", "dynamic_pruning", "action_controller"):
            if type(getattr(config, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("static_ratio", "dynamic_keep_ratio", "importance_beta"):
            value = getattr(config, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        for name in ("primary_similarity", "wrist_similarity"):
            value = getattr(config, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [-1, 1]")
        for name in ("importance_layers", "pruning_layers", "global_layers"):
            value = getattr(config, name)
            if not value or any(type(index) is not int or index < 0 for index in value):
                raise ValueError(f"{name} requires nonnegative layer indices")
            if tuple(sorted(set(value))) != value:
                raise ValueError(f"{name} must be sorted and unique")
        for name in allowed:
            value = getattr(config, name)
            if name.endswith("budget") and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer")
            if name.endswith("threshold") and (
                type(value) not in (int, float) or not math.isfinite(value) or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if type(config.min_visual_tokens) is not int or config.min_visual_tokens < 1:
            raise ValueError("min_visual_tokens must be a positive integer")
        return config

    def text_budgets(self, precise: bool):
        primary, wrist = int(30 * self.static_ratio), int(24 * self.static_ratio)
        if precise:
            primary = int(primary * self.static_ratio**2)
            wrist = int(wrist * self.static_ratio**2)
        return primary, wrist
