"""Strict portable presets; model restrictions belong to optional plugins."""

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")


def merge_settings(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = merge_settings(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def read_preset(path: Path, stack: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in stack:
        raise ValueError("Cyclic preset inheritance")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Experiment config must be a mapping")
    parents = data.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or any(not isinstance(item, str) for item in parents):
        raise ValueError("extends must name relative preset files")
    result = {}
    for parent in parents:
        if Path(parent).is_absolute() or parent.startswith("~"):
            raise ValueError("Preset references must be portable relative paths")
        result = merge_settings(result, read_preset(path.parent / parent, (*stack, path)))
    return merge_settings(result, data)


def validate_portable(value):
    if isinstance(value, dict):
        for item in value.values():
            validate_portable(item)
    elif isinstance(value, list):
        for item in value:
            validate_portable(item)
    elif isinstance(value, str) and (
        value.startswith(("/", "~", "\\\\")) or re.match(r"^[A-Za-z]:[\\/]", value)
    ):
        raise ValueError("External resource paths belong in CLI arguments, not presets")


@dataclass(frozen=True)
class ExperimentConfig:
    backend: str
    method: str = "none"
    precision: str = "auto"
    execution_horizon: int = 1
    parameters: dict[str, Any] = field(default_factory=dict)
    model_variant: str = "default"
    backend_parameters: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    profiling: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for value in (self.backend, self.method, self.model_variant):
            if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
                raise ValueError("Supply exactly one snake_case backend, method and model variant")
        if not isinstance(self.precision, str) or not self.precision:
            raise ValueError("Precision must be a nonempty string validated by the backend")
        if type(self.execution_horizon) is not int or self.execution_horizon < 1:
            raise ValueError("Execution horizon must be a positive integer")
        for field_name in ("parameters", "backend_parameters", "evaluation", "profiling"):
            value = getattr(self, field_name)
            if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
                raise ValueError(f"{field_name} must be a string-keyed mapping")
            validate_portable(value)
        eval_keys = {"suites", "tasks", "trials", "seed", "record_traces"}
        timing_keys = {"repeats", "warmup", "mode"}
        if set(self.evaluation) - eval_keys or set(self.profiling) - timing_keys:
            raise ValueError("Unknown evaluation or profiling setting")

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        data = read_preset(path)
        if set(data) - set(cls.__dataclass_fields__):
            raise ValueError("Unknown experiment configuration keys")
        if "backend" not in data:
            raise ValueError("An explicit backend is required")
        return cls(**data)
