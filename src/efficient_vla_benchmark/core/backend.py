"""Optional backend package contract (API version 1)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

API_VERSION = 1


@dataclass(frozen=True)
class RuntimeResources:
    model_path: Path
    device: str = "cuda:0"
    artifact_path: Path | None = None
    dataset_path: Path | None = None
    artifact_dir: Path | None = None
    runtime_path: Path | None = None


@dataclass(frozen=True)
class BackendSpec:
    name: str
    factory: str
    validator: str
    variants: tuple[str, ...]
    capabilities: frozenset[str]
    api_version: int = API_VERSION


class RuntimeMonitor(Protocol):
    """Device synchronization and counters, not the timing boundaries themselves."""

    def synchronize(self) -> None: ...

    def reset_peak_memory(self) -> None: ...

    def peak_memory(self) -> dict[str, int | None]: ...

    def hardware(self) -> dict: ...
