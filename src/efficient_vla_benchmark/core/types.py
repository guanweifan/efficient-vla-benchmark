"""The single interface shared by control, evaluation and profiling."""

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class EpisodeContext:
    suite: str
    task_id: int
    episode_id: int
    seed: int = 0


@dataclass(frozen=True)
class Observation:
    """Environment-oriented RGB cameras and raw proprioception.

    The LIBERO adapter supplies two cameras and 8D state. Other environments may
    supply a different state width; the selected backend validates its requirements.
    """

    image: np.ndarray
    wrist_image: np.ndarray
    proprio: np.ndarray
    instruction: str
    depth: np.ndarray | None = None

    def validate(self) -> None:
        for image in (self.image, self.wrist_image):
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError("Cameras must be HxWx3 uint8 arrays")
        if self.proprio.ndim != 1 or not np.isfinite(self.proprio).all():
            raise ValueError("Proprioception must be a finite vector")
        if not self.instruction.strip():
            raise ValueError("A nonempty language instruction is required")
        if self.depth is not None and (
            self.depth.shape != self.image.shape[:2]
            or self.depth.dtype != np.float32
            or not np.isfinite(self.depth).all()
            or not (self.depth > 0).all()
        ):
            raise ValueError(
                "Depth must be positive finite float32 meters aligned with primary RGB"
            )


@dataclass
class ActionDecision:
    """One environment-ready action, in the selected environment's convention."""

    action: np.ndarray
    queried: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)


class ChunkPredictor(Protocol):
    def reset(self, context: EpisodeContext) -> None: ...

    def observe(self, observation: Observation) -> None: ...

    def after_action(self, action: np.ndarray, remaining: int) -> bool: ...

    def predict_chunk(self, observation: Observation) -> np.ndarray: ...

    def metadata(self) -> dict[str, Any]: ...


class Policy(Protocol):
    def reset(self, context: EpisodeContext) -> None: ...

    def step(self, observation: Observation) -> ActionDecision: ...
