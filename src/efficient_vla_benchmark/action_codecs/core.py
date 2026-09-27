"""CPU-only codec contract. Decoder input excludes future observations/actions."""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class Context:
    suite: str
    initial_state: np.ndarray

    def __post_init__(self):
        state = np.array(self.initial_state, dtype=np.float32, copy=True)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Expected one finite LIBERO state, not a future state sequence")
        state.flags.writeable = False
        object.__setattr__(self, "initial_state", state)


@dataclass(frozen=True)
class Packet:
    """Discrete payload only. Shapes and vocabulary are explicit fixed side information."""

    ids: np.ndarray
    vocabulary: int
    horizon: int

    def __post_init__(self):
        ids = np.asarray(self.ids)
        if ids.size == 0 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("Codec payload must contain integer tokens")
        if self.vocabulary < 2 or self.horizon < 1:
            raise ValueError("Invalid vocabulary/horizon")
        if np.any(ids < 0) or np.any(ids >= self.vocabulary):
            raise ValueError("Token outside declared vocabulary")
        ids = np.array(ids, dtype=np.int64, copy=True)
        ids.flags.writeable = False
        object.__setattr__(self, "ids", ids)

    @property
    def token_count(self) -> int:
        return self.ids.size

    @property
    def nominal_bits(self) -> int:
        """Fixed-width code payload, NOT entropy coding or an actual serialized size."""
        return self.token_count * (self.vocabulary - 1).bit_length()


def actions_checked(actions, horizon=None):
    value = np.asarray(actions, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 7 or value.shape[0] < 1:
        raise ValueError("Expected nonempty [frames, 7] raw LIBERO control actions")
    if horizon is not None and value.shape[0] != horizon:
        raise ValueError("Codec changed the requested action horizon")
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite action reconstruction")
    return value


class ActionCodec(Protocol):
    def encode(self, actions: np.ndarray, context: Context) -> Packet: ...
    def decode(self, packet: Packet, context: Context) -> np.ndarray: ...
    def synchronize(self) -> None: ...
    def metadata(self) -> dict[str, Any]: ...
    def close(self) -> None: ...
