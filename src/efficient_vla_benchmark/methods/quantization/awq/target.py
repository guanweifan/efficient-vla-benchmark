"""Explicit decoder boundary; backends keep their own component layout."""

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class DecoderTarget:
    backend_id: str
    layers: object
    device: torch.device
    precision: str
    checkpoint_sha256: str
    causal: bool
    offload: Callable[[], None]

    @property
    def dtype(self):
        return getattr(torch, self.precision)
