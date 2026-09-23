"""Backend-neutral selection/merge result for visual decoder rows."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TokenReduction:
    """Keep current rows; optionally replace retained visual rows in position order.

    Nonvisual rows are immutable. Backends validate device, dtype and dimensions.
    Methods own grouping and must choose native anchors that respect causality.
    """

    keep: torch.Tensor
    visual_features: torch.Tensor | None = None
