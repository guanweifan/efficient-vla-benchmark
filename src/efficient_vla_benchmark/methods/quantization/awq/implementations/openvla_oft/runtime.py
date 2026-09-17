"""Explicit execution precision around the unmodified author's packed kernels."""

import torch
from torch import nn


class KernelPrecisionLinear(nn.Module):
    """Keep packed INT4 weights; expose any kernel-boundary casts in metadata."""

    def __init__(self, packed, dtype):
        super().__init__()
        self.quantized = packed.to(dtype=dtype)
        self.in_features = packed.in_features
        self.out_features = packed.out_features
        for name in ("scales", "scaled_zeros", "bias"):
            value = getattr(packed, name)
            if value is not None and not bool(torch.isfinite(value).all()):
                raise ValueError("AWQ buffers overflow the selected kernel precision")

    @torch.no_grad()
    def forward(self, inputs):
        output = self.quantized(inputs.to(dtype=self.quantized.scales.dtype))
        return output.to(dtype=inputs.dtype)
