"""Explicit FP16 packed execution with native activation rounding and bias."""

import torch
from torch import nn

from efficient_vla_benchmark.model_ops.quantization.channel_linear import ChannelPackedLinear


class DAMPALinear(nn.Module):
    def __init__(self, codes, scales, activation_amax, bias):
        super().__init__()
        self.linear = ChannelPackedLinear(codes, scales, activation_amax=activation_amax)
        self.register_buffer("bias", None if bias is None else bias.detach().clone())
        self.in_features, self.out_features = codes.shape[1], codes.shape[0]
        if bias is not None and (
            bias.shape != (self.out_features,) or not bool(torch.isfinite(bias).all())
        ):
            raise ValueError("Invalid native bias")

    def forward(self, inputs):
        output = self.linear(inputs)
        return output if self.bias is None else output + self.bias.to(output.dtype)
