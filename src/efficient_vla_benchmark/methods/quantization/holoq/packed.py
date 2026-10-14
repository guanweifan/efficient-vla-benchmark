"""Real packed W4/A4 execution of the explicitly scoped official-code variant."""

import torch
from torch import nn

from efficient_vla_benchmark.model_ops.quantization.linear import PackedLinear
from efficient_vla_benchmark.model_ops.quantization.native_fp32 import PackedFP32Linear

from .rotation import InputRotation, rotate


class HoloQLinear(nn.Module):
    def __init__(self, entry, step_provider):
        super().__init__()
        dtype = getattr(torch, entry["native_dtype"].split(".")[-1])
        self.input_rotation = InputRotation(entry["transform"], dtype)
        self.register_buffer(
            "output_rotation",
            None
            if entry["transform"]["output"] is None
            else entry["transform"]["output"].to(dtype),
        )
        self.register_buffer("scale_table", entry["activation_steps"].to(dtype))
        self.register_buffer("bias", None if entry["bias"] is None else entry["bias"].to(dtype))
        self.step_provider = step_provider
        self.is_dit = entry["side"] != "language"
        linear_type = PackedFP32Linear if self.is_dit and dtype == torch.float32 else PackedLinear
        self.linear = linear_type(
            entry["codes"],
            entry["weight_scales"],
            bits=4,
            group_size=entry["weight_group_size"],
            activation_bits=4,
            activation_step=self.scale_table[0],
            activation_qmin=-8,
            round_weight_scale=False,
        )
        self.in_features, self.out_features = self.linear.in_features, self.linear.out_features

    def forward(self, inputs):
        x = self.input_rotation(inputs)
        if self.is_dit:
            row = self.step_provider()
            if row is None or not 0 <= row < len(self.scale_table):
                raise ValueError("Missing or invalid native denoising timestep")
            self.linear.activation_step = self.scale_table[row]
        output = self.linear(x)
        if self.output_rotation is not None:
            output = rotate(output, self.output_rotation)
        return output if self.bias is None else output + self.bias
