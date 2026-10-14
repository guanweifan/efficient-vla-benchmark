"""Packed execution of calibrated DuQuant transforms, with native output dtype."""

import torch
from torch import nn

from efficient_vla_benchmark.model_ops.quantization.linear import PackedLinear


class BlockRotation(nn.Module):
    def __init__(self, rotations):
        super().__init__()
        if not rotations:
            raise ValueError("Expected at least one rotation")
        self.widths = tuple(r.shape[0] for r in rotations)
        for r in rotations:
            if r.ndim != 2 or r.shape[0] != r.shape[1] or not bool(torch.isfinite(r).all()):
                raise ValueError("Invalid rotation")
        self.uniform = len(set(self.widths)) == 1
        if self.uniform:
            self.register_buffer("rotations", torch.stack(rotations))
        else:
            for i, rotation in enumerate(rotations):
                self.register_buffer(f"rotation_{i}", rotation.detach().clone())

    def forward(self, inputs):
        if inputs.shape[-1] != sum(self.widths):
            raise ValueError("Rotation width mismatch")
        if self.uniform:
            shaped = inputs.reshape(-1, len(self.widths), self.widths[0]).transpose(0, 1)
            out = torch.bmm(shaped, self.rotations.to(inputs.dtype))
            return out.transpose(0, 1).reshape(inputs.shape)
        parts, offset = [], 0
        for i, width in enumerate(self.widths):
            parts.append(
                inputs[..., offset : offset + width]
                @ getattr(self, f"rotation_{i}").to(inputs.dtype)
            )
            offset += width
        return torch.cat(parts, dim=-1)


class PackedDuQuantLinear(nn.Module):
    """A8 stored in INT8; W4 packed; floating tile products and FP32 accumulation.

    This preserves the input-channel scales, not a claimed INT8/INT4 MMA kernel.
    """

    def __init__(
        self, qweight, weight_scale, activation_scale, bias, transform, *, frozen_step=False
    ):
        super().__init__()
        if qweight.ndim != 2 or weight_scale.ndim != 1:
            raise ValueError("Paper path expects per-output-row weight scales")
        n, k = qweight.shape
        if transform.smoothing.shape != (k,) or transform.permutation.shape != (k,):
            raise ValueError("Transform input geometry mismatch")
        permutation = transform.permutation
        if permutation.dtype != torch.int64 or not torch.equal(
            torch.sort(permutation).values.cpu(), torch.arange(k)
        ):
            raise ValueError("Invalid feature permutation")
        if not bool(torch.isfinite(transform.smoothing).all()) or not bool(
            (transform.smoothing > 0).all()
        ):
            raise ValueError("Invalid smoothing")
        self.register_buffer("smoothing", transform.smoothing.detach().clone())
        self.register_buffer("permutation", permutation.detach().clone())
        self.input_rotation = BlockRotation(transform.input_rotations)
        self.output_rotation = (
            BlockRotation(transform.output_rotations) if transform.row_restore else None
        )
        if sum(self.input_rotation.widths) != k or (
            self.output_rotation is not None and sum(self.output_rotation.widths) != n
        ):
            raise ValueError("Rotation geometry mismatch")
        self.linear = PackedLinear(
            qweight,
            weight_scale,
            bits=4,
            activation_amax=None if frozen_step else activation_scale,
            activation_step=activation_scale if frozen_step else None,
            activation_qmin=-128 if frozen_step else -127,
            round_weight_scale=True,
        )
        self.register_buffer("bias", None if bias is None else bias.detach().clone())
        self.in_features, self.out_features = k, n

    def forward(self, inputs):
        x = inputs * self.smoothing.to(inputs.dtype)
        x = x.index_select(-1, self.permutation)
        x = self.input_rotation(x)
        output = self.linear(x)
        if self.output_rotation is not None:
            output = self.output_rotation(output)
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output
