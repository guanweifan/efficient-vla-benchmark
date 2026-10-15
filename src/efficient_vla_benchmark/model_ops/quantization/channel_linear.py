"""Explicit FP16 packed W4 execution with a native channel activation quantizer."""

import torch
from torch import nn

from .channel_prepare import ChannelPreparation
from .tinychat import TinyChatLinear


class ChannelPackedLinear(nn.Module):
    """Preserves caller codes/scales, but explicitly changes GEMM precision to FP16.

    This is not a bitwise substitute for native FP32 or the generic FP32-accumulating
    packed kernel. A caller changing engines must revalidate whole-policy accuracy.
    Per-channel activation grids remain native-rounded; never reinterpreted as a
    per-token quantizer to obtain integer-MMA speed.
    """

    @torch.no_grad()
    def __init__(
        self,
        codes,
        scales,
        *,
        group_size=None,
        bias=None,
        activation_step=None,
        activation_amax=None,
        activation_bits=8,
        activation_qmin=None,
    ):
        super().__init__()
        if codes.ndim != 2:
            raise ValueError("Expected matrix of W4 codes")
        n, k = codes.shape
        group_size = k if group_size is None else group_size
        if type(group_size) is not int or group_size < 128 or group_size % 128 or k % group_size:
            raise ValueError("Packed group size must divide K and be a multiple of 128")
        if scales.ndim == 1:
            scales = scales[:, None]
        if scales.shape != (n, k // group_size):
            raise ValueError("Weight scale geometry mismatch")
        if type(activation_bits) is not int or activation_bits not in (2, 4, 8, 16):
            raise ValueError("Unsupported activation precision")
        if activation_step is not None and activation_amax is not None:
            raise ValueError("Choose a frozen activation step OR maximum")
        for scale in (activation_step, activation_amax):
            if scale is not None and (
                scale.shape != (k,)
                or scale.device != codes.device
                or not scale.is_floating_point()
                or not bool(torch.isfinite(scale).all() & (scale > 0).all())
            ):
                raise ValueError("Expected finite positive input-channel scales")
        self.in_features, self.out_features = k, n
        self.activation_bits = activation_bits
        maximum = 2 ** (min(activation_bits, 8) - 1) - 1
        self.activation_qmin = -maximum if activation_qmin is None else activation_qmin
        if type(self.activation_qmin) is not int or self.activation_qmin not in (
            -maximum,
            -maximum - 1,
        ):
            raise ValueError("Unsupported signed activation range")
        self.register_buffer(
            "activation_step", None if activation_step is None else activation_step.detach().clone()
        )
        self.register_buffer(
            "activation_amax", None if activation_amax is None else activation_amax.detach().clone()
        )
        self.kernel = TinyChatLinear(codes, scales.repeat_interleave(group_size // 128, 1), bias)
        self.preparation = (
            ChannelPreparation(
                activation_bits, qmin=self.activation_qmin, amax=activation_step is None
            )
            if activation_bits < 16 and (activation_step is not None or activation_amax is not None)
            else None
        )
        self.kernel_invocations = 0

    @torch.no_grad()
    def forward(self, inputs):
        if (
            not inputs.is_cuda
            or inputs.ndim < 2
            or inputs.shape[-1] != self.in_features
            or inputs.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or inputs.device != self.kernel.packed.qweight.device
        ):
            raise ValueError("Expected native floating CUDA input at the packed layer")
        if not inputs.numel():
            return inputs.new_empty((*inputs.shape[:-1], self.out_features))
        if self.preparation is None:
            prepared = inputs.half()
        else:
            scale = (
                self.activation_step if self.activation_step is not None else self.activation_amax
            )
            _, prepared = self.preparation(inputs, scale)
        output = self.kernel(prepared).to(inputs.dtype)
        self.kernel_invocations += 1
        return output

    def execution_metadata(self):
        result = self.kernel.execution_metadata()
        result.update(
            activation_bits=self.activation_bits if self.preparation is not None else 16,
            activation_scale_axis="input_channel" if self.preparation is not None else None,
            activation_qmin=self.activation_qmin,
            activation_preparation="native_rounded_int8_codes_then_fp16_reconstruction",
            boundary_casts="native_quantizer_then_fp16_products_to_native_output",
        )
        return result
