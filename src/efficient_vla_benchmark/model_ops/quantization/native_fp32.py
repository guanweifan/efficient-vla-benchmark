"""Explicit packed storage / per-call decode / native FP32 GEMM execution."""

import torch
from torch.nn import functional as F

from .linear import PackedLinear


class PackedFP32Linear(PackedLinear):
    """Decode to a transient workspace to retain native cuBLAS reduction order.

    This is real packed-weight execution, not integer MMA or tile-only decoding.
    Both reconstruction cost and temporary FP32 storage belong in measurements.
    No reconstructed tensor is retained by the module or reused by a later call.
    """

    @torch.no_grad()
    def forward(self, inputs):
        if (
            not inputs.is_cuda
            or inputs.device != self.packed_weight.device
            or inputs.dtype != torch.float32
            or inputs.shape[-1] != self.in_features
        ):
            raise ValueError("PackedFP32Linear requires colocated native FP32 CUDA inputs")
        import triton

        from .native_fp32_kernel import decode_fp32

        x = inputs.contiguous()
        weight = torch.empty(
            (self.out_features, self.in_features), device=x.device, dtype=torch.float32
        )
        decode_fp32[(triton.cdiv(self.out_features, 32), triton.cdiv(self.in_features, 128))](
            self.packed_weight,
            self.weight_scale,
            weight,
            self.out_features,
            self.in_features,
            self.bits,
            self.group_size,
            self.weight_scale.shape[1],
            32,
            128,
            num_warps=4,
        )
        if self.activation_bits < 16 and (
            self.activation_amax is not None or self.activation_step is not None
        ):
            maximum = (1 << (self.activation_bits - 1)) - 1
            step = (
                self.activation_step
                if self.activation_step is not None
                else (self.activation_amax / maximum).clamp_min(1e-8)
            )
            codes = (x / step).round().clamp(self.activation_qmin, maximum).to(torch.int8)
            x = codes.to(torch.float32) * step
        bias = None if self.bias is None else self.bias.to(torch.float32)
        with torch.autocast("cuda", enabled=False):
            output = F.linear(x, weight, bias)
        self.kernel_invocations += 1
        return output

    def execution_metadata(self):
        return {
            **super().execution_metadata(),
            "kernel": "triton_per_call_decode_then_native_fp32_gemm",
            "transient_full_weight_workspace": True,
            "full_dequantized_weight_cache": False,
        }
