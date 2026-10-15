"""True W8A8 / packed W4A8 GEMM with explicitly tokenwise activation quantization.

This optional tensor primitive owns no calibration, smoothing, layer targeting or
paper policy. W4 is expanded to INT8 in registers, NOT executed as INT4 MMA.
"""

import torch
from torch import nn

from .packed import pack_signed


class IntegerLinear(nn.Module):
    def __init__(self, codes, scales, *, bits=8, bias=None, activation_floor=1e-5):
        super().__init__()
        if type(bits) is not int or bits not in (4, 8):
            raise ValueError("IntegerLinear supports W4A8 or W8A8")
        if codes.ndim != 2 or codes.dtype != torch.int8 or min(codes.shape) < 1:
            raise ValueError("codes must be a nonempty int8 matrix")
        self.out_features, self.in_features = codes.shape
        # Worst-case signed INT8 products must fit an INT32 accumulator.
        if self.in_features > (2**31 - 1) // (128 * 127):
            raise ValueError("Input width risks INT32 overflow")
        if scales.shape != (self.out_features,) or not scales.is_floating_point():
            raise ValueError("Weight scales must be per output channel")
        if scales.device != codes.device or not bool(
            torch.isfinite(scales).all() & (scales > 0).all()
        ):
            raise ValueError("Weight scales must be colocated, finite and positive")
        if isinstance(activation_floor, bool) or not 0 < activation_floor < 1:
            raise ValueError("Invalid activation floor")
        if bias is not None and (
            bias.shape != (self.out_features,)
            or bias.device != codes.device
            or not bias.is_floating_point()
            or not bool(torch.isfinite(bias).all())
        ):
            raise ValueError("Invalid bias")
        self.bits = bits
        self.activation_floor = float(activation_floor)
        self.register_buffer("packed_weight", pack_signed(codes, bits).T.contiguous().T)
        self.register_buffer("weight_scale", scales.detach().float().clone().contiguous())
        self.register_buffer("bias", None if bias is None else bias.detach().float().clone())
        self.kernel_invocations = 0

    @torch.no_grad()
    def forward(self, inputs):
        if not inputs.is_cuda or inputs.device != self.packed_weight.device:
            raise ValueError("IntegerLinear needs colocated CUDA inputs; no CPU fallback")
        if inputs.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("Unsupported activation dtype")
        if inputs.ndim < 1 or inputs.shape[-1] != self.in_features:
            raise ValueError("Input width mismatch")
        if self.weight_scale.dtype != torch.float32 or (
            self.bias is not None and self.bias.dtype != torch.float32
        ):
            raise ValueError("IntegerLinear scales/bias must retain FP32 precision")
        import triton

        from .integer_kernel import integer_gemm, quantize_tokens

        flat = inputs.reshape(-1, self.in_features).contiguous()
        rows = flat.shape[0]
        output = torch.empty((rows, self.out_features), device=inputs.device, dtype=inputs.dtype)
        if rows:
            codes = torch.empty_like(flat, dtype=torch.int8)
            scales = torch.empty(rows, device=inputs.device, dtype=torch.float32)
            quantize_tokens[(rows,)](
                flat,
                codes,
                scales,
                rows,
                self.in_features,
                self.activation_floor,
                triton.next_power_of_2(self.in_features),
            )
            bm = 32 if rows < 64 else 64
            bn, bk = 64, 64
            integer_gemm[(triton.cdiv(rows, bm), triton.cdiv(self.out_features, bn))](
                codes,
                self.packed_weight,
                scales,
                self.weight_scale,
                self.bias if self.bias is not None else self.weight_scale,
                output,
                rows,
                self.out_features,
                self.in_features,
                self.bits,
                self.bias is not None,
                bm,
                bn,
                bk,
                num_warps=4,
                num_stages=3,
            )
            self.kernel_invocations += 1
        return output.reshape(*inputs.shape[:-1], self.out_features)

    def execution_metadata(self):
        return {
            "storage": "signed_twos_complement_low_bits_first_uint8",
            "packed_layout": "output_contiguous_transposed_strides_v2",
            "weight_bits": self.bits,
            "weight_scale_axis": "output_channel",
            "activation_bits": 8,
            "activation_storage": "int8",
            "activation_scale_axis": "token_dynamic",
            "activation_range": [-127, 127],
            "activation_amax_floor": self.activation_floor,
            "kernel": "triton_int8_mma_fp32_scaled_epilogue",
            "multiply": "int8",
            "accumulation": "int32",
            "integer_mma": True,
            "int4_mma": False,
            "full_dequantized_weight_cache": False,
        }
