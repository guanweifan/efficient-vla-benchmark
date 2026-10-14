"""Irreversible packed linear storage; calibration belongs to the calling method."""

import torch
from torch import nn

from .packed import pack_signed


class PackedLinear(nn.Module):
    def __init__(
        self,
        codes,
        scales,
        *,
        bits,
        group_size=None,
        bias=None,
        activation_amax=None,
        activation_step=None,
        activation_qmin=-127,
        round_weight_scale=False,
    ):
        super().__init__()
        self.out_features, self.in_features = codes.shape
        self.bits = bits
        self.group_size = self.in_features if group_size is None else group_size
        if type(self.group_size) is not int or self.group_size < 1:
            raise ValueError("group_size must be a positive integer")
        groups = (self.in_features + self.group_size - 1) // self.group_size
        if scales.ndim == 1:
            scales = scales[:, None]
        if scales.shape != (self.out_features, groups):
            raise ValueError("Weight scale geometry mismatch")
        if not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all()):
            raise ValueError("Weight scales must be finite and positive")
        self.register_buffer("packed_weight", pack_signed(codes, bits))
        self.register_buffer("weight_scale", scales.detach().float().contiguous())
        self.register_buffer("activation_amax", None)
        self.register_buffer("activation_step", None)
        if activation_amax is not None and activation_step is not None:
            raise ValueError("Supply an activation maximum OR a frozen step, not both")
        if type(activation_qmin) is not int or activation_qmin not in (-128, -127):
            raise ValueError("Unsupported signed A8 range")
        self.activation_qmin = activation_qmin
        if activation_step is not None:
            if activation_step.shape != (self.in_features,) or not bool(
                torch.isfinite(activation_step).all() & (activation_step > 0).all()
            ):
                raise ValueError("Activation step must be finite positive input-channel scales")
            self.activation_step = activation_step.detach().float().contiguous()
        if activation_amax is not None:
            if activation_amax.shape != (self.in_features,):
                raise ValueError("Activation scale must be per input channel")
            if not bool(torch.isfinite(activation_amax).all()) or not bool(
                (activation_amax > 0).all()
            ):
                raise ValueError("Activation amax must be finite and positive")
            self.activation_amax = activation_amax.detach().float().contiguous()
        if bias is not None and (
            bias.shape != (self.out_features,) or not bool(torch.isfinite(bias).all())
        ):
            raise ValueError("Invalid bias")
        self.register_buffer("bias", None if bias is None else bias.detach().clone())
        self.round_weight_scale = bool(round_weight_scale)
        self.kernel_invocations = 0

    @torch.no_grad()
    def forward(self, inputs):
        if not inputs.is_cuda or inputs.device != self.packed_weight.device:
            raise ValueError("PackedLinear needs colocated CUDA inputs; no CPU fallback")
        if inputs.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("Unsupported activation precision")
        if inputs.shape[-1] != self.in_features:
            raise ValueError("Input width mismatch")
        import triton

        from .kernel import packed_gemm

        shape = inputs.shape[:-1]
        flat = inputs.reshape(-1, self.in_features).contiguous()
        quant_a = self.activation_amax is not None or self.activation_step is not None
        step = self.weight_scale  # valid unused pointer for weight-only specialization
        if quant_a:
            step = (
                self.activation_step.to(inputs.dtype)
                if self.activation_step is not None
                else (self.activation_amax.to(inputs.dtype) / 127.0).clamp_min(1e-8)
            )
            flat = (flat / step).round().clamp(self.activation_qmin, 127).to(torch.int8)
        output = torch.empty(
            (flat.shape[0], self.out_features), device=inputs.device, dtype=inputs.dtype
        )
        if flat.shape[0]:
            # Keep the K reduction unchanged; wider M/N tiles reuse decoded weights.
            wide = (
                inputs.dtype in (torch.float16, torch.bfloat16)
                and flat.shape[0] >= 32
                and self.out_features >= 64
            )
            block_m = (64 if flat.shape[0] >= 64 else 32) if wide else 16
            block_n = 64 if wide else 32
            grid = (
                triton.cdiv(flat.shape[0], block_m),
                triton.cdiv(self.out_features, block_n),
            )
            packed_gemm[grid](
                flat,
                self.packed_weight,
                self.weight_scale,
                step,
                self.bias if self.bias is not None else self.weight_scale,
                output,
                flat.shape[0],
                self.out_features,
                self.in_features,
                self.bits,
                self.group_size,
                self.weight_scale.shape[1],
                quant_a,
                self.round_weight_scale,
                self.bias is not None,
                block_m,
                block_n,
                32,
                num_warps=4,
            )
            self.kernel_invocations += 1
        return output.reshape(*shape, self.out_features)

    def execution_metadata(self):
        return {
            "storage": "signed_twos_complement_low_bits_first_uint8",
            "weight_bits": self.bits,
            "activation_bits": 8
            if self.activation_amax is not None or self.activation_step is not None
            else None,
            "activation_scale_axis": "input_channel"
            if self.activation_amax is not None or self.activation_step is not None
            else None,
            "activation_qmin": self.activation_qmin,
            "frozen_activation_step": self.activation_step is not None,
            "kernel": "triton_tile_dequant_float_gemm",
            "accumulation": "float32",
            "integer_mma": False,
            "full_dequantized_weight_cache": False,
            "round_weight_scale_to_activation_dtype": self.round_weight_scale,
        }
