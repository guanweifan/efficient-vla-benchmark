"""One W4 bank with explicit INT4/INT8/BF16 execution; no paper-specific dispatch."""

import torch
from torch import nn

from .mixed_runtime import MixedW4Runtime
from .packed import pack_signed


class MixedPrecisionW4Linear(nn.Module):
    """Bias-free BF16 decoder projections with per-output weight scales.

    A2 is physically packed then expanded to INT4 in a separate timed operation;
    it is not a register-fused INT2 kernel. A8 expands W4 in registers for INT8 MMA.
    A16 retains BF16 tile dequantization. The caller owns precision scheduling.
    """

    def __init__(self, codes, scales, *, runtime):
        super().__init__()
        if codes.ndim != 2 or codes.dtype != torch.int8 or min(codes.shape) < 1:
            raise ValueError("Expected a nonempty signed integer matrix")
        self.out_features, self.in_features = codes.shape
        if self.out_features % 8 or self.in_features % 32:
            raise ValueError("INT4 GEMM requires N divisible by 8 and K divisible by 32")
        if self.in_features > (2**31 - 1) // (127 * 8):
            raise ValueError("Possible INT32 accumulator overflow")
        if scales.shape != (self.out_features,) or not scales.is_floating_point():
            raise ValueError("Expected output-channel scales")
        if scales.device != codes.device or not bool(
            torch.isfinite(scales).all() & (scales > 0).all()
        ):
            raise ValueError("Expected colocated positive finite scales")
        rounded = scales.detach().to(torch.bfloat16).float()
        if not bool(torch.isfinite(rounded).all() & (rounded > 0).all()):
            raise ValueError("Scales are not representable in the BF16 weight contract")
        if not isinstance(runtime, MixedW4Runtime):
            raise ValueError("Expected a validated explicit MixedW4Runtime")
        self.register_buffer("packed_weight", pack_signed(codes, 4).contiguous())
        self.register_buffer("weight_scale", rounded.clone().contiguous())
        self.runtime = runtime
        self.active_bits = 16
        self.calls = {2: 0, 4: 0, 8: 0, 16: 0}
        self._checked_devices = set()

    def set_active_bits(self, bits):
        if type(bits) is not int or bits not in self.calls:
            raise ValueError("Expected activation precision in 2/4/8/16")
        self.active_bits = bits

    @torch.no_grad()
    def forward(self, inputs):
        if inputs.ndim < 1 or inputs.shape[-1] != self.in_features:
            raise ValueError("Input width mismatch")
        if not inputs.is_cuda or inputs.device != self.packed_weight.device:
            raise ValueError("Only colocated CUDA execution is supported")
        if inputs.dtype != torch.bfloat16 or self.weight_scale.dtype != torch.float32:
            raise ValueError("BF16 activation and FP32 scale storage required")
        if inputs.device not in self._checked_devices:
            if torch.cuda.get_device_capability(inputs.device)[0] != 8:
                raise ValueError("Mixed W4 runtime currently requires SM8x CUDA hardware")
            self._checked_devices.add(inputs.device)
        import triton

        from .mixed_kernel import expand_two, packed_load, prepare_tokens, scale_integer

        flat = inputs.reshape(-1, self.in_features).contiguous()
        m, k = flat.shape
        n, bits = self.out_features, self.active_bits
        output = torch.empty((m, n), device=inputs.device, dtype=inputs.dtype)
        if not m:
            return output.reshape(*inputs.shape[:-1], n)
        with torch.cuda.device(inputs.device):
            xs, qx = self.weight_scale, flat
            if bits < 16:
                qx = torch.empty((m, k // (8 // bits)), device=inputs.device, dtype=torch.uint8)
                xs = torch.empty(m, device=inputs.device, dtype=torch.float32)
                prepare_tokens[(m,)](flat, qx, xs, m, k, bits, triton.next_power_of_2(k))
            if bits >= 8:
                bm, bk = (128, 64) if bits == 8 or m >= 512 else (64, 128)
                packed_load[(triton.cdiv(m, bm), triton.cdiv(n, 64))](
                    qx.view(torch.int8) if bits == 8 else flat,
                    self.packed_weight,
                    xs,
                    self.weight_scale,
                    output,
                    m,
                    n,
                    k,
                    bits,
                    bm,
                    64,
                    bk,
                    num_warps=4,
                    num_stages=3,
                )
            else:
                expanded = qx
                if bits == 2:
                    expanded = torch.empty((m, k // 2), device=inputs.device, dtype=torch.uint8)
                    expand_two[(triton.cdiv(expanded.numel(), 256),)](
                        qx,
                        expanded,
                        expanded.numel(),
                        256,
                    )
                integer = torch.empty((m, n), device=inputs.device, dtype=torch.int32)
                status = self.runtime.gemm(
                    expanded.data_ptr(),
                    self.packed_weight.data_ptr(),
                    integer.data_ptr(),
                    m,
                    n,
                    k,
                    torch.cuda.current_stream(inputs.device).cuda_stream,
                )
                if status:
                    raise RuntimeError(f"CUTLASS INT4 GEMM failed with status {status}")
                scale_integer[(triton.cdiv(output.numel(), 256),)](
                    integer,
                    xs,
                    self.weight_scale,
                    output,
                    output.numel(),
                    n,
                    256,
                )
        self.calls[bits] += 1
        return output.reshape(*inputs.shape[:-1], n)

    def execution_metadata(self):
        return dict(
            weight_storage="one_row_major_signed_W4_bank",
            weight_scale="per_output_channel_BF16_rounded_FP32_storage",
            activation_quantizer="dynamic_tokenwise_symmetric_nearest_even",
            activation_amax_floor=1e-5,
            activation_storage={
                "2": "packed2_then_packed4",
                "4": "packed4",
                "8": "int8",
                "16": "native_BF16",
            },
            multiply={"2": "int4", "4": "int4", "8": "int8", "16": "bfloat16"},
            accumulation={"2": "int32", "4": "int32", "8": "int32", "16": "float32"},
            full_dequantized_weight_cache=False,
            bias_supported=False,
            runtime=self.runtime.metadata(),
        )
