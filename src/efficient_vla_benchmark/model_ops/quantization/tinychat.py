"""Pinned TinyChat W4 execution for caller-supplied codes; no AWQ calibration."""

import importlib.util
import sys

import torch
from torch import nn

from efficient_vla_benchmark.core.resources import project_source


def reference_module():
    name = "_evb_tinychat_packed_linear"
    if name not in sys.modules:
        path = project_source("third_party/runtimes/llm_awq") / "awq/quantize/qmodule.py"
        if not path.is_file():
            raise RuntimeError("Initialize the pinned llm-awq runtime submodule")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules[name]


class TinyChatLinear(nn.Module):
    """Symmetric W4g128, FP16 kernel with explicit BF16 boundary casts.

    Codes and scales are already selected by the caller's quantizer. This class
    performs packing/execution only. The author kernel uses floating products,
    FP16 accumulation and tile dequantization, not integer MMA or a dense cache.
    """

    @torch.no_grad()
    def __init__(self, codes, scales, bias=None):
        super().__init__()
        if codes.ndim != 2 or codes.dtype != torch.int8:
            raise ValueError("TinyChat requires a signed INT8 matrix of W4 codes")
        n, k = codes.shape
        if not n or not k or n % 128 or k % 128:
            raise ValueError("Validated TinyChat geometry requires multiples of 128")
        if scales.shape != (n, k // 128) or not scales.is_floating_point():
            raise ValueError("TinyChat requires per-output, group-128 scales")
        if scales.device != codes.device or not bool((codes >= -8).all() & (codes <= 7).all()):
            raise ValueError("Invalid TinyChat codes/device")
        half = scales.to(torch.float16)
        if not bool(torch.isfinite(half).all() & (half > 0).all()):
            raise ValueError("TinyChat scale over/underflow in FP16")
        if not bool(torch.isfinite(-8 * half).all()):
            raise ValueError("TinyChat zero-point offset overflows FP16")
        if bias is not None and (
            bias.shape != (n,)
            or bias.device != codes.device
            or not bool(torch.isfinite(bias.to(torch.float16)).all())
        ):
            raise ValueError("Invalid TinyChat bias")
        author = reference_module()
        self.packed = author.WQLinear(4, 128, k, n, bias is not None, codes.device)
        self.packed.qweight = author.pack_intweight(codes.to(torch.int32) + 8, 4, 64)
        self.packed.scales[: k // 128].copy_(half.T)
        self.packed.scaled_zeros[: k // 128].copy_(-8 * half.T)
        if bias is not None:
            self.packed.bias.copy_(bias)
        self.in_features, self.out_features = k, n

    @torch.no_grad()
    def forward(self, inputs):
        if (
            not inputs.is_cuda
            or inputs.device != self.packed.qweight.device
            or inputs.dtype not in (torch.float16, torch.bfloat16)
            or inputs.ndim < 2
            or inputs.shape[-1] != self.in_features
        ):
            raise ValueError("TinyChat requires colocated CUDA FP16/BF16 activations")
        if inputs.numel() == 0:
            return inputs.new_empty((*inputs.shape[:-1], self.out_features))
        # Conversion and contiguity costs remain inside the measured forward.
        prepared = inputs.to(torch.float16).contiguous()
        rows = prepared.numel() // self.in_features
        if 8 <= rows <= 32:
            # The pinned short-M GEMM allocates split-K semaphores with empty(),
            # allowing stale allocator contents to release a tile prematurely.
            # Use its single-split kernel via one independent padded row range.
            # GEMM never mixes rows; include this allocation/copy in timing.
            padded = prepared.new_zeros((33, self.in_features))
            padded[:rows].copy_(prepared.reshape(rows, self.in_features))
            output = self.packed(padded)[:rows].reshape(*inputs.shape[:-1], self.out_features)
        else:
            output = self.packed(prepared)
        return output.to(inputs.dtype)

    def execution_metadata(self):
        return dict(
            engine="pinned_tinychat",
            weight_bits=4,
            group_size=128,
            weight_layout="int16_interleave4_kstride64",
            activation_bits=16,
            kernel_dtype="float16",
            accumulation="float16",
            integer_mma=False,
            boundary_casts="input_to_fp16_output_to_native",
            full_dequantized_weight_cache=False,
            short_row_dispatch="M8_to32_pad_to33_single_split_then_slice",
        )
