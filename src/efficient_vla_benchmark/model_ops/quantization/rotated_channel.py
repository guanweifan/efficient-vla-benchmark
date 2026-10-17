"""Fuse adjacent block rotations and native-rounded channel quantization.

These are floating preparation/epilogue operators, not integer MMA. Callers own
rotation construction, scale validation and packed GEMM. No paper policy lives
here. Small row counts retain the native CUDA BMM reduction explicitly.
"""

import torch
import triton
import triton.language as tl

from ..block_rotation import BlockRotation
from .channel_prepare import ChannelPreparation


@triton.jit(do_not_specialize=["m"])
def _prepare(
    x,
    rotations,
    permutation,
    scales,
    codes,
    output,
    m,
    k: tl.constexpr,
    width: tl.constexpr,
    permute: tl.constexpr,
    low: tl.constexpr,
    high: tl.constexpr,
    amax: tl.constexpr,
):
    group = tl.program_id(0)
    rows = tl.program_id(1) * 16 + tl.arange(0, 16)
    inner = tl.arange(0, width)
    cols = group * width + inner
    dtype = x.dtype.element_ty
    selected = tl.load(permutation + cols) if permute else cols
    values = tl.load(x + rows[:, None] * k + selected[None, :], rows[:, None] < m, other=0)
    matrix = tl.load(rotations + group * width * width + inner[:, None] * width + inner[None, :])
    rotated = tl.dot(values, matrix, allow_tf32=False).to(dtype).to(tl.float32)
    step = tl.load(scales + cols).to(dtype).to(tl.float32)
    if amax:
        step = (step / high).to(dtype).to(tl.float32)
        step = tl.maximum(step, 1.0e-8).to(dtype).to(tl.float32)
    quotient = (
        tl.inline_asm_elementwise(
            "div.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[rotated, step[None, :]],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        .to(dtype)
        .to(tl.float32)
    )
    bounded = tl.minimum(tl.maximum(quotient, low), high)
    rounded = tl.inline_asm_elementwise(
        "cvt.rni.s32.f32 $0, $1;",
        constraints="=r,f",
        args=[bounded],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    ).to(tl.int8)
    reconstructed = (rounded.to(tl.float32) * step[None, :]).to(dtype)
    offsets = rows[:, None] * k + cols[None, :]
    tl.store(codes + offsets, rounded, rows[:, None] < m)
    tl.store(output + offsets, reconstructed, rows[:, None] < m)


@triton.jit(do_not_specialize=["m"])
def _restore(
    x, rotations, bias, output, m, k: tl.constexpr, width: tl.constexpr, has_bias: tl.constexpr
):
    group = tl.program_id(0)
    rows = tl.program_id(1) * 16 + tl.arange(0, 16)
    inner = tl.arange(0, width)
    cols = group * width + inner
    dtype = output.dtype.element_ty
    values = tl.load(x + rows[:, None] * k + cols[None, :], rows[:, None] < m, other=0).to(dtype)
    matrix = tl.load(rotations + group * width * width + inner[:, None] * width + inner[None, :])
    result = tl.dot(values, matrix, allow_tf32=False).to(dtype)
    if has_bias:
        offset = tl.load(bias + cols).to(dtype)
        # An ordinary '+' can fold bias into the dot accumulator initializer.
        # Preserve the native rounded dot, followed by a separately rounded add.
        result = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[result.to(tl.float32), offset[None, :].to(tl.float32)],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        ).to(dtype)
    tl.store(output + rows[:, None] * k + cols[None, :], result, rows[:, None] < m)


def _validate(inputs, rotations):
    if (
        not inputs.is_cuda
        or inputs.ndim < 2
        or inputs.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or rotations.device != inputs.device
        or rotations.ndim != 3
        or rotations.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or rotations.shape[1] not in (16, 32, 64, 128)
        or rotations.shape[1] != rotations.shape[2]
        or inputs.shape[-1] != rotations.shape[0] * rotations.shape[1]
        or not inputs.shape[-1]
    ):
        raise ValueError("Expected colocated floating CUDA input and uniform rotation blocks")


class RotatedChannelPreparation:
    """Rotate, round in native dtype, quantize channels, then reconstruct FP16.

    Frozen positive scales and rotation values are validated by the caller outside
    inference. Runtime row count and scale contents never enter the dispatch key.
    Optional input-column permutation is gathered inside the rotation kernel.
    Callers validate its bijection outside inference; no smoothing is implied.
    """

    def __init__(self, bits: int, *, qmin: int | None = None, amax: bool = False):
        self.preparation = ChannelPreparation(bits, qmin=qmin, amax=amax)
        self.rotation = BlockRotation()
        self._compiled = {}

    @torch.no_grad()
    def __call__(self, inputs, rotations, scales, permutation=None):
        _validate(inputs, rotations)
        if (
            rotations.dtype != inputs.dtype
            or scales.device != inputs.device
            or scales.shape != (inputs.shape[-1],)
            or scales.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        ):
            raise ValueError("Expected native-dtype rotations and colocated channel scales")
        if permutation is not None and (
            permutation.device != inputs.device
            or permutation.dtype != torch.int64
            or permutation.shape != (inputs.shape[-1],)
        ):
            raise ValueError("Expected a colocated INT64 column permutation")
        inputs, rotations, scales = inputs.contiguous(), rotations.contiguous(), scales.contiguous()
        indices = rotations if permutation is None else permutation.contiguous()
        k, width = inputs.shape[-1], rotations.shape[1]
        m = inputs.numel() // k
        if m <= 16:
            return self.preparation(self.rotation(inputs, rotations, permutation), scales)
        codes = torch.empty_like(inputs, dtype=torch.int8)
        output = torch.empty_like(inputs, dtype=torch.float16)
        grid = (k // width, triton.cdiv(m, 16), 1)
        key = (
            inputs.device.index,
            inputs.dtype,
            scales.dtype,
            permutation is not None,
            k,
            width,
            tuple(t.data_ptr() % 16 for t in (inputs, rotations, indices, scales, codes, output)),
        )
        kernel = self._compiled.get(key)
        with torch.cuda.device(inputs.device):
            if kernel is None:
                kernel = _prepare[grid](
                    inputs,
                    rotations,
                    indices,
                    scales,
                    codes,
                    output,
                    m,
                    k,
                    width,
                    permutation is not None,
                    self.preparation.low,
                    self.preparation.high,
                    self.preparation.amax,
                    num_warps=4,
                )
                self._compiled[key] = kernel
            else:
                kernel[grid](inputs, rotations, indices, scales, codes, output, m)
        return codes, output

    def clear(self):
        self._compiled.clear()
        self.rotation.clear()
        self.preparation.clear()


class RotatedOutput:
    """Cast GEMM output to the rotation dtype, rotate and add native bias.

    Precision is explicit in the supplied rotation dtype. This operation cannot
    make FP16 GEMM equivalent to native FP32 GEMM.
    """

    def __init__(self):
        self.rotation = BlockRotation()
        self._compiled = {}

    @torch.no_grad()
    def __call__(self, inputs, rotations, bias=None):
        _validate(inputs, rotations)
        k, width = inputs.shape[-1], rotations.shape[1]
        if bias is not None and (
            bias.device != inputs.device
            or bias.shape != (k,)
            or bias.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        ):
            raise ValueError("Expected a colocated floating output bias")
        inputs, rotations = inputs.contiguous(), rotations.contiguous()
        offset = rotations if bias is None else bias.contiguous()
        m = inputs.numel() // k
        if m <= 16:
            output = self.rotation(inputs.to(rotations.dtype), rotations)
            return output if bias is None else output + offset.to(output.dtype)
        output = torch.empty_like(inputs, dtype=rotations.dtype)
        grid = (k // width, triton.cdiv(m, 16), 1)
        key = (
            inputs.device.index,
            inputs.dtype,
            rotations.dtype,
            offset.dtype,
            k,
            width,
            bias is not None,
            tuple(t.data_ptr() % 16 for t in (inputs, rotations, offset, output)),
        )
        kernel = self._compiled.get(key)
        with torch.cuda.device(inputs.device):
            if kernel is None:
                kernel = _restore[grid](
                    inputs, rotations, offset, output, m, k, width, bias is not None, num_warps=4
                )
                self._compiled[key] = kernel
            else:
                kernel[grid](inputs, rotations, offset, output, m)
        return output

    def clear(self):
        self._compiled.clear()
        self.rotation.clear()
