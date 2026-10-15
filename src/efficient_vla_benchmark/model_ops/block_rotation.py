"""Fused per-block matrix rotation with optional column permutation.

Owns no rotation construction, quantizer, calibration or paper-specific rule.
Runtime row count does not specialize the compiled kernel.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["m"])
def _rotate(
    x,
    rotations,
    permutation,
    output,
    m,
    k: tl.constexpr,
    width: tl.constexpr,
    permute: tl.constexpr,
    block_m: tl.constexpr,
):
    group = tl.program_id(0)
    rows = tl.program_id(1) * block_m + tl.arange(0, block_m)
    inner = tl.arange(0, width)
    cols = group * width + inner
    if permute:
        cols = tl.load(permutation + cols)
    values = tl.load(x + rows[:, None] * k + cols[None, :], rows[:, None] < m, other=0)
    matrix = tl.load(rotations + group * width * width + inner[:, None] * width + inner[None, :])
    rotated = tl.dot(values, matrix, allow_tf32=False)
    tl.store(
        output + rows[:, None] * k + group * width + inner[None, :], rotated, rows[:, None] < m
    )


class BlockRotation:
    """Instance-local dispatch cache; callers validate rotation/permutation values.

    Blocks must have input dtype; no hidden precision conversion or CPU fallback.
    A permutation, when supplied, must be a valid bijection over input columns.
    Construction/value validation belongs outside the timed inference call.
    Fewer than sixteen rows use native CUDA BMM: its vector-sized reduction can
    differ from tiled dot even in FP32. This explicit dispatch preserves that path.
    """

    def __init__(self):
        self._compiled = {}

    @torch.no_grad()
    def __call__(self, inputs, rotations, permutation=None):
        if (
            not inputs.is_cuda
            or inputs.ndim < 2
            or rotations.ndim != 3
            or inputs.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or rotations.device != inputs.device
            or rotations.dtype != inputs.dtype
            or rotations.shape[1] not in (16, 32, 64, 128)
            or rotations.shape[1] != rotations.shape[2]
            or inputs.shape[-1] != rotations.shape[0] * rotations.shape[1]
            or not inputs.shape[-1]
        ):
            raise ValueError("Expected colocated CUDA input and uniform rotation blocks")
        k, width = inputs.shape[-1], rotations.shape[1]
        if permutation is not None and (
            permutation.device != inputs.device
            or permutation.dtype != torch.int64
            or permutation.shape != (k,)
        ):
            raise ValueError("Expected a colocated INT64 column permutation")
        inputs, rotations = inputs.contiguous(), rotations.contiguous()
        indices = rotations if permutation is None else permutation.contiguous()
        output = torch.empty_like(inputs)
        m = inputs.numel() // k
        if not m:
            return output
        if m < 16:
            selected = inputs if permutation is None else inputs.index_select(-1, indices)
            grouped = selected.reshape(-1, len(rotations), width).transpose(0, 1)
            return torch.bmm(grouped, rotations).transpose(0, 1).reshape_as(inputs)
        grid = (k // width, triton.cdiv(m, 16), 1)
        key = (
            inputs.device.index,
            inputs.dtype,
            k,
            width,
            permutation is not None,
            tuple(t.data_ptr() % 16 for t in (inputs, rotations, indices, output)),
        )
        kernel = self._compiled.get(key)
        with torch.cuda.device(inputs.device):
            if kernel is None:
                kernel = _rotate[grid](
                    inputs,
                    rotations,
                    indices,
                    output,
                    m,
                    k,
                    width,
                    permutation is not None,
                    16,
                    num_warps=4,
                )
                self._compiled[key] = kernel
            else:
                kernel[grid](inputs, rotations, indices, output, m)
        return output

    def clear(self):
        self._compiled.clear()
