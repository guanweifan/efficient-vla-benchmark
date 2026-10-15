"""Fused native-rounded channel quantization and FP16 packed-kernel preparation.

This is not an integer GEMM. Callers own positive frozen scales, calibration and
the downstream execution precision. Integer codes and their FP16 reconstruction
are returned explicitly; the native quantizer's intermediate rounding is retained.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["size"])
def _prepare(
    x,
    scales,
    codes,
    output,
    size,
    k: tl.constexpr,
    low: tl.constexpr,
    high: tl.constexpr,
    amax: tl.constexpr,
    block: tl.constexpr,
):
    index = tl.program_id(0) * block + tl.arange(0, block)
    dtype = x.dtype.element_ty
    value = tl.load(x + index, index < size, other=0).to(tl.float32)
    step = tl.load(scales + index % k).to(dtype).to(tl.float32)
    if amax:
        step = (step / high).to(dtype).to(tl.float32)
        step = tl.maximum(step, 1.0e-8).to(dtype).to(tl.float32)
    quotient = (
        tl.inline_asm_elementwise(
            "div.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[value, step],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        .to(dtype)
        .to(tl.float32)
    )
    quotient = tl.minimum(tl.maximum(quotient, low), high)
    rounded = tl.inline_asm_elementwise(
        "cvt.rni.s32.f32 $0, $1;",
        constraints="=r,f",
        args=[quotient],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    ).to(tl.int8)
    reconstructed = (rounded.to(tl.float32) * step).to(dtype)
    tl.store(codes + index, rounded, index < size)
    tl.store(output + index, reconstructed, index < size)


class ChannelPreparation:
    """Instance-local compiled kernels; no tensor values or model state cached.

    Scales are caller-validated finite positive vectors. Values may change between
    calls (for example a denoising-step table); no scale value is compiled/cached.
    Native dtype conversion of a supplied scale must remain finite and positive.
    """

    def __init__(self, bits: int, *, qmin: int | None = None, amax: bool = False):
        if type(bits) is not int or bits not in (2, 4, 8) or type(amax) is not bool:
            raise ValueError("Expected A2/A4/A8 and an explicit scale convention")
        self.high = 2 ** (bits - 1) - 1
        self.low = -self.high if qmin is None else qmin
        if type(self.low) is not int or self.low not in (-self.high, -self.high - 1):
            raise ValueError("Unsupported signed range")
        self.amax = amax
        self._compiled = {}

    @torch.no_grad()
    def __call__(self, inputs, scales):
        if (
            not inputs.is_cuda
            or inputs.ndim < 2
            or inputs.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or scales.device != inputs.device
            or scales.shape != (inputs.shape[-1],)
            or scales.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or not inputs.shape[-1]
        ):
            raise ValueError("Expected colocated floating CUDA input and channel scales")
        inputs, scales = inputs.contiguous(), scales.contiguous()
        codes = torch.empty_like(inputs, dtype=torch.int8)
        output = torch.empty_like(inputs, dtype=torch.float16)
        if not inputs.numel():
            return codes, output
        grid = (triton.cdiv(inputs.numel(), 1024), 1, 1)
        key = (
            inputs.device.index,
            inputs.dtype,
            scales.dtype,
            inputs.shape[-1],
            tuple(t.data_ptr() % 16 for t in (inputs, scales, codes, output)),
        )
        kernel = self._compiled.get(key)
        with torch.cuda.device(inputs.device):
            if kernel is None:
                kernel = _prepare[grid](
                    inputs,
                    scales,
                    codes,
                    output,
                    inputs.numel(),
                    inputs.shape[-1],
                    self.low,
                    self.high,
                    self.amax,
                    1024,
                )
                self._compiled[key] = kernel
            else:
                kernel[grid](inputs, scales, codes, output, inputs.numel())
        return codes, output

    def clear(self):
        self._compiled.clear()
