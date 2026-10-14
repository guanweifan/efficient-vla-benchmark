"""Explicit signed, low-bits-first row packing, with no floating weight cache."""

import torch


def pack_signed(codes, bits):
    if type(bits) is not int or bits not in (2, 4, 8):
        raise ValueError("Supported physical weight widths are 2, 4 and 8")
    if codes.ndim != 2 or codes.dtype != torch.int8 or min(codes.shape) < 1:
        raise ValueError("codes must be a nonempty int8 matrix")
    bound = 1 << (bits - 1)
    # 128 cannot be represented by an int8 comparison scalar. An int8 tensor
    # already spans exactly the valid signed eight-bit domain.
    if bits < 8 and bool(((codes < -bound) | (codes >= bound)).any()):
        raise ValueError("Signed codes are outside the declared range")
    per_byte = 8 // bits
    n, k = codes.shape
    padded = torch.nn.functional.pad(codes, (0, (-k) % per_byte))
    values = (padded.to(torch.int16) & ((1 << bits) - 1)).reshape(n, -1, per_byte)
    packed = torch.zeros(values.shape[:2], device=codes.device, dtype=torch.int16)
    for offset in range(per_byte):
        packed |= values[:, :, offset] << (bits * offset)
    return packed.to(torch.uint8).contiguous()


def unpack_signed(packed, bits, width):
    """Diagnostic oracle only; production kernels unpack individual tiles."""
    if bits not in (2, 4, 8) or type(width) is not int or width < 1:
        raise ValueError("Invalid packed format")
    if packed.ndim != 2 or packed.dtype != torch.uint8:
        raise ValueError("Expected uint8 rows")
    per_byte = 8 // bits
    if packed.shape[1] != (width + per_byte - 1) // per_byte:
        raise ValueError("Packed row width mismatch")
    values = []
    bound = 1 << (bits - 1)
    for offset in range(per_byte):
        code = (packed.to(torch.int16) >> (bits * offset)) & ((1 << bits) - 1)
        values.append(torch.where(code >= bound, code - (1 << bits), code))
    return torch.stack(values, -1).reshape(packed.shape[0], -1)[:, :width].to(torch.int8)
