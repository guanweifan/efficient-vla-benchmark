import torch
from torch import nn


class PackedCodebookLinear(nn.Module):
    def __init__(self, entry, bias=None):
        super().__init__()
        self.out_features, self.in_features = entry["shape"]
        self.block = entry["block_size"]
        self.bits = entry["index_bits"]
        self.centers = entry["centers"]
        n, k = self.out_features, self.in_features
        if (
            (self.block, self.bits, self.centers) not in ((128, 6, 64), (256, 8, 256))
            or n % self.block
            or k % self.block
        ):
            raise ValueError("Unsupported packed MotionVQ geometry")
        books, indices = entry["codebooks"], entry["indices"]
        dtype = getattr(torch, entry["native_dtype"].split(".")[-1])
        if (
            books.shape != (n * k // self.block**2, self.centers, 2)
            or books.dtype != dtype
            or not torch.isfinite(books).all()
            or indices.dtype != torch.uint8
            or indices.shape != (k // 2, n * self.bits // 8)
        ):
            raise ValueError("Invalid MotionVQ codebook/index artifact")
        self.register_buffer("codebooks", books.contiguous())
        self.register_buffer("indices", indices.contiguous())
        self.register_buffer("bias", None if bias is None else bias.detach().clone())
        self.kernel_invocations = 0

    @torch.no_grad()
    def forward(self, x):
        if (
            not x.is_cuda
            or x.device != self.indices.device
            or x.dtype != self.codebooks.dtype
            or x.shape[-1] != self.in_features
        ):
            raise ValueError("MotionVQ needs colocated native-dtype CUDA inputs")
        import triton

        from .kernel import centroid_gemm

        flat = x.reshape(-1, self.in_features).contiguous()
        y = torch.empty((len(flat), self.out_features), device=x.device, dtype=x.dtype)
        centroid_gemm[(triton.cdiv(len(flat), 32), triton.cdiv(self.out_features, 64))](
            flat,
            self.indices,
            self.codebooks,
            self.bias if self.bias is not None else self.codebooks,
            y,
            len(flat),
            self.out_features,
            self.in_features,
            self.bits,
            self.block,
            self.centers,
            self.bias is not None,
            32,
            64,
            32,
            num_warps=4,
        )
        self.kernel_invocations += 1
        return y.reshape(*x.shape[:-1], self.out_features)


class DualCodebookLinear(nn.Module):
    def __init__(self, entry):
        super().__init__()
        self.high = PackedCodebookLinear(entry["high"], entry["bias"])
        self.low = PackedCodebookLinear(entry["low"], entry["bias"])
        self.use_low = False

    def forward(self, x):
        return self.low(x) if self.use_low else self.high(x)
