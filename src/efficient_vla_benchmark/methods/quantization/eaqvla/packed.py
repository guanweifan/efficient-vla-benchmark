"""Vision patch convolution via packed GEMM; no floating weight cache."""

from torch import nn
from torch.nn import functional as F

from efficient_vla_benchmark.model_ops.quantization.linear import PackedLinear


class PackedPatchConv(nn.Module):
    def __init__(self, native, entry, bits):
        super().__init__()
        self.kernel_size, self.dilation, self.padding, self.stride = (
            native.kernel_size,
            native.dilation,
            native.padding,
            native.stride,
        )
        self.linear = PackedLinear(entry["codes"], entry["scales"], bits=bits, bias=entry["bias"])

    def forward(self, x):
        patches = (
            F.unfold(x, self.kernel_size, self.dilation, self.padding, self.stride)
            .transpose(1, 2)
            .contiguous()
        )
        y = self.linear(patches).transpose(1, 2)
        shape = [
            (size + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1
            for size, pad, dilation, kernel, stride in zip(
                x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
            )
        ]
        return y.reshape(x.shape[0], self.linear.out_features, *shape)
