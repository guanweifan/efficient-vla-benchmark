"""Mixed output-channel bit execution; no sensitivity or allocation decisions."""

import torch
from torch import nn
from torch.nn import functional as F

from efficient_vla_benchmark.model_ops.quantization.linear import PackedLinear


class MixedBitLinear(nn.Module):
    @torch.no_grad()
    def __init__(self, original, gates):
        super().__init__()
        weight = original.weight.detach()
        if weight.ndim != 2 or gates.ndim != 1 or gates.numel() != weight.shape[0]:
            raise ValueError("Invalid mixed-bit geometry")
        if gates.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("Gates must be integers")
        if any(b not in (0, 2, 4, 8, 16) for b in gates.tolist()):
            raise ValueError("Unsupported channel precision")
        self.in_features, self.out_features = weight.shape[1], weight.shape[0]
        self.groups = nn.ModuleDict()
        self.active_bits = tuple(b for b in (2, 4, 8, 16) if bool((gates == b).any()))
        for bit in self.active_bits:
            indices = torch.where(gates == bit)[0].to(weight.device)
            self.register_buffer(f"rows_{bit}", indices)
            selected = weight.index_select(0, indices).contiguous()
            bias = None if original.bias is None else original.bias.index_select(0, indices)
            if bit == 16:
                dense = nn.Linear(
                    self.in_features,
                    len(indices),
                    bias=bias is not None,
                    device=weight.device,
                    dtype=weight.dtype,
                )
                dense.weight.copy_(selected)
                if bias is not None:
                    dense.bias.copy_(bias)
                dense.requires_grad_(False)
                self.groups[str(bit)] = dense
            else:
                maximum = (1 << (bit - 1)) - 1
                # inject_fake_w.py quantizes in the checkpoint's native dtype.
                # FP32 is used by sensitivity_hessian_proxy.py, NOT this step.
                scale = selected.abs().amax(1).clamp_min(1e-8) / maximum
                codes = (
                    (selected / scale[:, None]).round().clamp(-maximum - 1, maximum).to(torch.int8)
                )
                self.groups[str(bit)] = PackedLinear(
                    codes, scale, bits=bit, bias=bias, round_weight_scale=True
                )
        self.register_buffer(
            "bias", None if original.bias is None else original.bias.detach().clone()
        )
        self.register_buffer("gates", gates.detach().to(weight.device))
        self.native_dtype = weight.dtype
        self.uniform_group = (
            str(self.active_bits[0])
            if len(self.active_bits) == 1
            and getattr(self, f"rows_{self.active_bits[0]}").numel() == self.out_features
            else None
        )

    @torch.no_grad()
    def forward(self, inputs):
        if not inputs.is_cuda or inputs.device != self.gates.device:
            raise ValueError("MixedBitLinear needs colocated CUDA inputs")
        if self.uniform_group is not None:
            return self.groups[self.uniform_group](inputs)
        output = inputs.new_zeros(*inputs.shape[:-1], self.out_features)
        if self.bias is not None:
            # Zero-bit rows retain bias. Active groups include bias before rounding.
            output.copy_(self.bias.expand_as(output))
        for bit in self.active_bits:
            value = self.groups[str(bit)](inputs)
            output.index_copy_(-1, getattr(self, f"rows_{bit}"), value)
        return output


class MixedBitConv2d(nn.Module):
    """Unfold plus packed per-output-channel GEMM, including bias of pruned rows."""

    @torch.no_grad()
    def __init__(self, original, gates):
        super().__init__()
        if original.groups != 1 or original.padding_mode != "zeros":
            raise ValueError("Only non-grouped zero-padded convolution is supported")
        if isinstance(original.padding, str):
            raise ValueError("Convolution padding must be explicit")
        self.kernel_size = original.kernel_size
        self.stride, self.padding, self.dilation = (
            original.stride,
            original.padding,
            original.dilation,
        )
        flat = nn.Linear(
            original.weight[0].numel(),
            original.out_channels,
            bias=original.bias is not None,
            device=original.weight.device,
            dtype=original.weight.dtype,
        )
        flat.weight.copy_(original.weight.flatten(1))
        if original.bias is not None:
            flat.bias.copy_(original.bias)
        self.linear = MixedBitLinear(flat, gates)

    @torch.no_grad()
    def forward(self, inputs):
        patches = F.unfold(inputs, self.kernel_size, self.dilation, self.padding, self.stride)
        output = self.linear(patches.transpose(1, 2).contiguous()).transpose(1, 2)
        h, w = inputs.shape[-2:]
        shape = [
            (size + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1
            for size, pad, dilation, kernel, stride in zip(
                (h, w), self.padding, self.dilation, self.kernel_size, self.stride
            )
        ]
        return output.reshape(inputs.shape[0], self.linear.out_features, *shape)
