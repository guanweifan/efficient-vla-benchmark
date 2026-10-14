"""Factorized SVD-Hadamard/zigzag transforms; no dense K-by-K runtime matrix."""

import torch
from torch import nn


def hadamard(size, device):
    if size < 1 or size & (size - 1):
        raise ValueError("Hadamard requires a power of two")
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < size:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / size**0.5


@torch.no_grad()
def build(weight, block_size, row_restore):
    w = weight.float()
    order = w.square().mean(0).argsort(descending=True)
    permutation = torch.stack(
        (order[: len(order) // 2], order.flip(0)[: len(order) // 2]), 1
    ).flatten()
    if len(order) % 2:
        permutation = torch.cat((permutation, order[len(order) // 2 : len(order) // 2 + 1]))
    if w.shape[1] % block_size or (row_restore and w.shape[0] % block_size):
        raise ValueError("This binding requires divisible rotation blocks")
    wp = w[:, permutation]
    h = hadamard(block_size, w.device)
    rin = []
    for offset in range(0, w.shape[1], block_size):
        # Reduced SVD has the same complete left basis for a short-wide matrix.
        u, _, _ = torch.linalg.svd(wp[:, offset : offset + block_size].T, full_matrices=False)
        rin.append(u @ h)
    rout = []
    if row_restore:
        for offset in range(0, w.shape[0], block_size):
            u, _, _ = torch.linalg.svd(w[offset : offset + block_size], full_matrices=False)
            rout.append(u)
    return dict(
        permutation=permutation.cpu(),
        input=torch.stack(rin).cpu(),
        output=torch.stack(rout).cpu() if rout else None,
    )


def rotate(value, blocks):
    width = blocks.shape[1]
    if value.shape[-1] != blocks.shape[0] * width:
        raise ValueError("Rotation geometry mismatch")
    shape = value.shape
    grouped = value.reshape(-1, len(blocks), width).transpose(0, 1)
    return torch.bmm(grouped, blocks.to(value)).transpose(0, 1).reshape(shape)


@torch.no_grad()
def transformed_weight(weight, transform):
    w = rotate(
        weight.float()[:, transform["permutation"].to(weight.device)],
        transform["input"].to(weight.device),
    )
    if transform["output"] is not None:
        # W' = R_out W R_in; inference restores y' @ R_out.
        b = transform["output"].shape[1]
        blocks = transform["output"].to(w)
        w = torch.bmm(blocks, w.reshape(len(blocks), b, -1)).reshape_as(w)
    return w


class InputRotation(nn.Module):
    def __init__(self, transform, dtype):
        super().__init__()
        self.register_buffer("permutation", transform["permutation"].clone())
        self.register_buffer("blocks", transform["input"].to(dtype))

    def forward(self, value):
        return rotate(value.index_select(-1, self.permutation), self.blocks)


def validate_transform(transform, in_features, out_features, row_restore):
    if not isinstance(transform, dict) or set(transform) != {"permutation", "input", "output"}:
        raise ValueError("Invalid composite transform fields")
    p = transform["permutation"]
    if (
        p.dtype != torch.int64
        or p.shape != (in_features,)
        or not torch.equal(p.sort().values.cpu(), torch.arange(in_features))
    ):
        raise ValueError("Composite permutation must be a bijection")
    for key, size in (("input", in_features), ("output", out_features)):
        blocks = transform[key]
        if key == "output" and not row_restore:
            if blocks is not None:
                raise ValueError("Unexpected language output rotation")
            continue
        if (
            blocks is None
            or blocks.ndim != 3
            or blocks.shape[1] != blocks.shape[2]
            or len(blocks) * blocks.shape[1] != size
        ):
            raise ValueError("Composite rotation geometry mismatch")
        if not bool(torch.isfinite(blocks).all()):
            raise ValueError("Nonfinite composite rotation")
        gram = blocks.float().transpose(1, 2) @ blocks.float()
        eye = torch.eye(blocks.shape[1], device=blocks.device).expand_as(gram)
        if not torch.allclose(gram, eye, atol=2e-4, rtol=2e-4):
            raise ValueError("Composite transform is not orthogonal")
