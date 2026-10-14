"""Pinned author-code equations, not activation-aware SmoothQuant substitution.

Reference: AIoT-MLSys-Lab/QuantVLA 9e435e5, duquant_preprocess.py.
Energy smoothing affects permutation ranking only; it never rescales activations.
"""

from types import SimpleNamespace

import torch

from .packed import BlockRotation


@torch.no_grad()
def build_transform(weight, config):
    w = weight.detach().float()
    energy = w.square().mean(0)
    energy = (1 - config.energy_smoothing) * energy + config.energy_smoothing * energy.mean()
    if config.enable_permutation:
        order = torch.argsort(energy, descending=True).tolist()
        permutation = []
        left, right = 0, len(order) - 1
        while left <= right:
            permutation.append(order[left])
            left += 1
            if left <= right:
                permutation.append(order[right])
                right -= 1
        permutation = torch.tensor(permutation, device=w.device)
    else:
        permutation = torch.arange(w.shape[1], device=w.device)
    permuted = w.index_select(1, permutation)
    # Reduced SVD returns the same square left factor for these wide blocks,
    # avoiding an unused full right-singular-vector matrix.
    rin = [
        torch.linalg.svd(b.T, full_matrices=False).U
        for b in permuted.split(config.block_size, dim=1)
    ]
    rout = [torch.linalg.svd(b, full_matrices=False).U for b in w.split(config.block_size, dim=0)]
    if any(r.shape[0] != r.shape[1] for r in rin + rout):
        raise ValueError("Rotation blocks require a full-rank geometry")
    return SimpleNamespace(
        smoothing=torch.ones(w.shape[1], device=w.device),
        permutation=permutation,
        input_rotations=rin,
        output_rotations=rout,
        row_restore=True,
    )


@torch.no_grad()
def transformed_weight(weight, transform):
    w = BlockRotation(transform.input_rotations)(weight.index_select(1, transform.permutation))
    return torch.cat(
        [
            r.to(w.dtype) @ b
            for r, b in zip(
                transform.output_rotations, w.split(transform.output_rotations[0].shape[0], dim=0)
            )
        ]
    )


@torch.no_grad()
def quantize_mse(weight):
    """Author five-point per-output-row MSE scale search in native tensor dtype."""
    # Row batches bound temporary storage without changing row reductions/ties.
    codes, scales = [], []
    for w in weight.split(128):
        base = (w.abs().amax(1) / 7).clamp_min(1e-8)
        factors = w.new_tensor([0.5, 0.75, 1.0, 1.25, 1.5])
        candidates = base[:, None] / factors[None, :]
        q = (w[:, None, :] / candidates[:, :, None]).round().clamp(-8, 7)
        mse = ((q * candidates[:, :, None] - w[:, None, :]) ** 2).mean(-1)
        choice = mse.argmin(1)
        rows = torch.arange(w.shape[0], device=w.device)
        codes.append(q[rows, choice].to(torch.int8))
        scales.append(candidates[rows, choice].float())
    return torch.cat(codes), torch.cat(scales)


def attention_scales(teacher, student, config):
    ts, ss = teacher["logits"], student["logits"]
    alpha = torch.where(ss > 0, ts / (ss + 1e-6), torch.ones_like(ts))
    alpha = alpha.clamp(config.alpha_min, config.alpha_max)
    alpha = torch.where((alpha - 1).abs() < config.alpha_neutral, torch.ones_like(alpha), alpha)
    ratio = teacher["output"].clamp_min(1e-8) / student["output"].clamp_min(1e-8)
    log_beta = ratio.log().clamp(-config.beta_log_clip, config.beta_log_clip)
    beta = torch.where(
        log_beta.abs() < config.beta_neutral, torch.ones_like(log_beta), log_beta.exp()
    )
    return {"alpha": alpha.tolist(), "beta": float(beta)}
