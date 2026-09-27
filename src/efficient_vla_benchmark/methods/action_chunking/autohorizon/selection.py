"""Adapted AutoHorizon bidirectional soft pointer, Apache-2.0.

Source: hatchetProject/AutoHorizon, c7504f1756109103f2cfcc2e23f1b1a23841c885,
src/openpi/models_pytorch/pi0_pytorch.py. See root NOTICE for terms.
Changes: isolated selector, explicit shape validation and compact diagnostics;
boolean run detection replaces the equivalent float convolution.
"""

import math

import torch


def _pointer(matrix, config):
    probabilities = matrix / (matrix.sum(-1, keepdim=True) + 1e-12)
    length = len(matrix)
    keys = torch.arange(length, device=matrix.device, dtype=matrix.dtype)
    pointer = (probabilities * keys).sum(-1).cummax(dim=0).values
    safe = probabilities.clamp_min(1e-12)
    entropy = -(safe * safe.log()).sum(-1) / math.log(length)
    reliable = entropy <= torch.quantile(entropy.float(), min(config.entropy_quantile, 0.999))
    advances = torch.diff(pointer, prepend=pointer.new_zeros(1))
    holds = (advances < config.hold_threshold) & reliable
    runs = holds.unfold(0, config.run_length, 1).all(-1)
    positions = torch.nonzero(runs, as_tuple=False).flatten()
    stop = int(positions[0]) if len(positions) else length - 1
    return pointer, stop


def select_prefix(attention, config):
    if (
        attention.ndim != 2
        or attention.shape[0] != attention.shape[1]
        or len(attention) < 2
        or config.run_length > len(attention)
        or not torch.isfinite(attention).all()
        or (attention < 0).any()
        or not (attention.sum(-1) > 0).all()
    ):
        raise ValueError("AutoHorizon requires a finite positive-mass square action attention")
    length = len(attention)
    probabilities = attention / (attention.sum(-1, keepdim=True) + 1e-12)
    forward, stop_f = _pointer(probabilities, config)
    reverse, stop_r = _pointer(probabilities.flip((0, 1)), config)
    backward = (length - 1 - reverse).flip((0,))
    n_forward = int((forward[stop_f].floor() + 1).clamp(1, length))
    # Preserve the released code's backward-index mapping, not a new suffix rule.
    n_backward = int((backward[length - 1 - stop_r].floor() + 1).clamp(1, length))
    joined = bool(((backward - forward) <= 1).any())
    horizon = length if n_forward + n_backward >= length and joined else n_forward
    return min(config.max_horizon, horizon), dict(
        autohorizon_forward=n_forward,
        autohorizon_backward=n_backward,
        autohorizon_joined=joined,
    )
