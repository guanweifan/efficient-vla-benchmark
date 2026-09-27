"""Tensor-only independent candidate expansion with native CFG branch pairing."""

import torch


def expand_candidates(noise, condition, count, using_cfg):
    """Keep the native first draw and its RNG convention, then add real samples."""
    rows = 2 if using_cfg else 1
    if (
        type(count) is not int
        or count < 1
        or noise.ndim != 3
        or condition.ndim != 3
        or noise.shape[0] != rows
        or condition.shape[0] != rows
    ):
        raise ValueError("Expected one native observation and a positive candidate count")
    if using_cfg and not torch.equal(noise[0], noise[1]):
        raise ValueError("Native CFG noise branches must be paired")
    extra = torch.randn((count - 1, *noise.shape[1:]), device=noise.device).to(noise.dtype)
    expanded_noise = torch.cat((noise[:1], extra), dim=0)
    expanded_condition = condition[:1].expand(count, *condition.shape[1:])
    if using_cfg:
        expanded_noise = torch.cat((expanded_noise, expanded_noise), dim=0)
        expanded_condition = torch.cat(
            (expanded_condition, condition[1:2].expand(count, *condition.shape[1:])), dim=0
        )
    return expanded_noise, expanded_condition
