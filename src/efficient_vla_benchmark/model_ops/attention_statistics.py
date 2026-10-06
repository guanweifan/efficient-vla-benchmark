"""Compact statistics over complete head-mean attention, without policy decisions."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AttentionStatisticsRequest:
    text_start_row: int
    text_end_row: int
    key_start: int
    key_end: int


@dataclass
class AttentionStatistics:
    entropy: torch.Tensor
    text_attention: torch.Tensor


def attention_statistics(head_mean, request):
    """FP32 row entropy plus the requested text/key rectangle.

    Every computed query row and every retained key contribute to entropy.
    The text rectangle is AFTER softmax/head averaging, before the consumer's
    separate text-axis softmax. No layer/head subsampling is permitted.
    """
    if (
        head_mean.ndim != 2
        or not 0 <= request.text_start_row < request.text_end_row <= head_mean.shape[0]
        or not 0 <= request.key_start < request.key_end <= head_mean.shape[1]
    ):
        raise ValueError("Invalid compact attention statistics request")
    probability = head_mean.float()
    probability = probability / (probability.sum(-1, keepdim=True) + 1e-10)
    probability = torch.nan_to_num(probability, nan=0.0)
    entropy = -(probability * (probability + 1e-10).log()).sum(-1).mean()
    return AttentionStatistics(
        entropy,
        head_mean[
            request.text_start_row : request.text_end_row, request.key_start : request.key_end
        ].contiguous(),
    )
