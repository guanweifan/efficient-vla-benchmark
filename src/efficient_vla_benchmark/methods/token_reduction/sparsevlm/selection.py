"""Pinned SparseVLM+ V2.0 selection; no rank estimation or recycling."""

import torch

REFERENCE_REVISION = "9e1f649ef6a78878e2b10d1d6f49ecd7f4d84094"
# The author's profile names denote an effective budget, NOT final token counts.
REFERENCE_BUDGETS = {
    192: (300, 200, 118),
    128: (238, 108, 60),
    96: (246, 54, 28),
    64: (66, 34, 20),
}


def text_raters(visual, text):
    if not len(visual) or not len(text):
        raise ValueError("SparseVLM v2 requires projected visual and prompt tokens")
    # Preserve the author's native-dtype matmul/softmax and strict threshold.
    relevance = (visual @ text.T).softmax(-1).mean(0)
    selected = relevance > relevance.mean()
    if not bool(selected.any()):
        raise ValueError("SparseVLM v2 has no text raters for this degenerate prompt")
    return selected


def stage_budget(original_visual_count, stage, config):
    if not 0 <= stage < 3 or original_visual_count < 2:
        raise ValueError("Invalid SparseVLM stage or initial visual count")
    return max(
        1,
        int(
            REFERENCE_BUDGETS[config.budget_profile][stage]
            * original_visual_count
            / 576
            * config.budget_scale
        ),
    )


def select_visual(corrected_probabilities, visual_mask, budget, priority_heads):
    """Per-head AFTER-softmax probabilities divided by native ones-RoPE gravity.

    Probabilities include all current key columns; head selection and token ranking
    consume the rater-to-visual rectangle, exactly as the pinned author's code.
    """
    if corrected_probabilities.ndim != 3 or not corrected_probabilities.shape[1]:
        raise ValueError("Expected head/rater/key probabilities")
    if not 1 <= priority_heads <= corrected_probabilities.shape[0]:
        raise ValueError("Priority head count exceeds native attention heads")
    relation = corrected_probabilities[:, :, visual_mask]
    if not torch.isfinite(relation).all():
        raise ValueError("Nonfinite SparseVLM v2 gravity-corrected attention")
    heads = relation.sum((1, 2)).topk(priority_heads).indices
    score = relation[heads].mean(0).mean(0)
    keep_count = min(int(budget), len(score) - 1)
    if keep_count < 1:
        raise ValueError("SparseVLM v2 requires at least two visual tokens at a pruning layer")
    ids = score.topk(keep_count).indices.sort().values
    return ids, {
        "kept": keep_count,
        "removed": len(score) - keep_count,
        "priority_heads": priority_heads,
    }
