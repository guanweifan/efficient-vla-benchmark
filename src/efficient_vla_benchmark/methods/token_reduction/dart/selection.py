"""Key-norm pivots followed by sequential least-similar visual selection."""

import torch
from torch.nn import functional as F


def pivot_selection(hidden, keys, visual, text, ratio, visual_pivots, text_pivots):
    if hidden.ndim != 2 or keys.ndim != 2 or len(hidden) != len(keys):
        raise ValueError("Hidden/key rows must align")
    visual_ids, text_ids = visual.nonzero().flatten(), text.nonzero().flatten()
    if len(visual_ids) < visual_pivots or len(text_ids) < text_pivots:
        raise ValueError("Insufficient native visual/language tokens for DART pivots")
    norms = keys.float().abs().sum(-1)
    vp = visual_ids[torch.argsort(norms[visual_ids], descending=True, stable=True)[:visual_pivots]]
    tp = text_ids[torch.argsort(norms[text_ids], descending=True, stable=True)[:text_pivots]]
    # The reference set iteration is not portable. Fix ascending native positions
    # so repeated processes/backends use the same sequential exclusion rule.
    pivots = torch.cat((vp, tp)).sort().values
    per_pivot = int((len(visual_ids) * ratio) // len(pivots))
    selected = torch.zeros_like(visual)
    selected[vp] = True
    available = visual.clone()
    available[vp] = False
    unit = F.normalize(hidden.float(), dim=-1)
    selected_count = visual_pivots
    for pivot in pivots.unbind():
        # Fixed counts permit device-only selection; saturation protects small fixtures.
        take = min(per_pivot, len(visual_ids) - selected_count)
        if take == 0:
            break
        similarity = unit @ unit.index_select(0, pivot.reshape(1))[0]
        candidates = torch.argsort(similarity.masked_fill(~available, float("inf")), stable=True)[
            :take
        ]
        selected[candidates] = True
        available[candidates] = False
        selected_count += take
    return (~visual) | selected
