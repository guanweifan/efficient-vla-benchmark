"""Released static-patch ordering and entropy-to-reuse schedule."""

import numpy as np
import torch

from efficient_vla_benchmark.model_ops.vision import patch_cosine_similarity


def static_patch_ids(current, previous, budget, threshold):
    similarity = patch_cosine_similarity(current, previous)
    # Match the reference's stable Python sort, not SpecPrune's argpartition.
    eligible = np.flatnonzero(similarity >= threshold)
    return eligible[np.argsort(-similarity[eligible], kind="stable")[:budget]]


def reuse_schedule(entropies, weighted_growth=True, growth_factor=0.55):
    entropies = torch.stack(entropies)
    normalized = (entropies - entropies.min()) / (entropies.max() - entropies.min() + 1e-10)
    reuse = 1 - normalized
    if weighted_growth:
        values = reuse.tolist()
        for index in range(1, len(values)):
            delta = values[index] - values[index - 1]
            if delta > 0:
                values[index] = values[index - 1] + delta * growth_factor
        return values
    return reuse.tolist()
