"""Paper equations 1--2: native CLS saliency times marginal cosine diversity."""

import torch
import torch.nn.functional as F


def select_current_frame(patches, cls, target):
    if (
        patches.ndim != 2
        or cls.shape != (patches.shape[-1],)
        or patches.device != cls.device
        or not patches.is_floating_point()
        or not cls.is_floating_point()
        or not bool(torch.isfinite(patches).all() & torch.isfinite(cls).all())
    ):
        raise ValueError("A-MMR needs finite aligned native patch and CLS features")
    count = patches.shape[0]
    if type(target) is not int or not 1 <= target <= count:
        raise ValueError("A-MMR target must retain between one and all patches")
    features = F.normalize(patches.float(), dim=-1, eps=1e-12)
    global_feature = F.normalize(cls.float(), dim=-1, eps=1e-12)
    attention = features @ global_feature
    importance = (attention - attention.min()) / (attention.max() - attention.min() + 1e-6)
    similarities = (features @ features.T).clamp(-1, 1)
    selected = torch.zeros(count, dtype=torch.bool, device=patches.device)
    maximum_similarity = None
    for _ in range(target):
        # The empty-set seed is maximum saliency, as described in section III-B.
        # Do not clamp negative similarities to zero: distinctness may exceed one.
        scores = importance if maximum_similarity is None else importance * (1 - maximum_similarity)
        index = scores.masked_fill(selected, -torch.inf).argmax().reshape(1)
        selected.scatter_(0, index, True)
        current_similarity = similarities.index_select(0, index)[0]
        maximum_similarity = (
            current_similarity
            if maximum_similarity is None
            else torch.maximum(maximum_similarity, current_similarity)
        )
    return selected
