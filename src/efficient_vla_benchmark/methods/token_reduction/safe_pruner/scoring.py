"""SAFE-Pruner feature correspondence, saliency transport and fusion equations."""

from torch.nn import functional as F


def correspondences(current, reference):
    a = F.normalize(current, p=2, dim=1).float()
    b = F.normalize(reference, p=2, dim=1).float()
    return (a @ b.T).argmax(dim=1)


def transfer_saliency(current, reference, saliency):
    if reference.shape[0] != saliency.shape[-1]:
        raise ValueError("Reference features and saliency must share visual patch IDs")
    return saliency.index_select(1, correspondences(current, reference))


def fused_selection(shallow, forecast, weight, count):
    return (shallow + weight * forecast.mean(0)).topk(count).indices
