"""Model-independent exact FlashVLA ICS equation; imported only by selected runtimes."""

import torch


def information_scores(features, driver="auto"):
    # Exact Eq. 2; no random/low-rank approximation, squared-energy substitution,
    # discarded singular directions, or hidden offline token-selection pass.
    if driver not in ("auto", "gesvd", "gesvdj"):
        raise ValueError("Only complete, non-approximate SVD drivers are supported")
    # PyTorch auto uses Jacobi with QR-based gesvd fallback on nonconvergence.
    # Both compute ALL singular directions; neither is gesvda/low-rank SVD.
    kwargs = {"driver": None if driver == "auto" else driver} if features.is_cuda else {}
    left, singular, _ = torch.linalg.svd(features.float(), full_matrices=False, **kwargs)
    return (left * singular).abs().sum(-1)
