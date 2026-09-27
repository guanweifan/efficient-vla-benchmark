"""Independent implementation of KeyStone's guarded cluster medoid.

Equations 3--6 of arXiv:2605.08638. Deterministic first-C initialization,
allclose stopping and strict > guard follow the released author code at
dywsjtu/keystone, revision 29a15a606f7a32251ef48ac66c281e10c6785eb0.
"""

import torch


def select_index(candidates, config):
    """Return a real candidate index, never an averaged trajectory."""
    if (
        candidates.ndim != 3
        or candidates.shape[0] != config.num_samples
        or candidates.shape[1] < config.selection_horizon
        or candidates.shape[2] < 1
        or not candidates.is_floating_point()
    ):
        raise ValueError("Invalid KeyStone candidate geometry")
    if not torch.isfinite(candidates).all():
        raise ValueError("KeyStone received nonfinite candidates")
    points = candidates[:, : config.selection_horizon].flatten(1).float()
    distances = torch.cdist(points, points)
    medoid = distances.sum(1).argmin()
    triangle = torch.triu_indices(len(points), len(points), 1, device=points.device)
    typical = distances[triangle[0], triangle[1]].median()
    spread = (points.mean(0) - points[medoid]).norm() / (typical + 1e-8)
    clustered = bool(spread > config.unimodal_tau)
    if not clustered:
        return medoid, clustered
    centers = points[: config.num_clusters].clone()
    for _ in range(10):
        labels = torch.cdist(points, centers).argmin(1)
        membership = torch.nn.functional.one_hot(labels, config.num_clusters).to(points.dtype)
        counts = membership.sum(0)
        means = (membership.T @ points) / counts.clamp_min(1).unsqueeze(1)
        updated = torch.where(counts.unsqueeze(1) > 0, means, centers)
        if torch.allclose(centers, updated, atol=1e-6):
            break
        centers = updated
    labels = torch.cdist(points, centers).argmin(1)
    largest = torch.bincount(labels, minlength=config.num_clusters).argmax()
    members = torch.where(labels == largest)[0]
    within = distances[members[:, None], members[None, :]]
    return members[within.sum(1).argmin()], clustered
