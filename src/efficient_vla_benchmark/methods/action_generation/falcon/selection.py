"""Current-observation posterior estimates and thresholded noise-level sampling."""

import torch


def clean_estimates(latents, epsilon, alphas):
    if (
        latents.ndim != 3
        or latents.shape != epsilon.shape
        or alphas.shape != (latents.shape[0],)
        or latents.shape[0] < 1
        or not bool(((alphas > 0) & (alphas <= 1)).all())
    ):
        raise ValueError("Invalid Falcon posterior-estimation geometry")
    with torch.autocast(device_type=latents.device.type, enabled=False):
        a = alphas.float()[:, None, None]
        return (latents.float() - (1 - a).sqrt() * epsilon.float()) / a.sqrt()


def reference_distances(estimates, previous, executed_horizon):
    if (
        estimates.ndim != 3
        or previous.ndim != 2
        or estimates.shape[1:] != previous.shape
        or type(executed_horizon) is not int
        or not 0 < executed_horizon < len(previous)
    ):
        raise ValueError("Falcon needs an overlapping unexecuted action tail")
    reference = previous[executed_horizon:]
    with torch.autocast(device_type=estimates.device.type, enabled=False):
        difference = estimates[:, : len(reference)].float() - reference.float()
        # Released code uses L2 / number of scalar entries, not RMS error.
        return difference.flatten(1).norm(dim=1) / reference.numel()


def selection_probabilities(distances, noise_levels, threshold, min_noise_level, temperature):
    if (
        distances.ndim != 1
        or distances.shape != noise_levels.shape
        or not bool(torch.isfinite(distances).all())
        or not bool((distances >= 0).all())
    ):
        raise ValueError("Invalid Falcon candidate distances")
    valid = (distances <= threshold) & (noise_levels > min_noise_level)
    indices = valid.nonzero().flatten()
    if not len(indices):
        return indices, distances.new_empty(0, dtype=torch.float32)
    with torch.autocast(device_type=distances.device.type, enabled=False):
        probabilities = (-noise_levels[indices].float() / temperature).softmax(dim=0)
    return indices, probabilities
