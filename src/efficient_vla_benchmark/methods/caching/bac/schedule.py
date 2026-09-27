"""ACS dynamic programming and BUA, independent of any model runtime."""

import numpy as np


def optimal_updates(similarity, budget):
    """Maximize source-to-reused-feature cosine sum with exactly M segments.

    The released ACS solver scores interval [a,b] by sum_t similarity[a,t].
    Step zero is compulsory. Ties select the earliest previous endpoint.
    """
    similarity = np.asarray(similarity, dtype=np.float64)
    if (
        similarity.ndim != 2
        or similarity.shape[0] != similarity.shape[1]
        or not np.isfinite(similarity).all()
        or type(budget) is not int
        or not 1 <= budget <= len(similarity)
    ):
        raise ValueError("Invalid ACS similarity matrix or update budget")
    n = len(similarity)
    costs = np.zeros((n, n))
    for start in range(n):
        costs[start, start:] = np.cumsum(similarity[start, start:])
    score = np.full((budget + 1, n), -np.inf)
    parent = np.zeros((budget + 1, n), dtype=np.int64)
    score[1] = costs[0]
    for segments in range(2, budget + 1):
        for end in range(segments - 1, n):
            options = [
                score[segments - 1, previous] + costs[previous + 1, end]
                for previous in range(segments - 2, end)
            ]
            previous = segments - 2 + int(np.argmax(options))
            score[segments, end] = max(options)
            parent[segments, end] = previous
    updates, end = [0], n - 1
    for segments in range(budget, 1, -1):
        previous = int(parent[segments, end])
        updates.append(previous + 1)
        end = previous
    return tuple(sorted(updates))


def make_schedule(similarities, errors, budget, protected_blocks):
    """CogACT ordered attention/MLP residual branches; apply paper Eq. 14.

    Same-layer MLP is downstream of attention. Descending traversal realizes the
    complete union invariant. Add final refresh as in the released runtime.
    """
    matrices, errors = np.asarray(similarities), np.asarray(errors)
    if (
        matrices.ndim != 3
        or matrices.shape[1] != matrices.shape[2]
        or matrices.shape[0] % 2
        or errors.shape != (matrices.shape[0],)
        or not np.isfinite(errors).all()
        or (errors < 0).any()
        or type(protected_blocks) is not int
        or not 1 <= protected_blocks <= len(errors)
    ):
        raise ValueError("Invalid BAC calibration geometry or protected-block count")
    schedules = [set(optimal_updates(matrix, budget)) for matrix in matrices]
    # Match the author's always-refreshed final denoising step, generalized from 99.
    for updates in schedules:
        updates.add(matrices.shape[1] - 1)
    protected = np.argsort(-errors, kind="stable")[:protected_blocks].tolist()
    for upstream in sorted(protected, reverse=True):
        for downstream in range(upstream + 1, len(schedules)):
            if downstream % 2:  # native FFN residual, including the current layer
                schedules[upstream].update(schedules[downstream])
    return tuple(tuple(sorted(s)) for s in schedules), tuple(protected)
