"""Revised TIES: Kendall on common top-K identities, not raw index arrays."""

import math


class UndefinedRankCorrelation(ValueError):
    pass


def adjacent_kendall(first, second):
    if len(set(first)) != len(first) or len(set(second)) != len(second):
        raise ValueError("Rankings must contain unique token identities")
    positions = {token: index for index, token in enumerate(second)}
    relative = [positions[token] for token in first if token in positions]
    n = len(relative)
    if n < 2:
        raise UndefinedRankCorrelation("Adjacent top-K intersection contains fewer than two tokens")
    concordant = sum(relative[i] < relative[j] for i in range(n) for j in range(i + 1, n))
    pairs = n * (n - 1) // 2
    return 2 * concordant / pairs - 1


def rank_consistency(rankings):
    if len(rankings) < 2:
        raise ValueError("TIES requires at least two ranked layers")
    values = [adjacent_kendall(a, b) for a, b in zip(rankings, rankings[1:])]
    return sum(values) / len(values)


def attention_fraction(tau, quantiles, mode):
    if len(quantiles) != 3 or not all(math.isfinite(x) and -1 <= x <= 1 for x in [tau, *quantiles]):
        raise ValueError("Invalid Kendall score or calibration quantiles")
    q25, q50, q75 = quantiles
    if not q25 <= q50 <= q75:
        raise ValueError("Calibration quantiles are not ordered")
    if mode == "hard":
        return float(tau <= q50)
    if mode != "soft" or q75 == q25:
        raise ValueError("Soft-TIES requires nondegenerate calibration quantiles")
    return min(1.0, max(0.0, (q75 - tau) / (q75 - q25)))
