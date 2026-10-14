"""Explicit offline intersection estimator for the paper's otherwise unspecified fitting step."""

import numpy as np


def fit_thresholds(records, config):
    if len(records) < 4:
        raise ValueError("Insufficient independent trajectory calibration")
    for row in records:
        values = [row["sensitivity"], *(row["errors"][str(b)] for b in (2, 4, 8))]
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError("Nonfinite or negative DyQ calibration evidence")
    ordered = sorted(records, key=lambda x: x["sensitivity"])
    # Equal-count bins estimate conditional mean error, never max(accepted frame).
    bins = np.array_split(np.arange(len(ordered)), min(config.sensitivity_bins, len(ordered) // 4))
    groups = []
    for ids in bins:
        rows = [ordered[int(i)] for i in ids]
        s = float(np.mean([r["sensitivity"] for r in rows]))
        errors = {str(b): float(np.mean([r["errors"][str(b)] for r in rows])) for b in (2, 4, 8)}
        groups.append(
            dict(
                sensitivity=s,
                upper=max(r["sensitivity"] for r in rows),
                count=len(rows),
                errors=errors,
            )
        )
    boundaries = {}
    # Stop at the FIRST mean-error crossing: no isolated high-S acceptance can
    # incorrectly extend a low-precision interval over an unsafe lower bin.
    for bits in (2, 4):
        bound = -1.0
        for g in groups:
            if g["errors"][str(bits)] * (g["sensitivity"] + config.eta) > config.accuracy_budget:
                break
            bound = min(config.theta_fp, float(g["upper"]))
        boundaries[bits] = bound
    boundaries[2] = min(boundaries[2], boundaries[4])
    return dict(theta_2_4=boundaries[2], theta_4_8=boundaries[4], theta_fp=config.theta_fp), groups
