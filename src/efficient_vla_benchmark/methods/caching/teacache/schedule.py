"""Exact accumulated polynomial gate for a fixed timestep-only signal."""

import math


def refresh_schedule(input_changes, coefficients, threshold, warmup_steps):
    if not input_changes or not coefficients or warmup_steps < 1:
        raise ValueError("Incomplete TeaCache schedule")
    if any(not math.isfinite(x) or x < 0 for x in input_changes):
        raise ValueError("Invalid timestep-embedding changes")
    if any(not math.isfinite(x) for x in coefficients):
        raise ValueError("Invalid fitted polynomial")
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Invalid cache threshold")
    accumulated = 0.0
    result = []
    for step, difference in enumerate(input_changes):
        if step < warmup_steps or threshold == 0:
            fresh = True
        else:
            estimated = 0.0
            for coefficient in coefficients:
                estimated = estimated * difference + coefficient
            accumulated += estimated
            fresh = accumulated >= threshold or not math.isfinite(accumulated)
        if fresh:
            accumulated = 0.0
        result.append(fresh)
    return tuple(result)
