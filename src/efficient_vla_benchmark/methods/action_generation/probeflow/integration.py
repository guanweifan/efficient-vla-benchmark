"""Algorithm 1: lookahead, cosine scheduler and the two distinct reuse paths."""

import math

import torch
import torch.nn.functional as F


def step_count(similarity, config):
    if not math.isfinite(similarity):
        raise ValueError("Nonfinite probe similarity")
    similarity = min(1.0, max(-1.0, similarity))
    return min(
        config.max_steps, 2 + math.floor((1 - similarity) / config.epsilon) * config.step_increment
    )


def fixed_euler(velocity, initial, steps, start_velocity=None):
    value = initial
    for index in range(steps):
        v = (
            start_velocity
            if index == 0 and start_velocity is not None
            else velocity(value, index / steps)
        )
        value = value + v / steps
    return value


def integrate(velocity, initial, config):
    if config.diagnostic_fixed_steps:
        steps = config.diagnostic_fixed_steps
        return fixed_euler(velocity, initial, steps), dict(
            integration_steps=steps,
            denoiser_calls=steps,
            probe_similarity=None,
            diagnostic_fixed_euler=True,
        )
    start = velocity(initial, 0.0)
    probe = initial + config.probe_time * start
    lookahead = velocity(probe, config.probe_time)
    # Only the served conditional trajectory determines the budget, not the
    # auxiliary unconditional CFG row. Zero-norm vectors select the dense path.
    similarity = float(F.cosine_similarity(start[0].flatten(), lookahead[0].flatten(), dim=0))
    steps = step_count(similarity, config)
    if steps == 2:
        result = probe + (1 - config.probe_time) * lookahead
        evaluations = 2
    else:
        result = fixed_euler(velocity, initial, steps, start_velocity=start)
        evaluations = steps + 1  # The off-grid lookahead is NOT a free evaluation.
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite ProbeFlow trajectory")
    return result, dict(
        integration_steps=steps,
        denoiser_calls=evaluations,
        probe_similarity=similarity,
        diagnostic_fixed_euler=False,
    )
