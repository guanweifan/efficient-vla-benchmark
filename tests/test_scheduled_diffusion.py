"""Geometry tests without selecting a backend or paper."""

import numpy as np
import pytest

pytest.importorskip("torch")

from efficient_vla_benchmark.model_ops.scheduled_diffusion import inference_grid


def test_uniform_grid_is_exact_native_ten_step_reference():
    alpha = np.cos(np.linspace(0.02, 1.5, 100)) ** 2
    assert inference_grid(alpha, 90, 10, "time_uniform").tolist() == list(range(90, -1, -10))


@pytest.mark.parametrize("spacing", ["time_uniform", "log_snr"])
@pytest.mark.parametrize("steps", [3, 4, 6, 8, 10])
def test_budget_does_not_change_native_start_or_terminal_prediction(spacing, steps):
    alpha = np.cos(np.linspace(0.02, 1.5, 100)) ** 2
    grid = inference_grid(alpha, 90, steps, spacing)
    assert len(grid) == steps and grid[0] == 90 and grid[-1] == 0
    assert np.all(np.diff(grid) < 0)


def test_invalid_grid_rejected_instead_of_extrapolation():
    with pytest.raises(ValueError):
        inference_grid(np.linspace(0.99, 0.1, 100), 90, 1, "time_uniform")
    with pytest.raises(ValueError):
        inference_grid(np.linspace(0.99, 0.1, 100), 90, 3, "unknown")
