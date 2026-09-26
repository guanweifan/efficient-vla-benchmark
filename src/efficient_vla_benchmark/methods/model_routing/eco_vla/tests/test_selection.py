import pytest
import torch

from ..config import EcoVLAConfig
from ..selection import SparsityTrigger, importance, select_channels


def test_temporal_fusion_and_history_are_distinct():
    weight = torch.tensor([[1.0, 2.0, 3.0]])
    old = torch.tensor([[2.0, 4.0, 6.0]])
    new = torch.tensor([[8.0, 10.0, 12.0]])
    score, history = importance(weight, old, new, 0.7, 0.9)
    torch.testing.assert_close(score, weight * (0.7 * old + 0.3 * new))
    torch.testing.assert_close(history, 0.9 * old + 0.1 * new)
    torch.testing.assert_close(old, torch.tensor([[2.0, 4.0, 6.0]]))


def test_trigger_uses_only_previous_window_and_resets():
    trigger = SparsityTrigger(0, 2)
    # At quantile zero, including the current value would make '<' impossible.
    assert trigger.update(torch.tensor([[[1.0, 0.0]]])) == (True, None)
    assert trigger.update(torch.tensor([[[1.0, 0.0]]])) == (True, 1.0)
    dense, similarity = trigger.update(torch.tensor([[[0.0, 1.0]]]))
    assert dense and similarity == 0.0
    assert len(trigger.history) == 2
    assert trigger.update(torch.tensor([[[0.0, 1.0]]])) == (False, 1.0)
    trigger.reset()
    assert trigger.previous is None and not trigger.history


def test_coupled_head_groups_and_sorted_unique_channels():
    score = torch.tensor([[1.0, 1.0, 3.0, 4.0, 2.0, 2.0, 5.0, 5.0]])
    assert select_channels(score, 0.5, group_size=2).tolist() == [[2, 3, 6, 7]]
    assert select_channels(score, 0.25, alignment=4).shape == (1, 4)
    assert select_channels(torch.ones(1, 8), 0.5).tolist() == [[0, 1, 2, 3]]


@pytest.mark.parametrize(
    "parameters",
    [
        {"keep_ratio": 0},
        {"temporal_inertia": 1},
        {"history_momentum": 1},
        {"trigger_quantile": 1.1},
        {"history_window": 0},
        {"calibration_samples": 6},
    ],
)
def test_invalid_config(parameters):
    with pytest.raises(ValueError):
        EcoVLAConfig.parse(parameters)
