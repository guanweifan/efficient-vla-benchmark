import pytest
import torch

from ..config import ASRConfig
from ..state import CleanGate


def test_only_dense_refresh_can_supply_next_gate():
    state = CleanGate(0.5)
    mask = torch.tensor([False, True, True, True, True, False])
    assert state.selection(mask) is None
    with pytest.raises(RuntimeError, match="sparse"):
        state.record(torch.arange(4.0))
    state.begin_refresh()
    assert state.selection(mask) is None
    state.record(torch.tensor([0.1, 0.9, 0.8, 0.2]))
    state.publish()
    state.end_refresh()
    assert state.selection(mask).tolist() == [True, False, True, True, False, True]
    assert state.refreshes == 1
    state.reset()
    assert state.gate is None and state.refreshes == 0


def test_empty_refresh_is_not_silently_used():
    state = CleanGate(0.1)
    state.begin_refresh()
    with pytest.raises(RuntimeError, match="saliency"):
        state.publish()
    state.end_refresh()
    assert state.gate is None


@pytest.mark.parametrize(
    "params",
    [{"keep_ratio": 0}, {"gate_layers": []}, {"gate_layers": [32]}, {"actuation_window_ms": 0}],
)
def test_invalid_config(params):
    with pytest.raises(ValueError):
        ASRConfig.parse(params)
