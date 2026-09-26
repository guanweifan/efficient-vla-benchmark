from types import SimpleNamespace

import pytest
import torch

from ..config import ASRConfig
from ..reuse import CleanKVReuse


def test_served_kv_never_mutates_clean_base():
    method = CleanKVReuse(SimpleNamespace(), ASRConfig(gate_layers=(0,)), 2)
    positions = torch.arange(6)
    state = SimpleNamespace(positions=positions, layout=SimpleNamespace(length=6))
    base = torch.arange(24.0).reshape(1, 2, 6, 2)
    method.begin_auxiliary_query()
    for layer in range(2):
        method.update_kv(layer, base, base + 1, state)
    method.state.record(torch.arange(4.0))
    method.state.publish()
    method.end_auxiliary_query()
    active = torch.tensor([0, 2, 5])
    state.positions = active
    for layer in range(2):
        updated, values = method.update_kv(layer, base[:, :, active] + 9, base[:, :, active], state)
        expected = base.clone()
        expected[:, :, active] += 9
        torch.testing.assert_close(updated, expected, rtol=0, atol=0)
        torch.testing.assert_close(method.store.keys[layer], base, rtol=0, atol=0)
        torch.testing.assert_close(method.store.values[layer], base + 1, rtol=0, atol=0)
        assert updated.data_ptr() != method.store.keys[layer].data_ptr()
        assert values.shape == base.shape
    method.reset(None)
    assert not method.store.keys and method.state.gate is None


def test_missing_base_is_explicit_failure():
    method = CleanKVReuse(SimpleNamespace(), ASRConfig(gate_layers=(0,)), 2)
    method.state.gate = torch.ones(4)
    state = SimpleNamespace(positions=torch.tensor([0, 5]), layout=SimpleNamespace(length=6))
    with pytest.raises(RuntimeError, match="dense KV"):
        method.update_kv(0, torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2), state)
