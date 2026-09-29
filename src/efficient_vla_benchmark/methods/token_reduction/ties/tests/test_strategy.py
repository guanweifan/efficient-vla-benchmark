from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from ..config import TIESConfig
from ..strategy import TIESStrategy


@dataclass
class State:
    layout: object
    positions: object
    hidden: object = None


def test_prefix_is_real_and_control_refresh_is_not_query_refresh():
    host = SimpleNamespace(query_diagnostics={})
    cfg = TIESConfig(ranking_layers=3, ranking_topk=3, refresh_steps=24)
    method = TIESStrategy(host, cfg, SimpleNamespace(layers=[None] * 5), {"quantiles": [-1, 0, 1]})
    layout = SimpleNamespace(visual_mask=lambda positions: (positions >= 1) & (positions < 5))
    method.adapter = SimpleNamespace(state=State(layout, torch.arange(6)))
    calls = []

    def native(*args, **kwargs):
        for layer in range(5):
            calls.append((method.probing, layer))
            # Native probabilities are represented directly, with every query.
            scores = torch.arange(1, 7, dtype=torch.float32)
            scores = scores.flip(0) if layer == 1 else scores
            probabilities = (scores / scores.sum()).expand(6, 6)
            result = method.consume_attention(layer, method.adapter.state, probabilities)
            if result is not None:
                assert result.sum() == 4  # 2 protected + 2 visual.
        return "native-readout"

    run = method.wrap_forward(native)
    method.before_query({})
    assert run() == "native-readout"
    assert calls[:3] == [(True, 0), (True, 1), (True, 2)] and len(calls) == 8
    assert method.tau == -1 and method.fraction == 1
    calls.clear()
    for _ in range(8):
        method.after_action(None, 0)
    method.before_query({})
    assert run() == "native-readout" and len(calls) == 5 and not any(p for p, _ in calls)
    for _ in range(16):
        method.after_action(None, 0)
    calls.clear()
    method.before_query({})
    run()
    assert len(calls) == 8 and method.last_refresh == 24
    method.reset(None)
    assert method.last_refresh is None and method.control_step == 0 and method.generator is None


def test_missing_prefix_cannot_silently_supply_tau():
    method = TIESStrategy(
        SimpleNamespace(),
        TIESConfig(),
        SimpleNamespace(layers=[None] * 32),
        {"quantiles": [-1, 0, 1]},
    )
    method.adapter = SimpleNamespace(state=State(None, torch.arange(6)))
    with pytest.raises(RuntimeError, match="sentinel"):
        method.wrap_forward(lambda **kw: None)()
    assert not method.probing and method.adapter.state.hidden is None
