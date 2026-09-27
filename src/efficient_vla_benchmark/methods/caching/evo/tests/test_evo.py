import copy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ..config import EVOConfig
from ..implementations.cogact.method import EVO
from ..implementations.cogact.prior import NativePrior
from ..search import Evolution, SearchBudget, confirm_target, prior_weights, validate_layout


def test_global_budget_endpoints_canonicalization_and_rejection():
    layout = [[0, 3], [3, 2, 0], [0, 1, 2, 3]]
    assert validate_layout(layout, 3, 4, 9) == ((0, 3), (0, 2, 3), (0, 1, 2, 3))
    for bad, budget in (([[0], [0, 3], [0, 1, 3]], 6), (layout, 8), ([[0, 0, 3]] * 3, 9)):
        with pytest.raises(ValueError):
            validate_layout(bad, 3, 4, budget)
    for bad in ({"refresh_budget": True}, {"refresh_budget": 0}, {"enabled": 1}, {"x": 1}):
        with pytest.raises(ValueError):
            EVOConfig.parse(bad)


def test_prior_soft_support_and_closed_loop_only_evolution():
    prior = np.arange(240).reshape(24, 10) / 240
    weights = prior_weights(prior)
    assert weights.min() > 0 and weights.sum() == pytest.approx(1)
    evolution = Evolution(prior, 120, 7)
    replica = Evolution(prior, 120, 7)
    candidates = evolution.initial()
    assert candidates == replica.initial()
    assert len(set(candidates)) == 4
    with pytest.raises(ValueError, match="rollout fitness"):
        evolution.next_generation()
    for index, candidate in enumerate(candidates):
        evolution.observe(candidate, 10 + index, 20)
    assert len({len(row) for row in evolution.layout(candidates[0])}) > 1
    for _ in range(2):
        old = set(evolution.seen)
        fresh = evolution.next_generation()
        assert len(fresh) == 2 and not set(fresh) & old
        assert candidates[3] in evolution.population  # Best rollout elite retained.
        for candidate in fresh:
            layout = evolution.layout(candidate)
            assert sum(map(len, layout)) == 120
            evolution.observe(candidate, 10, 20)
    with pytest.raises(ValueError, match="already evaluated"):
        evolution.observe(candidates[0], 20, 20)


def test_confirmation_cannot_be_replaced_by_quick_score_and_hard_budget():
    assert confirm_target(1.0, 0.9) == (True, False)
    assert confirm_target(0.85, 0.9, 1.0) == (False, False)
    assert confirm_target(0.9, 0.9, 0.86) == (True, False)
    assert confirm_target(0.9, 0.9, 0.875) == (True, True)
    budget = SearchBudget()
    budget.reserve("native", "reference", 40)
    for index in range(24):
        budget.reserve(f"quick-{index}", "candidate", 20)
    with pytest.raises(ValueError, match="candidate limit"):
        budget.reserve("quick-extra", "candidate", 20)
    for index in range(3):
        budget.reserve(f"confirm-{index}", "confirmation", 40)
    assert sum(r["episodes"] for r in budget.reservations) == 640
    with pytest.raises(ValueError, match="budget"):
        budget.reserve("extra", "confirmation", 1)
    with pytest.raises(ValueError, match="already exists"):
        budget.reserve("native", "reference", 40)


def binding(host, layout):
    return dict(
        num_blocks=2 * len(host.model_access.action_denoiser.blocks),
        num_steps=host.parameters.num_ddim_steps,
        precision=str(next(host.model_access.action_denoiser.parameters()).dtype),
        sampler_parameters=asdict(host.parameters),
        checkpoint_sha256=host._model_manifest["sha256"],
        layouts={str(sum(map(len, layout))): {"layout": layout}},
    )


def fixture():
    pytest.importorskip("efficient_vla_backend_cogact")
    from action_model.models import DiT
    from efficient_vla_backend_cogact.config import CogACTParameters

    torch.manual_seed(12)
    net = DiT(hidden_size=24, depth=2, num_heads=3, future_action_window_size=3).eval()
    with torch.no_grad():
        net.final_layer.linear.weight.normal_(std=0.1)
    host = SimpleNamespace(
        model_access=SimpleNamespace(action_denoiser=net),
        parameters=CogACTParameters(num_ddim_steps=4),
        _model_manifest={"sha256": "fixture"},
        query_diagnostics={},
        method=None,
    )
    return host, net


def test_prior_captures_direct_cfg_forward_and_preserves_arithmetic():
    host, net = fixture()
    original = copy.deepcopy(net)
    prior = NativePrior(host, queries_per_episode=1)
    x, z = torch.randn(2, 4, 7), torch.randn(2, 1, 4096)
    with torch.inference_mode():
        for step in range(8):
            t = torch.full((2,), step % 4)
            torch.testing.assert_close(
                net.forward_with_cfg(x, t, z, 1.5),
                original.forward_with_cfg(x, t, z, 1.5),
                rtol=0,
                atol=0,
            )
    assert prior.queries == 1 and prior.result().shape == (4, 4)
    prior.close()
    prior.close()
    assert "forward" not in net.__dict__


def test_full_update_native_rng_parity_disabled_and_close():
    host, net = fixture()
    native = copy.deepcopy(net)
    manifest = binding(host, [list(range(4)) for _ in range(4)])
    x, z = torch.randn(2, 4, 7), torch.randn(2, 1, 4096)
    for enabled in (True, False):
        method = EVO(host, EVOConfig(enabled=enabled, refresh_budget=16), manifest)
        with torch.inference_mode():
            method.before_query({})
            rng = torch.get_rng_state().clone()
            for step in range(4):
                t = torch.full((2,), step)
                torch.testing.assert_close(net(x, t, z), native(x, t, z), rtol=0, atol=0)
            method.after_query(None)
            assert torch.equal(torch.get_rng_state(), rng)
        method.close()
        method.close()
        assert "forward" not in net.__dict__
        assert all("forward" not in m.__dict__ for b in net.blocks for m in (b.attn, b.mlp))


def test_cached_branch_reference_counts_current_residual_and_episode_reset():
    host, net = fixture()
    native = copy.deepcopy(net)
    layout = [[0, 3], [0, 1, 3], [0, 1, 2, 3], [0, 2, 3]]
    method = EVO(host, EVOConfig(refresh_budget=12), binding(host, layout))
    # Independent layer-level oracle explicitly sums current residuals with the
    # previously refreshed native attention/MLP outputs, not cached whole layers.
    saved, timestep = {}, [-1]
    for layer, block in enumerate(native.blocks):

        def forward(x, b=block, layer=layer):
            for offset, (norm, module) in enumerate(((b.norm1, b.attn), (b.norm2, b.mlp))):
                index = layer * 2 + offset
                if timestep[0] in layout[index]:
                    saved[index] = module(norm(x))
                x = x + saved[index]
            return x

        block.forward = forward
    x, z = torch.randn(2, 4, 7), torch.randn(2, 1, 4096)
    outputs = []
    with torch.inference_mode():
        for _ in range(2):
            method.reset(None)
            method.before_query({})
            for step in range(4):
                timestep[0] = step
                t = torch.full((2,), step)
                actual, expected = net(x + step / 10, t, z), native(x + step / 10, t, z)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            outputs.append(actual)
            method.after_query(None)
            assert host.query_diagnostics["evo_branch_evaluations"] == 12
            assert host.query_diagnostics["evo_branch_reuses"] == 4
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    method.close()


@pytest.mark.gpu
def test_real_model_native_prior_full_update_sparse_rng_and_reset(model_path, trace_path, request):
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    host = CogACTBackend(model_path)
    request.addfinalizer(host.close)
    context, frames, _ = load_trace(trace_path, require_control_steps=True)
    host.reset(context)
    native = host.predict_chunk(frames[0])
    native_rng = torch.cuda.get_rng_state().clone()
    prior = NativePrior(host)
    try:
        prior.before_episode(context)
        host.reset(context)
        np.testing.assert_array_equal(host.predict_chunk(frames[0]), native)
        assert torch.equal(torch.cuda.get_rng_state(), native_rng)
        values = prior.result()
        assert values.shape == (24, 10) and np.isfinite(values).all()
        assert prior.queries == 1
    finally:
        prior.close()
    dense = [list(range(10)) for _ in range(24)]
    evolution = Evolution(values, 120, 0)
    sparse = evolution.layout(evolution.initial()[0])
    for layout in (dense, sparse):
        config = EVOConfig(refresh_budget=sum(map(len, layout)))
        method = EVO(host, config, binding(host, layout))
        host.attach_method(method)
        outputs = []
        for _ in range(2):
            host.reset(context)
            outputs.append(host.predict_chunk(frames[0]))
            assert torch.equal(torch.cuda.get_rng_state(), native_rng)
        np.testing.assert_array_equal(*outputs)
        if config.refresh_budget == 240:
            np.testing.assert_array_equal(outputs[0], native)
        assert host.query_diagnostics["evo_branch_evaluations"] == config.refresh_budget
        host.detach_method()
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(frames[0]), native)
