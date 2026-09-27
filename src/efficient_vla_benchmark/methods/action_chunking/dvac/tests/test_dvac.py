from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from ..config import DVACConfig
from ..selection import select_prefix


@pytest.mark.parametrize(
    "bad",
    [
        {"enabled": 1},
        {"alpha": -1},
        {"alpha": float("nan")},
        {"alpha": True},
        {"tail_steps": 1},
        {"window_size": 0},
        {"min_horizon": 0},
        {"min_horizon": 17},
        {"max_horizon": 1.5},
        {"unknown": 2},
    ],
)
def test_invalid_config(bad):
    with pytest.raises(ValueError):
        DVACConfig.parse(bad)


def test_prior_buffer_and_exact_first_crossing():
    config = DVACConfig(max_horizon=4, alpha=0)
    history = [np.ones(4)]
    assert select_prefix([0, 1, 2, 0], history, config, 3) == (2, 1)
    assert select_prefix([2, 0, 0, 0], history, config, 3) == (1, 1)
    assert select_prefix([1, 1, 1, 1], history, config, 3) == (4, 1)
    assert select_prefix([0, 1, 99, 0], [], config, 3) == (3, None)
    np.testing.assert_array_equal(history[0], np.ones(4))


def test_scale_equivariance_and_bounds():
    config = DVACConfig(max_horizon=4, alpha=1, min_horizon=2)
    history = [np.array([0, 1, 2, 3])]
    a = select_prefix([0, 3, 8, 0], history, config, 3)
    b = select_prefix([0, 30, 80, 0], [history[0] * 10], config, 3)
    assert a[0] == b[0] == 2
    assert b[1] == pytest.approx(a[1] * 10)
    with pytest.raises(ValueError):
        select_prefix([0, 1, np.nan, 0], history, config, 3)


def native_host():
    pytest.importorskip("efficient_vla_backend_cogact")
    from action_model import create_diffusion

    kwargs = dict(
        noise_schedule="squaredcos_cap_v2", diffusion_steps=100, sigma_small=True, learn_sigma=False
    )
    full, sampler = create_diffusion("", **kwargs), create_diffusion("ddim10", **kwargs)
    host = SimpleNamespace(
        model=SimpleNamespace(
            action_model=SimpleNamespace(
                net=SimpleNamespace(training=False), diffusion=full, ddim_diffusion=sampler
            )
        ),
        parameters=SimpleNamespace(num_ddim_steps=10),
        query_diagnostics={},
    )
    return host, sampler


@pytest.mark.parametrize("rows", [1, 2])
def test_native_clean_capture_exact_actions_rng_and_restore(rows):
    torch = pytest.importorskip("torch")
    from ..implementations.cogact.api import build

    host, sampler = native_host()
    torch.manual_seed(8)
    noise = torch.randn(1, 16, 7).expand(rows, -1, -1).clone()
    state = torch.get_rng_state()
    kwargs = {"z": torch.arange(rows, dtype=torch.float32)[:, None, None]}
    clean_reference = []
    original = sampler.p_mean_variance

    def collect(*args, **kwargs):
        out = original(*args, **kwargs)
        clean_reference.append(out["pred_xstart"][0].clone())
        return out

    def epsilon(x, t, z):
        return 0.15 * x.tanh() + z * 0.2 + t[:, None, None] * 0.001

    sampler.p_mean_variance = collect
    with torch.inference_mode():
        expected = sampler.ddim_sample_loop(
            epsilon,
            noise.shape,
            noise=noise,
            clip_denoised=False,
            model_kwargs=kwargs,
            device=noise.device,
        )
    expected_rng = torch.get_rng_state().clone()
    del sampler.p_mean_variance
    method = build(host, DVACConfig(), None)
    for _ in range(2):
        method.reset(None)
        torch.set_rng_state(state)
        with torch.inference_mode():
            method.before_query({})
            actual = sampler.ddim_sample_loop(
                epsilon,
                noise.shape,
                noise=noise,
                clip_denoised=False,
                model_kwargs=kwargs,
                device=noise.device,
            )
            method.after_query(actual[0])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(expected_rng, torch.get_rng_state())
        reference_variance = (
            torch.stack(clean_reference[-5:]).var(dim=0, correction=0).sum(-1).numpy()
        )
        np.testing.assert_array_equal(method.variance, reference_variance)
        assert method.clean == []
        assert method.select_execution_horizon(actual[0], 8) == 8
        assert len(method.history) == 1
        with pytest.raises(RuntimeError, match="one horizon decision"):
            method.select_execution_horizon(actual[0], 8)
    method.close()
    method.close()
    assert "p_mean_variance" not in sampler.__dict__
    with pytest.raises(RuntimeError):
        method.before_query({})


def test_rolling_capacity_and_current_query_not_in_threshold():
    pytest.importorskip("torch")
    from ..implementations.cogact.api import build

    host, _ = native_host()
    method = build(host, replace(DVACConfig(), window_size=2, alpha=0), None)
    for value in (1, 3, 99):
        method.chosen = None
        method.variance = np.full(16, value, dtype=float)
        horizon = method.select_execution_horizon(np.zeros((16, 7)), 8)
    assert horizon == 1 and method.threshold == 2
    assert len(method.history) == 2
    assert [v[0] for v in method.history] == [3, 99]
    method.reset(None)
    assert len(method.history) == 0 and method.variance is None
    method.close()


def test_disabled_and_invalid_native_bounds():
    pytest.importorskip("torch")
    from ..implementations.cogact.api import build

    host, sampler = native_host()
    method = build(host, DVACConfig(enabled=False), None)
    assert "p_mean_variance" not in sampler.__dict__
    assert method.select_execution_horizon(np.zeros((16, 7)), 8) == 8
    method.close()
    for config in (DVACConfig(tail_steps=11), DVACConfig(max_horizon=17)):
        with pytest.raises(ValueError):
            build(host, config, None)


@pytest.mark.gpu
def test_real_model_prediction_rng_reset_and_adaptive_control(model_path, trace_path, request):
    torch = pytest.importorskip("torch")
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from efficient_vla_benchmark.control.chunk import ChunkPolicy
    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..implementations.cogact.api import build

    host = CogACTBackend(model_path)
    request.addfinalizer(host.close)
    context, frames, _ = load_trace(trace_path, require_control_steps=True)
    host.reset(context)
    native = [host.predict_chunk(frame) for frame in frames[:3]]
    native_rng = torch.cuda.get_rng_state().clone()
    for enabled in (False, True):
        method = build(host, DVACConfig(enabled=enabled), None)
        host.attach_method(method)
        for _ in range(2):
            host.reset(context)
            for i, frame in enumerate(frames[:3]):
                actions = host.predict_chunk(frame)
                np.testing.assert_array_equal(actions, native[i])
                horizon = host.select_execution_horizon(actions.copy(), 8)
                assert 1 <= horizon <= 16
            assert torch.equal(native_rng, torch.cuda.get_rng_state())
        if enabled:
            policy = ChunkPolicy(host, 8)
            policy.reset(context)
            decisions = [policy.step(frame) for frame in frames[:32]]
            assert sum(d.queried for d in decisions) == policy.query_count
            assert len(method.history) == min(policy.query_count, method.config.window_size)
            assert all(1 <= d.diagnostics["selected_execution_horizon"] <= 16 for d in decisions)
        host.detach_method()
        assert "p_mean_variance" not in host.model.action_model.ddim_diffusion.__dict__
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(frames[0]), native[0])
