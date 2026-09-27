from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ..config import ProbeFlowConfig
from ..implementations.cogact.coordinates import DDIMCoordinates
from ..implementations.cogact.method import ProbeFlow
from ..integration import integrate, step_count


def samplers():
    pytest.importorskip("efficient_vla_backend_cogact")
    from action_model import create_diffusion

    kwargs = dict(
        noise_schedule="squaredcos_cap_v2", diffusion_steps=100, sigma_small=True, learn_sigma=False
    )
    return create_diffusion("", **kwargs), create_diffusion("ddim10", **kwargs)


def native_grid(bridge, sampler, model, noise, kwargs):
    y = noise / bridge.initial_scale
    for index in reversed(range(sampler.num_timesteps)):
        r = bridge.r_grid[sampler.timestep_map[index]]
        next_r = bridge.r_grid[sampler.timestep_map[index - 1]] if index else 0.0
        u, next_u = 1 - r / bridge.r_start, 1 - next_r / bridge.r_start
        y = y + (next_u - u) * bridge.velocity(model, y, u, kwargs)
    return y


def test_scheduler_paper_equation_monotonicity_and_validation():
    config = ProbeFlowConfig()
    assert [step_count(v, config) for v in (1, 0.999, 0.98, 0.94, -1)] == [2, 2, 6, 10, 10]
    assert [step_count(0.96, replace(config, epsilon=e)) for e in (0.001, 0.01, 0.1)] == [10, 10, 2]
    for bad in (
        {"enabled": 1},
        {"epsilon": float("nan")},
        {"epsilon": True},
        {"epsilon": 0},
        {"probe_time": 1},
        {"max_steps": 1},
        {"step_increment": 0},
        {"max_steps": True},
        {"diagnostic_fixed_steps": 1},
        {"unknown": 1},
    ):
        with pytest.raises(ValueError):
            ProbeFlowConfig.parse(bad)


def test_straight_path_reuses_both_evaluations_and_ignores_unconditional_score():
    calls = []
    initial = torch.ones(2, 4, 2)

    def velocity(x, t):
        calls.append(t)
        v = torch.ones_like(x)
        v[1] *= 1 - 4 * t  # Unserved CFG branch cannot drive the scheduler.
        return v

    result, info = integrate(velocity, initial, ProbeFlowConfig())
    assert calls == [0.0, 0.5]
    assert info["integration_steps"] == info["denoiser_calls"] == 2
    torch.testing.assert_close(result[0], torch.full((4, 2), 2.0))


def test_curved_path_uses_initial_evaluation_once_and_counts_discarded_probe():
    calls = []
    initial = torch.tensor([[[1.0, 0.0]]])

    def velocity(x, t):
        calls.append(t)
        return torch.stack((-x[..., 1], x[..., 0]), dim=-1)

    result, info = integrate(velocity, initial, ProbeFlowConfig())
    expected = initial
    for _ in range(10):
        expected = expected + torch.stack((-expected[..., 1], expected[..., 0]), dim=-1) / 10
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert info["integration_steps"] == 10 and info["denoiser_calls"] == 11
    assert calls == [0.0, 0.5, *[index / 10 for index in range(1, 10)]]


@pytest.mark.parametrize("rows", [1, 2])
def test_original_ddim_grid_coordinate_equivalence_and_endpoint(rows):
    diffusion, sampler = samplers()
    bridge = DDIMCoordinates(diffusion, sampler)
    bridge.validate(ProbeFlowConfig())
    seen = []

    def model(x, t, z):
        seen.append(t.clone())
        return 0.025 * x.tanh() + z * 0.001 + t[:, None, None] * 0.00001

    torch.manual_seed(9)
    noise = torch.randn(rows, 16, 7)
    kwargs = {"z": torch.arange(rows, dtype=torch.float32)[:, None, None]}
    with torch.inference_mode():
        expected = sampler.ddim_sample_loop(
            model,
            noise.shape,
            noise=noise,
            clip_denoised=False,
            model_kwargs=kwargs,
            device=noise.device,
        )
        actual = native_grid(bridge, sampler, model, noise, kwargs)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert bridge.start_index == 90
    assert 0 < bridge.r_start < 1
    for index in sampler.timestep_map:
        _, _, t = bridge.coefficients(1 - bridge.r_grid[index] / bridge.r_start)
        assert t == pytest.approx(index, abs=1e-10)
    for time in (-0.1, 1.0, 0.99999):
        with pytest.raises(ValueError):
            bridge.coefficients(time)
    with pytest.raises(ValueError):
        bridge.validate(ProbeFlowConfig(max_steps=1000))


def test_sampler_cfg_rng_lifecycle_and_detach():
    diffusion, sampler = samplers()
    host = SimpleNamespace(
        model=SimpleNamespace(
            action_model=SimpleNamespace(
                net=SimpleNamespace(training=False),
                diffusion=diffusion,
                ddim_diffusion=sampler,
            )
        ),
        parameters=SimpleNamespace(num_ddim_steps=10),
        query_diagnostics={},
    )
    method = ProbeFlow(host, ProbeFlowConfig())

    def model(x, t, z, cfg_scale):
        return x * 0.01 + z * 0.001

    torch.manual_seed(21)
    noise = torch.randn(1, 16, 7).expand(2, -1, -1).clone()
    start_state = torch.get_rng_state().clone()
    for _ in range(10):
        torch.randn_like(noise)
    expected_rng = torch.get_rng_state().clone()
    outputs = []
    with torch.inference_mode():
        for _ in range(2):
            torch.set_rng_state(start_state)
            method.reset(None)
            method.before_query({})
            out = sampler.ddim_sample_loop(
                model,
                noise.shape,
                noise=noise,
                clip_denoised=False,
                model_kwargs={"z": torch.tensor([0.1, -0.2])[:, None, None], "cfg_scale": 1.5},
            )
            outputs.append(out)
            method.after_query(out)
            assert torch.equal(torch.get_rng_state(), expected_rng)
            assert host.query_diagnostics["denoiser_calls"] in (2, 5, 7, 9, 11)
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    method.close()
    method.close()
    assert "ddim_sample_loop" not in sampler.__dict__
    with pytest.raises(RuntimeError):
        method.before_query({})


@pytest.mark.gpu
def test_real_native_grid_bridge_adaptive_rng_reset_and_detach(model_path, trace_path, request):
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace
    from efficient_vla_benchmark.model_ops.forward_patch import ForwardPatches

    host = CogACTBackend(model_path)
    request.addfinalizer(host.close)
    context, frames, _ = load_trace(trace_path, require_control_steps=True)
    host.reset(context)
    native = host.predict_chunk(frames[0])
    native_rng = torch.cuda.get_rng_state().clone()
    action = host.model.action_model
    sampler = action.ddim_diffusion
    bridge = DDIMCoordinates(action.diffusion, sampler)
    patch = ForwardPatches()

    def reference(original):
        def sample(model, shape, noise=None, model_kwargs=None, **kwargs):
            return native_grid(bridge, sampler, model, noise, model_kwargs)

        return sample

    patch.wrap(sampler, reference, "ddim_sample_loop")
    try:
        host.reset(context)
        actual = host.predict_chunk(frames[0])
        np.testing.assert_allclose(actual, native, rtol=2e-4, atol=2e-4)
    finally:
        patch.close()
    for config in (
        ProbeFlowConfig(enabled=False),
        ProbeFlowConfig(),
        ProbeFlowConfig(epsilon=0.1),
        ProbeFlowConfig(diagnostic_fixed_steps=10),
    ):
        method = ProbeFlow(host, config)
        host.attach_method(method)
        outputs = []
        for _ in range(2):
            host.reset(context)
            outputs.append(host.predict_chunk(frames[0]))
            assert outputs[-1].shape == (16, 7) and np.isfinite(outputs[-1]).all()
            assert torch.equal(torch.cuda.get_rng_state(), native_rng)
        np.testing.assert_array_equal(*outputs)
        if not config.enabled:
            np.testing.assert_array_equal(outputs[0], native)
        else:
            assert 2 <= host.query_diagnostics["denoiser_calls"] <= 11
        host.detach_method()
        assert "ddim_sample_loop" not in sampler.__dict__
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(frames[0]), native)
