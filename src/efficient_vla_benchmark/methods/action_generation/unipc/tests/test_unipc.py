from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from ..config import UniPCConfig
from ..implementations.cogact.api import build

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")


def native_host():
    pytest.importorskip("efficient_vla_backend_cogact")
    from action_model import create_diffusion

    kwargs = dict(
        noise_schedule="squaredcos_cap_v2",
        diffusion_steps=100,
        sigma_small=True,
        learn_sigma=False,
    )
    full, sampler = create_diffusion("", **kwargs), create_diffusion("ddim10", **kwargs)
    action = SimpleNamespace(
        net=SimpleNamespace(training=False), diffusion=full, ddim_diffusion=sampler
    )
    return SimpleNamespace(
        model=SimpleNamespace(action_model=action),
        parameters=SimpleNamespace(num_ddim_steps=10),
        query_diagnostics={},
    ), sampler


def epsilon_model(x, t, z, **kwargs):
    return 0.15 * x.tanh() + z * 0.03 + t[:, None, None] * 0.0001


def test_config_rejects_noop_budgets_and_invalid_parameters():
    for bad in (
        {"enabled": 1},
        {"num_steps": True},
        {"num_steps": 2},
        {"num_steps": 11},
        {"order": 1},
        {"order": 3, "num_steps": 4},
        {"spacing": "unknown"},
        {"num_steps": float("nan")},
        {"unexpected": 3},
    ):
        with pytest.raises(ValueError):
            UniPCConfig.parse(bad)
    assert UniPCConfig.parse({}).num_steps == 6


@pytest.mark.parametrize("rows", [1, 2])
def test_first_order_native_grid_equivalence_rng_and_detach(rows):
    host, sampler = native_host()
    torch.manual_seed(17)
    noise = torch.randn(1, 16, 7).expand(rows, -1, -1).clone()
    state = torch.get_rng_state()
    kwargs = {"z": torch.arange(rows, dtype=torch.float32)[:, None, None]}
    if rows == 2:
        kwargs["cfg_scale"] = 1.5
    with torch.inference_mode():
        expected = sampler.ddim_sample_loop(
            epsilon_model,
            noise.shape,
            noise=noise,
            clip_denoised=False,
            model_kwargs=kwargs,
            device=noise.device,
        )
        expected_state = torch.get_rng_state()
        # Order one exists only as a diagnostic fixture, never a selectable preset.
        method = build(host, replace(UniPCConfig(), order=1, num_steps=10), None)
        # Isolate first-order predictor geometry; active UniC is tested separately.
        method.scheduler.disable_corrector = list(range(10))
        torch.set_rng_state(state)
        method.before_query({})
        actual = sampler.ddim_sample_loop(
            epsilon_model,
            noise.shape,
            noise=noise,
            clip_denoised=False,
            model_kwargs=kwargs,
            device=noise.device,
        )
        method.after_query(actual)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
    assert torch.equal(torch.get_rng_state(), expected_state)
    method.close()
    method.close()
    assert "ddim_sample_loop" not in sampler.__dict__


@pytest.mark.parametrize("steps", [3, 4, 5, 6, 8, 10])
@pytest.mark.parametrize("spacing", ["time_uniform", "log_snr"])
def test_high_order_active_and_history_resets(steps, spacing, monkeypatch):
    host, sampler = native_host()
    method = build(host, UniPCConfig(num_steps=steps, spacing=spacing), None)
    calls = []
    attribute = "multistep_uni_c_bh_update"
    original = getattr(method.scheduler, attribute)

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(method.scheduler, attribute, counted)
    noise = torch.linspace(-1, 1, 112).reshape(1, 16, 7)
    outputs = []
    with torch.inference_mode():
        for _ in range(2):
            method.reset(None)
            method.before_query({})
            outputs.append(
                sampler.ddim_sample_loop(
                    epsilon_model,
                    noise.shape,
                    noise=noise,
                    clip_denoised=False,
                    model_kwargs={"z": torch.ones(1, 1, 1)},
                )
            )
            method.after_query(outputs[-1])
    assert calls, "Defining high-order/corrector equation was never active"
    assert host.query_diagnostics["denoiser_calls"] == steps
    assert host.query_diagnostics["solver_timesteps"][0] == 90
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    assert torch.isfinite(outputs[0]).all()
    method.close()
    with pytest.raises(RuntimeError):
        method.before_query({})


@pytest.mark.gpu
def test_real_model_parity_active_solver_reset_and_cleanup(model_path, trace_path, request):
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    host = CogACTBackend(model_path)
    request.addfinalizer(host.close)
    context, frames, _ = load_trace(trace_path, require_control_steps=True)
    host.reset(context)
    native = host.predict_chunk(frames[0])
    native_rng = torch.cuda.get_rng_state().clone()
    for config in (
        UniPCConfig(enabled=False),
        UniPCConfig(),
        UniPCConfig(num_steps=3),
        UniPCConfig(num_steps=6, spacing="log_snr"),
        UniPCConfig(num_steps=6, order=3),
        UniPCConfig(num_steps=5, spacing="log_snr"),
        UniPCConfig(num_steps=5, order=3),
    ):
        method = build(host, config, None)
        host.attach_method(method)
        results = []
        for _ in range(2):
            host.reset(context)
            results.append(host.predict_chunk(frames[0]))
            assert results[-1].shape == (16, 7) and np.isfinite(results[-1]).all()
            assert torch.equal(torch.cuda.get_rng_state(), native_rng)
        np.testing.assert_array_equal(*results)
        if not config.enabled:
            np.testing.assert_array_equal(results[0], native)
        else:
            assert host.query_diagnostics["denoiser_calls"] == config.num_steps
        host.detach_method()
        assert "ddim_sample_loop" not in host.model.action_model.ddim_diffusion.__dict__
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(frames[0]), native)
