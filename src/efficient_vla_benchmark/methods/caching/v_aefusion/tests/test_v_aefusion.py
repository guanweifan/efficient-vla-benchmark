from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ..config import VAEFusionConfig
from ..implementations.cogact.api import validate_experiment
from ..implementations.cogact.method import VAEFusion
from ..implementations.cogact.sampling import private_ddim_step


def test_config_and_fresh_tail():
    for bad in ({"stale_steps": -1}, {"stale_steps": True}, {"enabled": 1}, {"extra": 0}):
        with pytest.raises(ValueError):
            VAEFusionConfig.parse(bad)
    for steps in (1, 5, 10):
        experiment = SimpleNamespace(backend_parameters={"num_ddim_steps": steps})
        with pytest.raises(ValueError, match="fresh"):
            validate_experiment(experiment, VAEFusionConfig(stale_steps=steps))
        validate_experiment(experiment, VAEFusionConfig(stale_steps=steps - 1))


@pytest.mark.parametrize("rows", [1, 2])
def test_private_sampler_exact_native_arithmetic_and_global_rng_isolation(rows):
    pytest.importorskip("efficient_vla_backend_cogact")
    from action_model import create_diffusion

    sampler = create_diffusion("ddim10", learn_sigma=False)
    seen = []

    def model(x, t, z):
        seen.append(t.clone())
        return x * 0.025 + z * 0.002 + t[:, None, None] * 0.00001

    torch.manual_seed(34)
    state = torch.get_rng_state().clone()
    initial = torch.randn(1, 16, 7)
    noise = initial.expand(rows, -1, -1).clone()
    z = torch.arange(rows, dtype=torch.float32)[:, None, None]
    kwargs = {"z": z}
    with torch.inference_mode():
        native = sampler.ddim_sample_loop(
            model,
            noise.shape,
            noise=noise,
            clip_denoised=False,
            model_kwargs=kwargs,
            device=noise.device,
        )
        native_rng = torch.get_rng_state().clone()
        generator = torch.Generator().set_state(state)
        first = torch.randn((1, 16, 7), generator=generator)
        result = first.expand(rows, -1, -1).clone()
        for index in reversed(range(10)):
            result = private_ddim_step(
                sampler, model, result, torch.full((rows,), index), kwargs, generator
            )
    torch.testing.assert_close(native, result, rtol=0, atol=0)
    assert torch.equal(generator.get_state(), native_rng)
    assert torch.equal(torch.get_rng_state(), native_rng)
    assert seen[0][0] == 900 and seen[9][0] == 0  # Native timestep mapping.


def test_disabled_lifecycle_has_no_cuda_or_worker_dependency():
    host = SimpleNamespace(query_diagnostics={})
    method = VAEFusion(host, VAEFusionConfig(enabled=False))
    method.before_query({})
    with pytest.raises(RuntimeError, match="unfinished"):
        method.before_query({})
    method.after_query(None)
    method.reset(None)
    method.close()
    method.close()
    assert method.executor is None and not method.patches.saved
    with pytest.raises(RuntimeError, match="Closed"):
        method.before_query({})


def test_worker_failure_propagates_and_reset_clears_state():
    method = VAEFusion(SimpleNamespace(query_diagnostics={}), VAEFusionConfig(enabled=False))
    method.future = Future()
    method.future.set_exception(RuntimeError("worker failed"))
    method.previous = object()
    with pytest.raises(RuntimeError, match="worker failed"):
        method.reset(None)
    assert method.future is None and method.previous is None
    method.close()


@pytest.mark.gpu
def test_real_native_serial_stale_oracle_rng_overlap_reset_and_detach(
    model_path, trace_path, request
):
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace
    from efficient_vla_benchmark.model_ops.forward_patch import ForwardPatches

    host = CogACTBackend(model_path)
    request.addfinalizer(host.close)
    context, frames, _ = load_trace(trace_path, require_control_steps=True)
    observations = [frames[0], frames[min(8, len(frames) - 1)]]
    host.reset(context)
    native = [host.predict_chunk(frame) for frame in observations]
    native_rng = torch.cuda.get_rng_state().clone()
    for config in (VAEFusionConfig(enabled=False), VAEFusionConfig(stale_steps=0)):
        host.attach_method(VAEFusion(host, config))
        host.reset(context)
        actual = [host.predict_chunk(frame) for frame in observations]
        np.testing.assert_array_equal(actual, native)
        assert torch.equal(torch.cuda.get_rng_state(), native_rng)
        host.detach_method()

    sampler = host.model.action_model.ddim_diffusion
    for stale in (1, 5, 9):
        # Independent SERIAL oracle, with native transitions and no private RNG.
        previous = []
        patch = ForwardPatches()

        def reference(original):
            def sample(model, shape, noise=None, model_kwargs=None, **kwargs):
                if not previous:
                    previous.append(dict(model_kwargs, z=model_kwargs["z"].clone()))
                    return original(model, shape, noise=noise, model_kwargs=model_kwargs, **kwargs)
                x = noise
                for count, index in enumerate(reversed(range(sampler.num_timesteps))):
                    condition = previous[0] if count < stale else model_kwargs
                    x = sampler.ddim_sample(
                        model,
                        x,
                        torch.full((shape[0],), index, device=x.device),
                        clip_denoised=False,
                        model_kwargs=condition,
                        eta=0.0,
                    )["sample"]
                previous[0] = dict(model_kwargs, z=model_kwargs["z"].clone())
                return x

            return sample

        host.reset(context)
        patch.wrap(sampler, reference, "ddim_sample_loop")
        try:
            expected = [host.predict_chunk(frame) for frame in observations]
        finally:
            patch.close()
        expected_rng = torch.cuda.get_rng_state().clone()
        method = VAEFusion(host, replace(VAEFusionConfig(), stale_steps=stale))
        host.attach_method(method)
        host.reset(context)
        actual = [host.predict_chunk(frame) for frame in observations]
        np.testing.assert_array_equal(actual, expected)
        assert torch.equal(torch.cuda.get_rng_state(), expected_rng)
        assert host.query_diagnostics["stale_denoiser_steps"] == stale
        assert host.query_diagnostics["fresh_denoiser_steps"] == 10 - stale
        assert host.query_diagnostics["pipeline_stream_interval_overlap_ms"] > 0
        assert method.future is None and not method.events
        host.reset(context)
        np.testing.assert_array_equal(host.predict_chunk(observations[0]), native[0])
        assert host.query_diagnostics["stale_denoiser_steps"] == 0
        host.detach_method()
        assert "ddim_sample_loop" not in sampler.__dict__
        assert "forward" not in host.model.vlm.__dict__
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(observations[0]), native[0])
