"""Real CogACT: noise labels, CFG probes, native fallbacks and partial-tail parity."""

import numpy as np
import pytest
import torch

from ..config import FalconConfig
from ..implementations.cogact.api import build


@pytest.mark.gpu
def test_real_native_history_probe_partial_sampling_reset_rng_and_detach(
    model_path,
    trace_path,
    request,
):
    pytest.importorskip("efficient_vla_backend_cogact")
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    host = CogACTBackend(model_path)
    request.addfinalizer(host.close)
    context, frames, _ = load_trace(trace_path, require_control_steps=True)
    observations = [frames[min(i * 8, len(frames) - 1)] for i in range(3)]
    host.reset(context)
    native, native_rng = [], []
    for observation in observations:
        native.append(host.predict_chunk(observation))
        native_rng.append(torch.cuda.get_rng_state().clone())
    sampler = host.model.action_model.ddim_diffusion
    original_step = sampler.ddim_sample
    for config in (
        FalconConfig(enabled=False),
        FalconConfig(exploration_rate=1, distance_threshold=1e6),
        FalconConfig(exploration_rate=0, distance_threshold=1e6),
    ):
        method = build(host, config, None)
        host.attach_method(method)
        repeats = []
        for _ in range(2):
            host.reset(context)
            results = []
            for index, observation in enumerate(observations):
                entries = list(method.buffer.entries)
                calls, transitions = [], []

                def capture(module, args):
                    calls.append(args[0].detach().cpu().numpy().copy())

                def step(model, value, times, **kwargs):
                    transitions.append((value.detach().clone(), times.detach().clone(), kwargs))
                    return original_step(model, value, times, **kwargs)

                handle = host.model.action_model.net.t_embedder.register_forward_pre_hook(capture)
                sampler.ddim_sample = step
                try:
                    results.append(host.predict_chunk(observation))
                finally:
                    handle.remove()
                    del sampler.ddim_sample
                assert results[-1].shape == (16, 7) and np.isfinite(results[-1]).all()
                assert torch.equal(torch.cuda.get_rng_state(), native_rng[index])
                if not config.enabled or index == 0 or config.exploration_rate == 1:
                    np.testing.assert_array_equal(results[-1], native[index])
                if config.enabled:
                    assert method.previous is not None
                    if index == 0:
                        assert len(calls) == len(transitions) == 10
                        for entry, (value, times, _) in zip(method.buffer.entries, transitions):
                            assert entry.step == int(times[0])
                            assert entry.noise_level == sampler.timestep_map[entry.step]
                            torch.testing.assert_close(entry.latent, value[0], rtol=0, atol=0)
                    else:
                        expected_times = [entry.noise_level for entry in entries] * 2
                        np.testing.assert_array_equal(calls[0], expected_times)
                        assert host.query_diagnostics["falcon_probe_candidates"] == len(entries)
                        assert len(calls) == len(transitions) + 1
                        assert host.query_diagnostics["denoiser_evaluated_rows"] == 2 * (
                            len(entries) + len(transitions)
                        )
                        if config.exploration_rate == 0:
                            assert host.query_diagnostics["falcon_reused"]
                            assert len(transitions) < 10
                            value, times, kwargs = transitions[0]
                            assert any(
                                entry.step == int(times[0]) and torch.equal(entry.latent, value[0])
                                for entry in entries
                            )
                            # Independent native replay from the captured starting latent.
                            # Restore RNG afterwards; this validation is not measured work.
                            rng = torch.cuda.get_rng_state().clone()
                            with torch.inference_mode():
                                for offset in range(int(times[0]), -1, -1):
                                    value = original_step(
                                        host.model.action_model.net.forward_with_cfg,
                                        value,
                                        torch.full_like(times, offset),
                                        **kwargs,
                                    )["sample"]
                            torch.cuda.set_rng_state(rng)
                            torch.testing.assert_close(method.previous, value[0], rtol=0, atol=0)
                for remaining in range(7, -1, -1):
                    assert host.after_action(results[-1][7 - remaining], remaining) is False
            repeats.append(results)
        for first, second in zip(*repeats):
            np.testing.assert_array_equal(first, second)
        host.detach_method()
        method.close()
        assert not method.buffer.entries and method.previous is None
        assert "ddim_sample_loop" not in sampler.__dict__
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(observations[0]), native[0])
