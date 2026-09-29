"""Independent posterior, selection, CFG and historical DDIM trajectory oracles."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ..buffer import LatentBuffer
from ..config import FalconConfig
from ..implementations.cogact.api import validate_experiment
from ..method import Falcon
from ..selection import clean_estimates, reference_distances, selection_probabilities


def test_posterior_distance_and_softmax_match_independent_numpy():
    latents = np.linspace(-1.2, 1.7, 3 * 16 * 7).reshape(3, 16, 7).astype(np.float32)
    epsilon = np.sin(latents) * 0.13
    alphas = np.array([0.2, 0.6, 0.95], dtype=np.float32)
    previous = np.linspace(0.1, 0.8, 112).reshape(16, 7).astype(np.float32)
    expected = (latents - np.sqrt(1 - alphas[:, None, None]) * epsilon) / np.sqrt(
        alphas[:, None, None]
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = clean_estimates(
            torch.from_numpy(latents), torch.from_numpy(epsilon), torch.from_numpy(alphas)
        )
        distances = reference_distances(actual, torch.from_numpy(previous), 8)
    assert actual.dtype == distances.dtype == torch.float32
    np.testing.assert_allclose(actual.numpy(), expected, atol=3e-7, rtol=3e-7)
    reference = previous[8:]
    expected_distances = (
        np.linalg.norm((expected[:, :8] - reference).reshape(3, -1), axis=1) / reference.size
    )
    np.testing.assert_allclose(distances.numpy(), expected_distances, atol=1e-7)
    indices, probabilities = selection_probabilities(
        torch.tensor([0.1, 0.2, 0.15, 0.09, 0.4]),
        torch.tensor([20, 30, 40, 10, 50]),
        0.2,
        10,
        20,
    )
    np.testing.assert_array_equal(indices.numpy(), [0, 1, 2])
    weights = np.exp(-np.array([20, 30, 40]) / 20)
    np.testing.assert_allclose(probabilities.numpy(), weights / weights.sum(), rtol=1e-6)
    indices, probabilities = selection_probabilities(
        torch.tensor([0.2]),
        torch.tensor([10]),
        0.2,
        10,
        1,
    )
    assert indices.numel() == probabilities.numel() == 0


@pytest.mark.parametrize("horizon", [0, 16, 20, True])
def test_invalid_reference_overlap(horizon):
    with pytest.raises(ValueError, match="overlapping"):
        reference_distances(torch.ones(2, 16, 7), torch.ones(16, 7), horizon)


@pytest.mark.parametrize(
    "parameters",
    [
        {"enabled": 1},
        {"distance_threshold": 0},
        {"distance_threshold": float("nan")},
        {"temperature": 0},
        {"temperature": float("inf")},
        {"exploration_rate": -0.1},
        {"exploration_rate": 1.1},
        {"buffer_size": True},
        {"buffer_size": 0},
        {"buffer_size": 129},
        {"min_noise_level": -1},
        {"min_noise_level": 2.5},
        {"reference_horizon": 0},
        {"selection_seed": -1},
        {"selection_seed": 2**32},
        {"unknown": 1},
    ],
)
def test_invalid_config(parameters):
    with pytest.raises(ValueError):
        FalconConfig.parse(parameters)


def test_buffer_clones_latents_and_evicts_oldest_then_highest_noise():
    buffer = LatentBuffer(3)
    value = torch.arange(112.0).reshape(16, 7)
    buffer.add(value, 0, 1, 10)
    buffer.add(value + 1, 0, 3, 30)
    buffer.add(value + 2, 1, 9, 90)
    value.add_(1000)
    assert buffer.entries[1].latent[0, 0] == 0
    buffer.add(value, 1, 8, 80)
    assert [(e.query, e.noise_level) for e in buffer.entries] == [(0, 10), (1, 90), (1, 80)]
    buffer.add(value, 2, 7, 70)
    assert [(e.query, e.noise_level) for e in buffer.entries] == [(1, 90), (1, 80), (2, 70)]
    buffer.clear()
    assert not buffer.entries


class NativeSampler:
    """Tiny native-grid fixture, not a substitute for the real sampler tests."""

    timestep_map = list(range(0, 100, 10))

    def __init__(self, alphas):
        self.alphas = alphas
        self.transitions = []

    def ddim_sample(self, model, value, times, *, model_kwargs, **kwargs):
        step = int(times[0])
        self.transitions.append((step, value.clone()))
        raw_time = self.timestep_map[step]
        epsilon = model(value, torch.full_like(times, raw_time), **model_kwargs)
        alpha = float(self.alphas[raw_time])
        prior = float(self.alphas[self.timestep_map[step - 1]]) if step else 1.0
        clean = (value - np.sqrt(1 - alpha) * epsilon) / np.sqrt(alpha)
        torch.randn_like(value)
        return {"sample": np.sqrt(prior) * clean + np.sqrt(1 - prior) * epsilon}

    def ddim_sample_loop(self, model, shape, *, noise, model_kwargs, **kwargs):
        value = noise
        for step in range(9, -1, -1):
            value = self.ddim_sample(
                model, value, torch.full((len(value),), step), model_kwargs=model_kwargs
            )["sample"]
        return value


class Predictor:
    def __init__(self):
        self.calls = []

    def __call__(self, value, times, z, cfg_scale=1):
        self.calls.append((value.clone(), times.clone(), z.clone()))
        prediction = 0.15 * value.tanh() + times[:, None, None] * 0.001 + z * 0.02
        if cfg_scale > 1:
            conditional, unconditional = prediction.chunk(2)
            guided = unconditional + cfg_scale * (conditional - unconditional)
            return torch.cat((guided, guided))
        return prediction


def host_contract():
    betas = np.linspace(0.001, 0.02, 100)
    alphas = np.cumprod(1 - betas)
    contract = SimpleNamespace(
        sampler=NativeSampler(alphas),
        alphas_cumprod=alphas,
        betas=betas,
        native_steps=10,
        start_index=90,
    )
    return SimpleNamespace(query_diagnostics={}), contract


def query(method, predictor, noise, kwargs):
    method.before_query({})
    result = method.contract.sampler.ddim_sample_loop(
        predictor,
        noise.shape,
        noise=noise,
        clip_denoised=False,
        model_kwargs=kwargs,
    )
    method.after_query(result)
    for remaining in range(7, -1, -1):
        assert method.after_action(result[0, 7 - remaining], remaining) is False
    return result


def numpy_tail(value, start, alphas, z, cfg_scale):
    value = value.numpy().astype(np.float64)
    for step in range(start, -1, -1):
        raw_time = step * 10
        epsilon = 0.15 * np.tanh(value) + raw_time * 0.001 + z.numpy() * 0.02
        if cfg_scale > 1:
            cond, uncond = np.split(epsilon, 2)
            epsilon = np.concatenate([uncond + cfg_scale * (cond - uncond)] * 2)
        alpha = alphas[raw_time]
        prior = alphas[raw_time - 10] if step else 1.0
        clean = (value - np.sqrt(1 - alpha) * epsilon) / np.sqrt(alpha)
        value = np.sqrt(prior) * clean + np.sqrt(1 - prior) * epsilon
    return value


@pytest.mark.parametrize("rows", [1, 2])
def test_active_probe_cfg_order_preupdate_labels_reference_refresh_and_native_rng(rows):
    host, contract = host_contract()
    config = FalconConfig(distance_threshold=100, exploration_rate=0, buffer_size=20)
    method = Falcon(host, contract, config)
    predictor = Predictor()
    kwargs = {"z": torch.arange(1, rows + 1.0)[:, None, None]}
    if rows == 2:
        kwargs["cfg_scale"] = 1.5
    noise = torch.linspace(-1.2, 1.5, 112).reshape(1, 16, 7).expand(rows, -1, -1).clone()
    try:
        with torch.inference_mode():
            for index in range(3):
                old_entries = list(method.buffer.entries)
                old_reference = None if method.previous is None else method.previous.clone()
                contract.sampler.transitions.clear()
                predictor.calls.clear()
                state = torch.get_rng_state().clone()
                for _ in range(10):
                    torch.randn_like(noise)
                expected_rng = torch.get_rng_state().clone()
                torch.set_rng_state(state)
                kwargs["z"] = kwargs["z"] + 0.3
                result = query(method, predictor, noise, kwargs)
                assert torch.equal(torch.get_rng_state(), expected_rng)
                transitions = contract.sampler.transitions
                if index == 0:
                    assert len(predictor.calls) == 10
                    assert not host.query_diagnostics["falcon_reused"]
                    for entry, (step, pre_value) in zip(method.buffer.entries, transitions):
                        assert entry.step == step and entry.noise_level == 10 * step
                        torch.testing.assert_close(entry.latent, pre_value[0], rtol=0, atol=0)
                else:
                    assert host.query_diagnostics["falcon_reused"]
                    probe_values, probe_times, probe_z = predictor.calls[0]
                    count = len(old_entries)
                    expected_values = torch.stack([e.latent for e in old_entries])
                    for branch in range(rows):
                        torch.testing.assert_close(
                            probe_values[branch * count : (branch + 1) * count], expected_values
                        )
                        assert probe_times[branch * count : (branch + 1) * count].tolist() == [
                            e.noise_level for e in old_entries
                        ]
                        torch.testing.assert_close(
                            probe_z[branch * count : (branch + 1) * count],
                            kwargs["z"][branch : branch + 1].expand(count, 1, 1),
                        )
                    start, initial = transitions[0]
                    assert 1 < start < 9
                    assert any(
                        e.step == start and torch.equal(e.latent, initial[0]) for e in old_entries
                    )
                    assert old_reference is not None
                    assert not torch.equal(old_reference, method.previous)
                    assert host.query_diagnostics["falcon_probe_candidates"] == count
                    assert host.query_diagnostics["denoiser_calls"] == len(transitions) + 1
                    assert (
                        host.query_diagnostics["denoiser_candidate_evaluations"]
                        == len(transitions) + count
                    )
                expected = numpy_tail(
                    transitions[0][1],
                    transitions[0][0],
                    contract.alphas_cumprod,
                    kwargs["z"],
                    kwargs.get("cfg_scale", 1),
                )
                np.testing.assert_allclose(result.numpy(), expected, rtol=3e-5, atol=1e-6)
                torch.testing.assert_close(method.previous, result[0], rtol=0, atol=0)
                assert len(method.buffer.entries) <= config.buffer_size
                assert host.query_diagnostics["denoiser_evaluated_rows"] == rows * (
                    len(transitions) + len(old_entries)
                )
    finally:
        method.close()
    assert not method.buffer.entries and method.previous is None
    assert "ddim_sample_loop" not in contract.sampler.__dict__


@pytest.mark.parametrize("exploration,threshold", [(1, 100), (0, 1e-12)])
def test_gaussian_fallback_is_exact_native_and_private_selection_is_repeatable(
    exploration, threshold
):
    host, contract = host_contract()
    predictor = Predictor()
    noise = torch.linspace(-1, 1, 112).reshape(1, 16, 7)
    kwargs = {"z": torch.ones(1, 1, 1)}
    state = torch.get_rng_state().clone()
    with torch.inference_mode():
        native = contract.sampler.ddim_sample_loop(
            predictor,
            noise.shape,
            noise=noise,
            model_kwargs=kwargs,
        )
    method = Falcon(
        host,
        contract,
        FalconConfig(exploration_rate=exploration, distance_threshold=threshold),
    )
    try:
        results = []
        with torch.inference_mode():
            for _ in range(2):
                method.reset(SimpleNamespace(seed=71))
                torch.set_rng_state(state)
                run = []
                for _ in range(3):
                    output = query(method, predictor, noise, kwargs)
                    torch.testing.assert_close(output, native, rtol=0, atol=0)
                    run.append(output.clone())
                results.append((run, method.generator.get_state().clone()))
        assert torch.equal(results[0][1], results[1][1])
        assert not host.query_diagnostics["falcon_reused"]
        assert host.query_diagnostics["falcon_probe_calls"] == 1
    finally:
        method.close()


def test_probe_uses_actual_noise_alpha_and_current_unexecuted_reference():
    host, contract = host_contract()
    method = Falcon(host, contract, FalconConfig(exploration_rate=0, distance_threshold=1e-4))
    try:
        method.previous = torch.arange(112.0).reshape(16, 7) / 112
        method.generator = torch.Generator().manual_seed(91)
        for index, level in enumerate((20, 30, 40)):
            clean = torch.zeros(16, 7)
            clean[:8] = method.previous[8:] + index * 0.1
            alpha = contract.alphas_cumprod[level]
            latent = np.sqrt(alpha) * clean + np.sqrt(1 - alpha) * 0.2
            method.buffer.add(latent, 0, level // 10, level)

        def model(value, times, **kwargs):
            return torch.full_like(value, 0.2)

        with torch.inference_mode():
            selected, count, eligible, explore = method._probe(
                model,
                {"z": torch.ones(2, 1, 1), "cfg_scale": 1.5},
                torch.zeros(2, 16, 7),
            )
        assert count == 3 and eligible == 1 and not explore
        assert selected.noise_level == 20
        method.previous = method.previous + 0.1
        with torch.inference_mode():
            selected, count, eligible, explore = method._probe(
                model,
                {"z": torch.ones(2, 1, 1), "cfg_scale": 1.5},
                torch.zeros(2, 16, 7),
            )
        assert eligible == 1 and selected.noise_level == 30
    finally:
        method.close()


def test_disabled_detach_ownership_reset_and_invalid_cadence():
    host, contract = host_contract()
    disabled = Falcon(host, contract, FalconConfig(enabled=False))
    assert not disabled.requires_control_history
    assert "ddim_sample_loop" not in contract.sampler.__dict__
    disabled.close()
    disabled.close()
    method = Falcon(host, contract, FalconConfig())
    try:
        with pytest.raises(ValueError, match="Another"):
            Falcon(host, contract, FalconConfig())
        noise = torch.ones(1, 16, 7)
        with torch.inference_mode():
            query(method, Predictor(), noise, {"z": torch.ones(1, 1, 1)})
        method.emitted = 7
        with pytest.raises(ValueError, match="executed"):
            method.before_query({})
        method.reset(SimpleNamespace(seed=11))
        assert method.previous is None and not method.buffer.entries and method.seed == 11
        method.before_query({})
        with pytest.raises(RuntimeError, match="unfinished"):
            method.before_query({})
    finally:
        method.close()
    with pytest.raises(RuntimeError, match="Closed"):
        method.before_query({})
    for horizon in (1, 4, 16):
        with pytest.raises(ValueError, match="H8"):
            validate_experiment(SimpleNamespace(execution_horizon=horizon), FalconConfig())
    validate_experiment(SimpleNamespace(execution_horizon=8), FalconConfig())


@pytest.mark.parametrize(
    "changes",
    [
        {"eta": 0.1},
        {"clip_denoised": True},
        {"noise": None},
        {"progress": True},
        {"model_kwargs": {"z": torch.ones(1, 1, 1), "unknown": 1}},
    ],
)
def test_unsupported_native_options_fail_without_sampling(changes):
    host, contract = host_contract()
    method = Falcon(host, contract, FalconConfig())
    noise = torch.ones(1, 16, 7)
    try:
        method.before_query({})
        parameters = dict(noise=noise, clip_denoised=False, model_kwargs={"z": torch.ones(1, 1, 1)})
        with torch.inference_mode(), pytest.raises(ValueError, match="Unsupported"):
            contract.sampler.ddim_sample_loop(Predictor(), noise.shape, **(parameters | changes))
        assert not contract.sampler.transitions and not method.buffer.entries
    finally:
        method.close()
