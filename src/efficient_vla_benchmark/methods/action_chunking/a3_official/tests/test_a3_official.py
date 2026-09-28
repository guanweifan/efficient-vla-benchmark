from types import SimpleNamespace

import numpy as np
import pytest
import torch

from .. import decoding
from ..config import A3OfficialConfig
from ..decoding import Branch, decode, select_draft
from ..implementations.cogact.api import validate_experiment
from ..implementations.cogact.sampling import conditional_ddim, make_sampler, native_commands


@pytest.mark.parametrize(
    "bad",
    [
        {"enabled": 1},
        {"num_samples": 1},
        {"num_samples": True},
        {"num_samples": 33},
        {"action_horizon": 17},
        {"action_horizon": 1},
        {"verification_chunk_size": 3},
        {"verification_chunk_size": 0},
        {"confidence_alpha": 0},
        {"confidence_epsilon": -1},
        {"threshold_scale": float("inf")},
        {"delta_thresholds": "1"},
        {"delta_thresholds": [1] * 7},
        {"delta_thresholds": [True] * 6},
        {"unknown": 1},
    ],
)
def test_strict_config(bad):
    with pytest.raises(ValueError):
        A3OfficialConfig.parse(bad)


def test_portable_config_and_h8_reference():
    config = A3OfficialConfig.parse({"delta_thresholds": [0.5] * 6})
    assert config.delta_thresholds == (0.5,) * 6
    validate_experiment(SimpleNamespace(execution_horizon=8), config)
    with pytest.raises(ValueError):
        validate_experiment(SimpleNamespace(execution_horizon=1), config)


def test_global_confidence_selection_and_first_tie():
    actions = np.random.default_rng(18).normal(size=(8, 16, 7))
    best, confidence = select_draft(actions, 0.8, 1e-8)
    trajectory = actions[:, :, :6].cumsum(1)
    distance = ((trajectory - trajectory.mean(0)) ** 2 / (trajectory.var(0) + 1e-8)).sum(-1)
    expected = np.exp(-0.4 * distance)
    high = expected >= np.median(expected, axis=0)
    count = high.sum(1)
    candidates = np.flatnonzero(count == count.max())
    sums = (expected.astype(np.float32) * high).sum(1)
    assert best == candidates[np.argmax(sums[candidates])]
    np.testing.assert_array_equal(confidence, expected.astype(np.float32))
    assert select_draft(np.zeros((3, 4, 7)), 0.8, 1e-8)[0] == 0


def test_strict_comparison_ignores_gripper():
    raw = np.zeros((3, 4, 7), dtype=np.float32)
    execution = raw.copy()
    execution[0, 1, 6] = 99
    execution[1, 2, 0] = 0.5
    branch = Branch()
    branch.verify(np.array([1, 2, 3]), raw, execution, raw[0], np.full(6, 0.5), 4)
    assert branch.accepted == {0, 1}
    assert branch.fail_position == 2 and branch.prefix(4) == 2 and branch.rank == 1


def test_confidence_hole_does_not_append_nonprefix_rejection(monkeypatch):
    monkeypatch.setattr(
        decoding, "select_draft", lambda *args: (0, np.array([[1, 0.1, 0.9, 0.2]] * 2))
    )
    raw = np.zeros((2, 4, 7), dtype=np.float32)

    def verify(fixed, mask):
        assert mask.shape == (6, 4)
        np.testing.assert_array_equal(mask[0], [True, False, False, False])
        np.testing.assert_array_equal(mask[1], [True, False, True, False])
        result = fixed.copy()
        result[0, 2] = 2
        return result, result.copy()

    selected, horizon, info = decode(
        raw,
        raw.copy(),
        A3OfficialConfig(num_samples=2, action_horizon=4, verification_chunk_size=4),
        verify,
    )
    assert horizon == 1 and info["append_pos"] == -1
    assert info["accepted_prefix_len_conf"] == 1 and info["accepted_prefix_len_seq"] == 4
    np.testing.assert_array_equal(selected, raw[0])


def test_tied_rejections_append_temporal_action_and_never_serve_padding():
    raw = np.zeros((2, 4, 7), dtype=np.float32)

    def verify(fixed, mask):
        result = fixed.copy()
        result[0, 1] = 2
        result[3, 1] = 3
        return result, result.copy()

    selected, horizon, info = decode(
        raw,
        raw.copy(),
        A3OfficialConfig(num_samples=2, action_horizon=4, verification_chunk_size=4),
        verify,
    )
    assert horizon == 2 and info["append_pos"] == 1
    assert info["accepted_prefix_len"] == 1
    np.testing.assert_array_equal(selected[1], np.full(7, 3))
    np.testing.assert_array_equal(selected[2:], 0)


@pytest.mark.parametrize("chunk", [1, 2, 4, 8, 16])
def test_full_acceptance_parallel_tentative_masks_and_real_horizon(chunk):
    raw = np.zeros((3, 16, 7), dtype=np.float32)
    seen = []

    def verify(fixed, mask):
        seen.append(mask.copy())
        assert np.all(mask[:, 0])
        return fixed.copy(), fixed.copy()

    selected, horizon, details = decode(
        raw, raw.copy(), A3OfficialConfig(num_samples=3, verification_chunk_size=chunk), verify
    )
    assert horizon == 16 and details["append_pos"] == -1
    assert details["verification_rows"] == 30 and details["accepted_rank_seq"] == 15
    assert details["verification_batches"] == len(seen)
    np.testing.assert_array_equal(selected, raw[0])


def test_invalid_verifier_geometry_and_nonfinite():
    raw = np.zeros((2, 16, 7), dtype=np.float32)
    config = A3OfficialConfig(num_samples=2)
    with pytest.raises(ValueError):
        decode(raw, raw, config, lambda fixed, mask: (fixed[:1], fixed))
    with pytest.raises(ValueError):
        decode(raw, raw, config, lambda fixed, mask: (fixed, fixed + np.nan))
    raw[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        decode(raw, raw, config, None)


def test_native_conversion_matches_readonly_clipping_and_signed_gripper():
    raw = np.random.default_rng(7).normal(size=(3, 16, 7)).astype(np.float32)
    saved = raw.copy()
    stats = dict(q01=[-0.5] * 7, q99=[0.8] * 7, mask=[True] * 6 + [False])
    normalized = np.clip(raw, -1, 1)
    normalized[..., 6] = np.where(normalized[..., 6] < 0.5, 0, 1)
    expected = np.where(
        stats["mask"],
        0.5 * (normalized + 1) * (np.array(stats["q99"]) - np.array(stats["q01"]))
        + np.array(stats["q01"]),
        normalized,
    )
    expected[..., 6] = 1 - 2 * expected[..., 6]
    np.testing.assert_array_equal(native_commands(raw, stats), expected)
    np.testing.assert_array_equal(raw, saved)


def test_conditional_native_ddim_noop_rng_and_noisy_constraints():
    action = pytest.importorskip("action_model")
    sampler = action.create_diffusion("ddim10", diffusion_steps=100, learn_sigma=False)
    noise = torch.randn(4, 16, 7)
    fixed = torch.randn_like(noise)
    mask = torch.zeros(4, 16, dtype=torch.bool)

    def model(x, t):
        return 0.1 * x

    torch.manual_seed(53)
    native = sampler.ddim_sample_loop(
        model, noise.shape, noise=noise, clip_denoised=False, model_kwargs={}, device="cpu", eta=0
    )
    expected_rng = torch.get_rng_state()
    torch.manual_seed(53)
    result = conditional_ddim(sampler, model, noise, {}, fixed, mask)
    torch.testing.assert_close(result, native, atol=0, rtol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    mask[:, [0, 2, 8]] = True
    observed = []

    def recording_model(x, t):
        observed.append((x.clone(), t.clone()))
        return 0.1 * x

    result = conditional_ddim(sampler, recording_model, noise, {}, fixed, mask)
    torch.testing.assert_close(result[mask], fixed[mask], atol=0, rtol=0)
    for index, (x, t) in zip(reversed(range(10)), observed):
        assert torch.all(t == sampler.timestep_map[index])
        alpha = float(sampler.alphas_cumprod[index])
        expected = np.sqrt(alpha) * fixed + np.sqrt(1 - alpha) * noise
        torch.testing.assert_close(x[mask], expected[mask], rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError):
        conditional_ddim(sampler, model, noise, {}, fixed, mask[:1])


@pytest.mark.parametrize("steps", [2, 3, 4, 5, 10])
def test_native_schedule_thinning_keeps_training_noise_endpoints(steps):
    action = pytest.importorskip("action_model")
    pytest.importorskip("efficient_vla_backend_cogact")
    native = action.create_diffusion("ddim10", diffusion_steps=100, learn_sigma=False)
    full = action.create_diffusion("", diffusion_steps=100, learn_sigma=False)
    backend = SimpleNamespace(
        parameters=SimpleNamespace(num_ddim_steps=10),
        model=SimpleNamespace(
            action_model=SimpleNamespace(
                net=SimpleNamespace(training=False), ddim_diffusion=native, diffusion=full
            )
        ),
    )
    sampler = make_sampler(backend, steps)
    assert sampler.num_timesteps == steps
    assert sampler.timestep_map[0] == 0 and sampler.timestep_map[-1] == 90
    np.testing.assert_allclose(
        sampler.alphas_cumprod, full.alphas_cumprod[sampler.timestep_map], rtol=1e-14, atol=5e-16
    )
    if steps == 10:
        assert sampler is native
    if steps == 4:
        assert sampler.timestep_map == [0, 30, 60, 90]
    with pytest.raises(ValueError):
        make_sampler(backend, 11)


@pytest.mark.parametrize("steps", [True, 1, 11, 4.0])
def test_invalid_denoising_budget(steps):
    with pytest.raises(ValueError):
        A3OfficialConfig.parse({"num_denoising_steps": steps})
