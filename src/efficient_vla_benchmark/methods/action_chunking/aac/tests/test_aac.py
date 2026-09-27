from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ..config import AACConfig
from ..implementations.cogact.method import AAC, native_commands
from ..selection import movement_profile, select_prefix, step_entropy


@pytest.mark.parametrize(
    "bad",
    [
        {"enabled": 1},
        {"num_samples": 1},
        {"num_samples": 3},
        {"num_samples": True},
        {"num_samples": 41},
        {"movement_threshold": -1},
        {"movement_threshold": float("nan")},
        {"covariance_regularization": 0},
        {"gripper_movement_weight": True},
        {"unknown": 1},
    ],
)
def test_strict_config(bad):
    with pytest.raises(ValueError):
        AACConfig.parse(bad)


def stats():
    return dict(q01=[-1.0] * 7, q99=[1.0] * 7, mask=[True] * 6 + [False])


def test_entropy_gaussian_covariance_and_gripper_complement():
    x = np.random.default_rng(4).normal(size=(20, 16, 7))
    g = (x[:, :, 6] > 0.5).astype(float)
    actual = step_entropy(x, g, 1e-6)
    expected = []
    for t in range(16):
        pose = sum(
            0.5
            * (
                3 * np.log(2 * np.pi * np.e)
                + np.linalg.slogdet(np.cov(x[:, t, offset : offset + 3].T) + np.eye(3) * 1e-6)[1]
            )
            for offset in (0, 3)
        )
        probability = np.clip(g[:, t].mean(), 1e-9, 1 - 1e-9)
        expected.append(
            pose - probability * np.log(probability) - (1 - probability) * np.log(1 - probability)
        )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual, step_entropy(x, 1 - g, 1e-6), rtol=1e-12, atol=1e-12)


def test_native_conversion_keeps_input_and_binary_convention():
    x = np.random.default_rng(3).normal(size=(5, 16, 7)).astype(np.float32)
    original = x.copy()
    converted = native_commands(x, stats())
    np.testing.assert_array_equal(x, original)
    expected = []
    for chunk in x:
        normalized = np.clip(chunk, -1, 1)
        normalized[:, 6] = np.where(normalized[:, 6] < 0.5, 0, 1)
        statistics = stats()
        high, low = np.array(statistics["q99"]), np.array(statistics["q01"])
        native = np.where(
            statistics["mask"],
            0.5 * (normalized + 1) * (high - low) + low,
            normalized,
        )
        expected.append(native[:, :6])
    np.testing.assert_array_equal(converted[:, :, :6], np.stack(expected))
    np.testing.assert_array_equal(converted[:, :, 6], 1 - (np.clip(x[:, :, 6], -1, 1) >= 0.5))
    bad = stats()
    bad["mask"][6] = True
    with pytest.raises(ValueError):
        native_commands(x, bad)


def test_movement_ordered_rotation_not_vector_sum():
    actions = np.zeros((4, 7))
    actions[:, 0] = 0.25
    actions[0, 3] = np.pi / 2
    actions[1, 4] = np.pi / 2
    values = movement_profile(actions, 0.2)
    assert values[0] == pytest.approx(0.5 + 2 * np.pi / 3, abs=5e-7)
    assert abs(values[0] - (0.5 + np.linalg.norm(actions[:2, 3:6].sum(0)))) > 0.1
    actions[1, 6] = 1
    assert movement_profile(actions, 0.2)[0] == pytest.approx(values[0] + 0.2)
    assert np.all(movement_profile(np.zeros((16, 7)), 0.2) == 0)


def test_movement_lower_bound_entropy_elbow_and_real_horizon():
    x = np.random.default_rng(8).normal(size=(5, 16, 7))
    commands = native_commands(x, stats())
    horizon, details = select_prefix(x, commands, AACConfig(num_samples=5))
    assert horizon == max(details.values())
    assert 2 <= horizon <= 16
    horizon, details = select_prefix(x, commands, AACConfig(num_samples=5, movement_threshold=1e6))
    assert horizon == details["movement_horizon"] == 16
    with pytest.raises(ValueError):
        select_prefix(x[:1], commands[:1], AACConfig(num_samples=5))
    with pytest.raises(ValueError):
        step_entropy(np.full((5, 16, 7), np.nan), np.zeros((5, 16)), 1e-6)


class SpySampler:
    num_timesteps = 10

    def ddim_sample_loop(self, model, shape, noise=None, model_kwargs=None, **kwargs):
        self.input = noise.clone()
        self.condition = model_kwargs["z"].clone()
        self.output = noise * 0.25 + model_kwargs["z"][:, :, :1]
        return self.output


def fake_host():
    sampler = SpySampler()
    model = SimpleNamespace(
        action_model=SimpleNamespace(ddim_diffusion=sampler, net=SimpleNamespace(training=False)),
        get_action_stats=lambda key: stats(),
    )
    return SimpleNamespace(
        model=model,
        parameters=SimpleNamespace(num_ddim_steps=10),
        unnorm_key="suite",
        query_diagnostics={},
    ), sampler


@pytest.mark.parametrize("using_cfg", [False, True])
def test_one_context_candidate_zero_pairing_reset_and_detach(using_cfg):
    host, sampler = fake_host()
    method = AAC(host, AACConfig(num_samples=5))
    rows = 2 if using_cfg else 1
    condition = torch.zeros(rows, 1, 16)
    condition[0] = 0.3
    if using_cfg:
        condition[1] = -0.1
    kwargs = dict(z=condition, cfg_scale=1.5 if using_cfg else 1)
    outcomes = []
    for _ in range(2):
        method.reset(None)
        with pytest.raises(RuntimeError):
            method.select_execution_horizon(np.zeros((16, 7)), 8)
        with torch.inference_mode():
            torch.manual_seed(7)
            noise = torch.randn(1, 16, 7).expand(rows, -1, -1)
            method.before_query({})
            result = sampler.ddim_sample_loop(
                None, noise.shape, noise=noise, clip_denoised=False, model_kwargs=kwargs
            )
            method.after_query(result)
            assert sampler.input.shape == (rows * 5, 16, 7)
            assert not torch.equal(sampler.input[0], sampler.input[1])
            torch.testing.assert_close(result[0], sampler.output[0], rtol=0, atol=0)
            if using_cfg:
                torch.testing.assert_close(sampler.input[:5], sampler.input[5:], rtol=0, atol=0)
                torch.testing.assert_close(result[1], sampler.output[5], rtol=0, atol=0)
                torch.testing.assert_close(sampler.condition[5:], condition[1:2].expand(5, -1, -1))
            assert 2 <= method.select_execution_horizon(np.zeros((16, 7)), 8) <= 16
            outcomes.append(result.clone())
            assert host.query_diagnostics["shared_context_evaluations"] == 1
            assert host.query_diagnostics["denoiser_candidate_evaluations"] == 50
    torch.testing.assert_close(*outcomes, rtol=0, atol=0)
    method.close()
    method.close()
    assert "ddim_sample_loop" not in sampler.__dict__
    assert not hasattr(sampler, "_evb_aac")
    with pytest.raises(RuntimeError):
        method.before_query({})


def test_disabled_and_invalid_lifecycle():
    host, sampler = fake_host()
    disabled = AAC(host, AACConfig(enabled=False))
    assert not disabled.patches.saved
    assert disabled.select_execution_horizon(np.zeros((16, 7)), 8) == 8
    disabled.close()
    method = AAC(host, AACConfig(num_samples=5))
    with pytest.raises(RuntimeError):
        method.after_query(np.zeros((16, 7)))
    method.reset(None)
    with torch.inference_mode():
        method.before_query({})
        with pytest.raises(ValueError):
            sampler.ddim_sample_loop(
                None,
                (2, 16, 7),
                noise=torch.zeros(2, 16, 7),
                clip_denoised=False,
                eta=1,
                model_kwargs=dict(z=torch.zeros(2, 1, 16)),
            )
    method.close()
