"""Polynomial/calibration references and native-stack lifecycle, not smoke scores."""

import copy
import json
from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from efficient_vla_benchmark.core.provenance import digest_file, stable_hash

from ..artifact import read_manifest
from ..config import TeaCacheConfig
from ..implementations.cogact.calibration import fit_polynomial, relative_changes
from ..implementations.cogact.method import TeaCache
from ..schedule import refresh_schedule


def test_schedule_against_online_polynomial_recurrence():
    rng = np.random.default_rng(312)
    for _ in range(20):
        differences = np.r_[0, rng.uniform(0.01, 0.9, 9)]
        coefficients = rng.uniform(-0.3, 0.4, 5)
        for threshold in (0, 0.15, 0.5, 10):
            for warmup in (1, 2, 10):
                expected, error = [], 0
                for step, difference in enumerate(differences):
                    if step < warmup or threshold == 0:
                        fresh = True
                    else:
                        error += np.polyval(coefficients, difference)
                        fresh = error >= threshold
                    if fresh:
                        error = 0
                    expected.append(fresh)
                assert refresh_schedule(
                    differences.tolist(), coefficients.tolist(), threshold, warmup
                ) == tuple(expected)
    # Released variant neither clamps a negative fit nor forces the final step.
    assert refresh_schedule([0, 0.2, 0.3], [-1, 0], 0.01, 1) == (True, False, False)
    assert refresh_schedule([0, 0.2], [1, 0], 0.2, 1) == (True, True)


def test_relative_l1_and_polynomial_fit_use_every_training_sample():
    features = [torch.tensor([[1.0, 3.0]]), torch.tensor([[2.0, 6.0]]), torch.tensor([[1.0, 3.0]])]
    np.testing.assert_array_equal(relative_changes(features), [0, 1, 0.5])
    x = np.r_[0, np.linspace(0.1, 1, 9)]
    expected = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
    y = np.tile(np.polyval(expected, x), (4, 1))
    y[:, 0] = 0
    np.testing.assert_allclose(fit_polynomial(x, y, 4), expected, rtol=1e-10, atol=1e-10)
    with pytest.raises(ValueError, match="distinct"):
        fit_polynomial(np.zeros(5), np.zeros((4, 5)), 4)
    with pytest.raises(ValueError, match="Zero norm"):
        relative_changes([features[0] * 0, features[1]])


class TimeEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 8)

    def forward(self, t):
        return self.linear(t.float()[:, None] * 0.01)


class ConditionEmbedder(nn.Linear):
    def forward(self, z, training):
        assert not training
        return super().forward(z)


class ToyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_embedder = nn.Linear(3, 8)
        self.t_embedder = TimeEmbedder()
        self.z_embedder = ConditionEmbedder(5, 8)
        self.positional_embedding = nn.Parameter(torch.randn(5, 8))
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(8, 8), nn.Tanh()) for _ in range(3)])
        self.final_layer = nn.Linear(8, 6)

    def forward(self, x, t, z):
        hidden = self.x_embedder(x)
        condition = self.t_embedder(t)[:, None] + self.z_embedder(z, self.training)
        hidden = torch.cat((condition, hidden), 1) + self.positional_embedding
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_layer(hidden)[:, 1:]


@dataclass(frozen=True)
class Parameters:
    num_ddim_steps: int = 8
    cfg_scale: float = 1.5


def fixture(root):
    torch.manual_seed(122)
    net = ToyDiT().eval()
    host = SimpleNamespace(
        model_access=SimpleNamespace(action_denoiser=net),
        parameters=Parameters(),
        _model_manifest={"sha256": "fixture"},
        query_diagnostics={},
    )
    config = TeaCacheConfig(threshold=0.15, calibration_samples=4)
    x = np.array([0, 0.02, 0.04, 0.07, 0.06, 0.05, 0.2, 0.4])
    np.savez(
        root / "statistics.npz",
        input_changes=x,
        residual_changes=np.tile(x, (4, 1)),
        coefficients=np.array([0.0, 0.0, 0.0, 1.0, 0.0]),
    )
    samples = [dict(split="train", episode_index=i) for i in range(4)]
    manifest = dict(
        schema_version=1,
        algorithm="teacache",
        backend="cogact",
        complete=True,
        signal="timestep_embedding",
        calibration_parameters=config.calibration_parameters(),
        samples=samples,
        samples_sha256=stable_hash(samples),
        checkpoint_sha256="fixture",
        sampler_parameters=asdict(host.parameters),
        precision=str(next(net.parameters()).dtype),
        num_blocks=3,
        num_steps=8,
        cfg_rows=2,
        timesteps=list(range(70, -1, -10)),
        statistics_sha256=digest_file(root / "statistics.npz"),
    )
    (root / "manifest.json").write_text(json.dumps(manifest))
    return net, host, config, manifest


@torch.inference_mode()
def test_active_residual_matches_independent_stack_and_clears(tmp_path):
    net, host, config, _ = fixture(tmp_path)
    native = copy.deepcopy(net)
    method = TeaCache(host, config, tmp_path)
    x, z = torch.randn(2, 4, 3), torch.randn(2, 1, 5)
    runs = []
    try:
        for _ in range(2):
            method.before_query({})
            saved, run = None, []
            for step, noise in enumerate(range(70, -1, -10)):
                current = x + step * 0.1
                t = torch.full((2,), noise)
                condition = native.t_embedder(t)[:, None] + native.z_embedder(z, False)
                initial = (
                    torch.cat((condition, native.x_embedder(current)), 1)
                    + native.positional_embedding
                )
                hidden = initial
                if step in (0, 1, 4, 6, 7):
                    for block in native.blocks:
                        hidden = block(hidden)
                    saved = hidden - initial
                else:
                    hidden = initial + saved
                expected = native.final_layer(hidden)[:, 1:]
                actual = net(current, t, z)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                if step == 0:
                    torch.testing.assert_close(actual, native(current, t, z), rtol=0, atol=0)
                run.append(actual)
            method.after_query(None)
            assert host.query_diagnostics["teacache_reused_steps"] == 3
            assert method.saved is None and not method.query_open
            runs.append(torch.stack(run))
        torch.testing.assert_close(*runs, rtol=0, atol=0)
    finally:
        method.close()
    method.close()
    assert "forward" not in net.__dict__ and not hasattr(net, "_evb_feature_cache")
    torch.testing.assert_close(net(x, t, z), native(x, t, z), rtol=0, atol=0)


@torch.inference_mode()
@pytest.mark.parametrize("options", [{"enabled": False}, {"threshold": 0}, {"warmup_steps": 8}])
def test_noop_native_parity_without_wrappers(tmp_path, options):
    net, host, config, _ = fixture(tmp_path)
    native = copy.deepcopy(net)
    method = TeaCache(host, replace(config, **options), tmp_path)
    assert "forward" not in net.__dict__
    x, t, z = torch.randn(2, 4, 3), torch.ones(2), torch.randn(2, 1, 5)
    method.before_query({})
    torch.testing.assert_close(net(x, t, z), native(x, t, z), rtol=0, atol=0)
    method.after_query(None)
    method.close()


def test_corrupt_artifact_and_checkpoint_rejected(tmp_path):
    _, host, config, manifest = fixture(tmp_path)
    host._model_manifest = {"sha256": "other"}
    with pytest.raises(ValueError, match="checkpoint"):
        TeaCache(host, config, tmp_path)
    manifest["samples"][0]["split"] = "validation"
    manifest["samples_sha256"] = stable_hash(manifest["samples"])
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="calibration"):
        read_manifest(tmp_path, config)
    fixture(tmp_path)
    (tmp_path / "statistics.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="calibration"):
        read_manifest(tmp_path, config)


def test_noise_grid_gradients_partial_query_and_second_owner_rejected(tmp_path):
    net, host, config, _ = fixture(tmp_path)
    method = TeaCache(host, config, tmp_path)
    x, t, z = torch.ones(2, 4, 3), torch.full((2,), 70), torch.ones(2, 1, 5)
    try:
        with pytest.raises(ValueError, match="Only one"):
            TeaCache(host, config, tmp_path)
        method.before_query({})
        with pytest.raises(RuntimeError, match="inference-only"):
            net(x, t, z)
        assert not method.query_open
        method.before_query({})
        with torch.inference_mode():
            net(x, t, z)
            with pytest.raises(RuntimeError, match="noise grid"):
                net(x, t, z)
        assert method.saved is None and not method.query_open
        method.before_query({})
        with pytest.raises(RuntimeError, match="Incomplete"):
            method.after_query(None)
    finally:
        method.close()


@pytest.mark.parametrize(
    "parameters",
    [
        {"enabled": 1},
        {"threshold": True},
        {"threshold": -1},
        {"threshold": float("nan")},
        {"warmup_steps": 0},
        {"polynomial_degree": 5},
        {"calibration_samples": 6},
        {"calibration_seed": -1},
        {"unrecognized": 1},
    ],
)
def test_invalid_config(parameters):
    with pytest.raises(ValueError):
        TeaCacheConfig.parse(parameters)


@pytest.mark.gpu
def test_native_gpu_parity_active_cache_rng_reset_detach(
    model_path, trace_path, artifact_path, request
):
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    host = CogACTBackend(model_path)
    request.addfinalizer(host.close)
    context, observations, _ = load_trace(trace_path)
    host.reset(context)
    native = host.predict_chunk(observations[0])
    rng = torch.cuda.get_rng_state().clone()
    for options in ({"enabled": False}, {"threshold": 0}, {"warmup_steps": 10}):
        host.attach_method(TeaCache(host, TeaCacheConfig(**options), artifact_path))
        host.reset(context)
        np.testing.assert_array_equal(host.predict_chunk(observations[0]), native)
        host.detach_method()
    host.attach_method(TeaCache(host, TeaCacheConfig(threshold=100), artifact_path))
    host.reset(context)
    actual = host.predict_chunk(observations[0])
    assert actual.shape == native.shape and np.isfinite(actual).all()
    assert host.query_diagnostics["teacache_reused_steps"] > 0
    assert host.query_diagnostics["denoiser_calls"] == 10
    assert torch.equal(rng, torch.cuda.get_rng_state())
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(observations[0]), actual)
    host.detach_method()
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(observations[0]), native)
