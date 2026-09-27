import copy
import itertools
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from efficient_vla_benchmark.core.provenance import digest_file, stable_hash

from ..artifact import read_manifest
from ..config import BACConfig
from ..implementations.cogact.method import BAC
from ..schedule import make_schedule, optimal_updates


def objective(matrix, updates):
    ends = [*updates[1:], len(matrix)]
    return sum(matrix[a, a:b].sum() for a, b in zip(updates, ends))


@pytest.mark.parametrize("length", [3, 5, 7])
def test_acs_exhaustive_global_optimality(length):
    random = np.random.default_rng(31 + length)
    x = random.normal(size=(length, 8))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    matrix = x @ x.T
    for budget in range(1, length + 1):
        updates = optimal_updates(matrix, budget)
        possibilities = [(0, *v) for v in itertools.combinations(range(1, length), budget - 1)]
        best = max(objective(matrix, option) for option in possibilities)
        assert objective(matrix, updates) == pytest.approx(best)
        assert len(updates) == budget and updates[0] == 0
    with pytest.raises(ValueError):
        optimal_updates(matrix, length + 1)


def test_bubbling_union_every_downstream_ffn_and_last_refresh():
    random = np.random.default_rng(21)
    matrices = random.random((8, 10, 10))
    errors = np.asarray([9, 4, 8, 3, 7, 2, 6, 1], dtype=float)
    for count in (1, 2, 5, 8):
        schedules, protected = make_schedule(matrices, errors, 3, count)
        assert protected == tuple(np.argsort(-errors, kind="stable")[:count])
        for upstream in protected:
            for downstream in range(upstream + 1, len(schedules)):
                if downstream % 2:
                    assert set(schedules[downstream]) <= set(schedules[upstream])
        assert all(0 in steps and 9 in steps for steps in schedules)
    dense, _ = make_schedule(matrices, errors, 10, 5)
    assert all(steps == tuple(range(10)) for steps in dense)


def artifact(root, net, config, parameters, checkpoint="fixture"):
    count = 2 * len(net.blocks)
    steps = parameters.num_ddim_steps
    root.mkdir(exist_ok=True)
    np.savez(
        root / "statistics.npz",
        similarity=np.repeat(np.eye(steps)[None], count, axis=0),
        l1_error=np.arange(count, dtype=float),
    )
    samples = [{"split": "train", "episode_index": i} for i in range(config.calibration_samples)]
    manifest = dict(
        schema_version=1,
        algorithm="bac",
        backend="cogact",
        complete=True,
        calibration_parameters=config.calibration_parameters(),
        samples=samples,
        samples_sha256=stable_hash(samples),
        checkpoint_sha256=checkpoint,
        sampler_parameters=asdict(parameters),
        precision=str(next(net.parameters()).dtype),
        num_blocks=count,
        num_steps=steps,
        statistics_sha256=digest_file(root / "statistics.npz"),
    )
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def fixture(tmp_path):
    pytest.importorskip("efficient_vla_backend_cogact")
    from action_model.models import DiT
    from efficient_vla_backend_cogact.config import CogACTParameters

    torch.manual_seed(5)
    net = DiT(hidden_size=24, depth=2, num_heads=3, future_action_window_size=3).eval()
    with torch.no_grad():
        net.final_layer.linear.weight.normal_(std=0.1)
    config = BACConfig(num_updates=2, protected_blocks=2, calibration_samples=4)
    parameters = CogACTParameters(num_ddim_steps=4)
    host = SimpleNamespace(
        model_access=SimpleNamespace(action_denoiser=net),
        parameters=parameters,
        _model_manifest={"sha256": "fixture"},
        query_diagnostics={},
    )
    artifact(tmp_path, net, config, parameters)
    return net, host, config


def test_native_full_update_disabled_and_rng_parity(tmp_path):
    net, host, cfg = fixture(tmp_path)
    reference = copy.deepcopy(net)
    x, z = torch.randn(2, 4, 7), torch.randn(2, 1, 4096)
    for config in (replace(cfg, num_updates=4), replace(cfg, enabled=False)):
        method = BAC(host, config, tmp_path)
        with torch.inference_mode():
            rng = torch.get_rng_state().clone()
            method.before_query({})
            for index in range(4):
                t = torch.full((2,), 90 - index * 20)
                expected = reference.forward_with_cfg(x, t, z, 1.5)
                result = net.forward_with_cfg(x, t, z, 1.5)
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
            method.after_query(result)
            assert torch.equal(torch.get_rng_state(), rng)
            assert not method.saved
            if config.enabled:
                assert host.query_diagnostics["bac_branch_evaluations"] == 16
                assert host.query_diagnostics["bac_branch_reuses"] == 0
        method.close()
        method.close()
        assert "forward" not in net.__dict__
        assert all(
            "forward" not in module.__dict__
            for block in net.blocks
            for module in (block.attn, block.mlp)
        )


def test_active_cache_counts_reset_detach_and_no_cross_query_state(tmp_path):
    net, host, config = fixture(tmp_path)
    reference = copy.deepcopy(net)
    method = BAC(host, replace(config, num_updates=1), tmp_path)
    x, z = torch.randn(2, 4, 7), torch.randn(2, 1, 4096)
    with torch.inference_mode():
        runs = []
        for _ in range(2):
            method.before_query({})
            results = []
            for index in range(4):
                t = torch.full((2,), 90 - index * 20)
                results.append(net.forward_with_cfg(x, t, z, 1.5))
            method.after_query(results[-1])
            assert host.query_diagnostics["bac_branch_reuses"] == 8
            assert host.query_diagnostics["bac_branch_evaluations"] == 8
            assert not method.saved
            runs.append(torch.stack(results))
        torch.testing.assert_close(*runs, rtol=0, atol=0)
        assert not torch.allclose(
            runs[0][1], reference.forward_with_cfg(x, torch.full((2,), 70), z, 1.5)
        )
    method.close()
    with torch.inference_mode():
        torch.testing.assert_close(
            net.forward_with_cfg(x, t, z, 1.5),
            reference.forward_with_cfg(x, t, z, 1.5),
            rtol=0,
            atol=0,
        )


def test_artifact_validation_and_unsupported_lifecycle(tmp_path):
    net, host, config = fixture(tmp_path)
    with pytest.raises(ValueError, match="checkpoint"):
        other = copy.copy(host)
        other._model_manifest = {"sha256": "other"}
        BAC(other, config, tmp_path)
    method = BAC(host, config, tmp_path)
    with pytest.raises(RuntimeError, match="nonnested"):
        net(torch.zeros(2, 4, 7), torch.ones(2), torch.zeros(2, 1, 4096))
    method.before_query({})
    with pytest.raises(RuntimeError, match="nonnested"):
        net(torch.zeros(2, 4, 7), torch.ones(2), torch.zeros(2, 1, 4096))
    method.close()
    (tmp_path / "statistics.npz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="corrupt"):
        read_manifest(tmp_path, config)


def test_config_rejects_invalid_parameters():
    for bad in (
        {"num_updates": 0},
        {"protected_blocks": 0},
        {"enabled": 1},
        {"num_updates": True},
        {"calibration_samples": 5},
        {"calibration_seed": -1},
        {"extra": 1},
    ):
        with pytest.raises(ValueError):
            BACConfig.parse(bad)


@pytest.mark.gpu
def test_real_native_dense_active_reset_and_detach(model_path, trace_path, artifact_path, request):
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    host = CogACTBackend(model_path)
    request.addfinalizer(host.close)
    context, frames, _ = load_trace(trace_path, require_control_steps=True)
    host.reset(context)
    native = host.predict_chunk(frames[0])
    for config in (BACConfig(enabled=False), BACConfig(num_updates=10)):
        host.attach_method(BAC(host, config, artifact_path))
        host.reset(context)
        np.testing.assert_array_equal(native, host.predict_chunk(frames[0]))
        host.detach_method()
    host.attach_method(BAC(host, BACConfig(num_updates=4), artifact_path))
    results = []
    for _ in range(2):
        host.reset(context)
        results.append(host.predict_chunk(frames[0]))
        assert host.query_diagnostics["bac_branch_reuses"] > 0
        assert host.query_diagnostics["denoiser_calls"] == 10
        assert np.isfinite(results[-1]).all()
    np.testing.assert_array_equal(*results)
    host.detach_method()
    host.reset(context)
    np.testing.assert_array_equal(native, host.predict_chunk(frames[0]))
