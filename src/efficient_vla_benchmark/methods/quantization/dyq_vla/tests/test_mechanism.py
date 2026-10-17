import itertools
import json

import numpy as np
import pytest
import torch

from efficient_vla_benchmark.methods.quantization.dyq_vla.config import DyQConfig
from efficient_vla_benchmark.methods.quantization.dyq_vla.dispatcher import DyQDispatcher
from efficient_vla_benchmark.methods.quantization.dyq_vla.thresholds import fit_thresholds


def make():
    return DyQDispatcher(DyQConfig(), dict(theta_2_4=0.1, theta_4_8=0.3, theta_fp=0.5))


@pytest.mark.parametrize("window", [1, 3])
def test_algorithm_one_matches_exhaustive_target_sequences(window):
    for sequence in itertools.product((2, 4, 8, 16), repeat=5):
        d = DyQDispatcher(
            DyQConfig(hysteresis_window=window),
            dict(theta_2_4=0.1, theta_4_8=0.3, theta_fp=0.5),
        )
        active, counter, maximum = 16, 0, 16
        for target in sequence:
            if target >= active:
                active, counter, maximum = target, 0, target
            else:
                previous = maximum
                maximum = max(target, maximum * int(counter > 0))
                counter = counter * int(maximum == previous) + 1
                active = maximum if counter == window else active
                counter %= window
            assert d.dispatch_target(target) == active
            assert (d.counter, d.pending_max) == (counter, maximum)


@pytest.mark.parametrize("theta_fp,budget", [(0.65, 0.02), (0.75, 0.02), (0.9, 0.2)])
def test_routing_parameters_do_not_change_calibration_observation_sensitivity(theta_fp, budget):
    original = make()
    candidate = DyQDispatcher(
        DyQConfig(hysteresis_window=1, theta_fp=theta_fp, accuracy_budget=budget),
        dict(theta_2_4=0.1, theta_4_8=0.3, theta_fp=theta_fp),
    )
    actions = np.random.default_rng(7).normal(size=(100, 7))
    for action in actions:
        original.observe(action)
        candidate.observe(action)
        assert original.last_sensitivity == candidate.last_sensitivity
        assert original.motion == candidate.motion and original.jerk == candidate.jerk


def test_pending_upgrade_restarts_downgrade_delay():
    d = make()
    assert [d.dispatch_target(b) for b in (2, 2, 4, 4, 4)] == [16, 16, 16, 16, 4]
    assert d.dispatch_target(8) == 8
    d.reset()
    assert d.active_bits == 16 and not d.motion


def test_only_fused_sensitivity_is_clamped_and_no_future_commands():
    d = make()
    assert d.target() == 16
    for v in (1.0, 1.0, 1.0, 1.0, 10.0):
        d.observe(np.array([v, 0, 0, 0, 0, 0, -1]))
    assert d.motion[-1] < 0
    assert d.last_sensitivity == 0
    other = make()
    assert not other.motion
    with pytest.raises(ValueError):
        d.observe(np.full(7, np.nan))


def test_threshold_fit_stops_at_first_crossing():
    cfg = DyQConfig.parse(dict(sensitivity_bins=3, accuracy_budget=0.02))
    records = []
    for s, e in ((0.1, 0.01), (0.2, 1.0), (0.4, 0.001)):
        records += [dict(sensitivity=s, errors={"2": e, "4": 0.001, "8": 0.0001})] * 4
    thresholds, bins = fit_thresholds(records, cfg)
    assert thresholds["theta_2_4"] == 0.1
    assert thresholds["theta_4_8"] == 0.4 and len(bins) == 3
    records = [dict(sensitivity=0.1, errors={"2": 10.0, "4": 10.0, "8": 0.0001})] * 4
    thresholds, _ = fit_thresholds(records, cfg)
    assert thresholds["theta_2_4"] == thresholds["theta_4_8"] == -1


def test_configuration_bounds():
    for values in (
        {"motion_window": True},
        {"eta": 0},
        {"accuracy_budget": float("nan")},
        {"frames_per_trajectory": 1000},
        {"unexpected": 1},
        {"runtime": "fake"},
    ):
        with pytest.raises(ValueError):
            DyQConfig.parse(values)


def test_explicit_runtime_preserves_legacy_identity_and_requires_new_calibration(
    tmp_path, monkeypatch
):
    from efficient_vla_benchmark.core.provenance import digest_file, stable_hash
    from efficient_vla_benchmark.methods.quantization.dyq_vla.artifact import read_manifest
    from efficient_vla_benchmark.model_ops.quantization import mixed_runtime

    legacy = DyQConfig()
    mixed = DyQConfig.parse({"runtime": "mixed_integer"})
    assert "runtime" not in legacy.calibration_parameters()
    assert mixed.calibration_parameters()["runtime"] == "mixed_integer"
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"fixture only")
    samples = [{"flat": i, "step": 0} for i in range(4)]
    identity = {"engine": "checked test runtime", "files": {"binary": "hash"}}
    monkeypatch.setattr(mixed_runtime, "read_runtime", lambda _: identity)
    manifest = dict(
        schema_version=3,
        algorithm="dyq_vla",
        complete=True,
        config=mixed.calibration_parameters(),
        files={"weights.pt": digest_file(weights)},
        population_kind="independent_native_quantization_population",
        sample_identities=samples,
        sample_identities_sha256=stable_hash(samples),
        error_records=[{} for _ in samples],
        execution_runtime=identity,
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert read_manifest(tmp_path, mixed, tmp_path) == manifest
    with pytest.raises(ValueError, match="not admitted"):
        read_manifest(tmp_path, legacy)
    manifest["execution_runtime"] = {"engine": "other runtime"}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exact calibrated"):
        read_manifest(tmp_path, mixed, tmp_path)
    manifest.update(schema_version=2, config=legacy.calibration_parameters())
    manifest.pop("execution_runtime")
    path.write_text(json.dumps(manifest))
    assert read_manifest(tmp_path, legacy) == manifest
    with pytest.raises(ValueError, match="not admitted"):
        read_manifest(tmp_path, mixed, tmp_path)


def test_mixed_runtime_rejects_non_native_boundary_before_conversion():
    from efficient_vla_benchmark.methods.quantization.dyq_vla.runtime import make_linear

    config = DyQConfig.parse({"runtime": "mixed_integer"})
    for entry in (
        {"native_dtype": "torch.float32", "bias": None},
        {"native_dtype": "torch.bfloat16", "bias": torch.ones(8)},
    ):
        with pytest.raises(ValueError, match="bias-free BF16"):
            make_linear(entry, config, None)


@pytest.mark.gpu
@torch.inference_mode()
def test_dynamic_packed_calls_match_native_dtype_dense_oracle_and_reset():
    from efficient_vla_benchmark.methods.quantization.dyq_vla.runtime import DynamicW4Linear

    torch.manual_seed(17)
    q = torch.randint(-8, 8, (64, 128), dtype=torch.int8)
    s = torch.rand(64) * 0.01 + 0.001
    a = torch.rand(128) * 2 + 1
    e = dict(codes=q, weight_scale=s, activation_amax=a, bias=None)
    op = DynamicW4Linear(e).cuda()
    x = torch.randn(33, 128, device="cuda", dtype=torch.bfloat16)
    w = q.to(x) * s.to(x)[:, None]
    for bits in (16, 2, 8, 4, 16):
        op.set_active_bits(bits)
        refx = x
        if bits < 16:
            maximum = (1 << (bits - 1)) - 1
            step = (a.to(x) / maximum).clamp_min(1e-8)
            refx = (x / step).round().clamp(-maximum - 1, maximum) * step
        torch.testing.assert_close(op(x), torch.nn.functional.linear(refx, w), atol=0.01, rtol=0.01)
    assert op.kernel_invocations == 5
    assert op.packed_weight.numel() == 64 * 128 // 2
