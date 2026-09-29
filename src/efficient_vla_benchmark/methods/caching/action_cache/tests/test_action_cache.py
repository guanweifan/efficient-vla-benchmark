import json
from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ..config import ActionCacheConfig
from ..projection import projection


def test_projection_exact_balance_and_rng_isolation():
    before = torch.get_rng_state().clone()
    matrix = projection(4096, 50, 0.01, 42, "cpu")
    assert torch.equal(before, torch.get_rng_state())
    assert torch.all((matrix == 1).sum(1) == 20)
    assert torch.all((matrix == -1).sum(1) == 20)
    assert torch.equal(matrix, projection(4096, 50, 0.01, 42, "cpu"))
    for bad in (
        {"hit_threshold": 2},
        {"refinement_steps": -1},
        {"projection_seed": True},
        {"projection_density": 0},
    ):
        with pytest.raises(ValueError):
            ActionCacheConfig.parse(bad)


def test_native_capture_frozen_hit_miss_tail_rng_and_detach(tmp_path):
    pytest.importorskip("efficient_vla_backend_cogact")
    from action_model import create_diffusion
    from action_model.models import DiT

    from efficient_vla_benchmark.core.provenance import digest_file

    from ..implementations.cogact.capture import NativeActionCapture
    from ..implementations.cogact.method import ActionCache

    @dataclass
    class Parameters:
        num_ddim_steps: int = 10
        cfg_scale: float = 1.5

    torch.manual_seed(5)
    model = DiT(hidden_size=24, depth=2, num_heads=3, future_action_window_size=15).eval()
    with torch.no_grad():
        model.final_layer.linear.weight.normal_(std=0.1)
    sampler = create_diffusion(
        timestep_respacing="ddim10",
        noise_schedule="squaredcos_cap_v2",
        diffusion_steps=100,
        sigma_small=True,
        learn_sigma=False,
    )
    host = SimpleNamespace(
        model=SimpleNamespace(action_model=SimpleNamespace(ddim_diffusion=sampler)),
        parameters=Parameters(),
        device="cpu",
        query_diagnostics={},
        metadata=lambda: {"checkpoint_sha256": "test-checkpoint"},
    )
    config = ActionCacheConfig(hit_threshold=0.99)
    capture = NativeActionCapture(host, config)
    noise, z = torch.randn(2, 16, 7), torch.randn(2, 1, 4096)
    noise[1] = noise[0]
    kwargs = {"z": z, "cfg_scale": 1.5}
    with torch.inference_mode():
        capture.before_query({})
        rng = torch.get_rng_state().clone()
        native = sampler.ddim_sample_loop(
            model.forward_with_cfg,
            noise.shape,
            noise=noise,
            clip_denoised=False,
            model_kwargs=kwargs,
            device="cpu",
            eta=0,
        )
        expected_rng = torch.get_rng_state().clone()
        capture.after_query(native)
        key, states, clean = capture.pending[0]
        torch.testing.assert_close(clean, native[0], rtol=0, atol=0)
        capture.close()
        np.savez(
            tmp_path / "bank.npz",
            keys=key.numpy()[None],
            states=states.numpy()[None],
            clean_actions=clean.numpy()[None],
            suites=np.array([0]),
        )
        manifest = dict(
            kind="action_cache_cogact_native_success_bank",
            schema_version=2,
            clean_action_source="native_final_ddim_sample",
            key_dim=config.key_dim,
            projection_density=config.projection_density,
            projection_seed=config.projection_seed,
            bank_sha256=digest_file(tmp_path / "bank.npz"),
            checkpoint_sha256="test-checkpoint",
            sampler_parameters=asdict(host.parameters),
            episodes_disjoint_from_evaluation=True,
            suites=["libero_goal"],
            population_sha256="test-population",
        )
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        for settings, hit in (
            (config, True),
            (replace(config, refinement_steps=0), True),
            (replace(config, refinement_steps=1), True),
            (replace(config, refinement_steps=0, hit_threshold=1.0), False),
            (replace(config, hit_threshold=1.0), False),
            (replace(config, enabled=False), False),
        ):
            method = ActionCache(host, settings, tmp_path)
            method.reset(SimpleNamespace(suite="libero_goal"))
            method.before_query({})
            torch.set_rng_state(rng)
            calls = []

            def denoiser(*args, **kwargs):
                calls.append(1)
                return model.forward_with_cfg(*args, **kwargs)

            actual = sampler.ddim_sample_loop(
                denoiser,
                noise.shape,
                noise=noise,
                clip_denoised=False,
                model_kwargs=kwargs,
                device="cpu",
                eta=0,
            )
            method.after_query(actual)
            torch.testing.assert_close(actual, native, rtol=0, atol=0)
            assert len(calls) == (settings.refinement_steps if settings.enabled and hit else 10)
            assert torch.equal(expected_rng, torch.get_rng_state())
            if settings.enabled:
                assert host.query_diagnostics["action_cache_hit"] == hit
                assert host.query_diagnostics["denoiser_evaluations"] == (
                    settings.refinement_steps if hit else 10
                )
            method.close()
            method.close()
            assert "ddim_sample_loop" not in sampler.__dict__
        assert digest_file(tmp_path / "bank.npz") == manifest["bank_sha256"]
        # Old banks stay usable for few-step refinement, never for zero steps.
        manifest.pop("schema_version")
        manifest.pop("clean_action_source")
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="captured native clean"):
            ActionCache(host, replace(config, refinement_steps=0), tmp_path)
        ActionCache(host, config, tmp_path).close()


@pytest.mark.gpu
def test_real_frozen_bank_reset_and_disabled_parity(model_path, trace_path, artifact_path, request):
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..implementations.cogact.method import ActionCache

    host = CogACTBackend(model_path)
    request.addfinalizer(host.close)
    context, frames, _ = load_trace(trace_path, require_control_steps=True)
    host.reset(context)
    native = host.predict_chunk(frames[0])
    for cfg in (ActionCacheConfig(enabled=False), ActionCacheConfig(hit_threshold=1.0)):
        host.attach_method(ActionCache(host, cfg, artifact_path))
        host.reset(context)
        np.testing.assert_array_equal(native, host.predict_chunk(frames[0]))
        host.detach_method()
    host.attach_method(ActionCache(host, ActionCacheConfig(hit_threshold=-1.0), artifact_path))
    outputs = []
    for _ in range(2):
        host.reset(context)
        outputs.append(host.predict_chunk(frames[0]))
        assert np.isfinite(outputs[-1]).all()
        assert host.query_diagnostics["action_cache_hit"]
        assert host.query_diagnostics["denoiser_evaluations"] == 2
    np.testing.assert_array_equal(*outputs)
    host.detach_method()
    host.reset(context)
    np.testing.assert_array_equal(native, host.predict_chunk(frames[0]))


@pytest.mark.gpu
def test_native_clean_capture_zero_tail_and_rng(model_path, trace_path, tmp_path, request):
    """Ephemeral equation fixture only; never a deployable population artifact."""
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from efficient_vla_benchmark.core.provenance import digest_file
    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..implementations.cogact.capture import NativeActionCapture
    from ..implementations.cogact.method import ActionCache

    host = CogACTBackend(model_path)
    request.addfinalizer(host.close)
    context, frames, _ = load_trace(trace_path, require_control_steps=True)
    host.reset(context)
    native = host.predict_chunk(frames[0])
    expected_rng = torch.cuda.get_rng_state().clone()
    cfg = ActionCacheConfig(hit_threshold=-1)
    capture = NativeActionCapture(host, cfg)
    host.attach_method(capture)
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(frames[0]), native)
    assert torch.equal(torch.cuda.get_rng_state(), expected_rng)
    key, states, clean = capture.pending[0]
    assert clean.shape == (16, 7) and bool(torch.isfinite(clean).all())
    host.detach_method()
    np.savez_compressed(
        tmp_path / "bank.npz",
        keys=key.numpy()[None],
        states=states.numpy()[None],
        clean_actions=clean.numpy()[None],
        suites=np.array([0]),
    )
    manifest = dict(
        kind="action_cache_cogact_native_success_bank",
        schema_version=2,
        clean_action_source="native_final_ddim_sample",
        key_dim=cfg.key_dim,
        projection_density=cfg.projection_density,
        projection_seed=cfg.projection_seed,
        bank_sha256=digest_file(tmp_path / "bank.npz"),
        checkpoint_sha256=host.metadata()["checkpoint_sha256"],
        sampler_parameters=asdict(host.parameters),
        episodes_disjoint_from_evaluation=True,
        suites=[context.suite],
        population_sha256="EPHEMERAL-TEST-FIXTURE-NOT-FOR-EVALUATION",
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    for refinement in (0, 1, 2):
        method = ActionCache(host, replace(cfg, refinement_steps=refinement), tmp_path)
        host.attach_method(method)
        for _ in range(2):
            host.reset(context)
            np.testing.assert_array_equal(host.predict_chunk(frames[0]), native)
            assert host.query_diagnostics["action_cache_hit"]
            assert host.query_diagnostics["denoiser_evaluations"] == refinement
            assert torch.equal(torch.cuda.get_rng_state(), expected_rng)
        host.detach_method()
        method.close()
        host.reset(context)
        np.testing.assert_array_equal(host.predict_chunk(frames[0]), native)
    assert digest_file(tmp_path / "bank.npz") == manifest["bank_sha256"]
