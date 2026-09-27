from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ..config import KeyStoneConfig
from ..implementations.cogact.method import KeyStone
from ..selection import select_index


def reference(points, clusters, tau):
    """Independent NumPy oracle following the released deterministic selector."""
    points = points.astype(np.float32)
    distances = np.linalg.norm(points[:, None] - points[None], axis=-1)
    index = distances.sum(1).argmin()
    upper = np.sort(distances[np.triu_indices(len(points), 1)])
    median = upper[(len(upper) - 1) // 2]  # torch.median uses the lower median
    spread = np.linalg.norm(points.mean(0) - points[index]) / (median + 1e-8)
    if spread <= tau:
        return index, False
    centers = points[:clusters].copy()
    for _ in range(10):
        labels = np.linalg.norm(points[:, None] - centers[None], axis=-1).argmin(1)
        means = np.stack(
            [
                points[labels == c].mean(0) if np.any(labels == c) else centers[c]
                for c in range(clusters)
            ]
        )
        if np.allclose(centers, means, atol=1e-6, rtol=1e-5):
            break
        centers = means
    labels = np.linalg.norm(points[:, None] - centers[None], axis=-1).argmin(1)
    members = np.flatnonzero(labels == np.bincount(labels, minlength=clusters).argmax())
    return members[distances[np.ix_(members, members)].sum(1).argmin()], True


@pytest.mark.parametrize("k", [3, 4, 8, 16])
@pytest.mark.parametrize("tau", [0.0, 0.3, 100.0])
def test_selection_reference_real_sample_and_prefix(k, tau):
    cfg = KeyStoneConfig(num_samples=k, unimodal_tau=tau)
    for seed in range(8):
        x = torch.tensor(np.random.default_rng(seed).normal(size=(k, 16, 7)), dtype=torch.float32)
        actual, clustered = select_index(x, cfg)
        expected, expected_clustered = reference(x[:, :8].flatten(1).numpy(), 2, tau)
        assert int(actual) == expected
        assert clustered == expected_clustered
        x[:, 8:] *= 1000
        assert int(select_index(x, cfg)[0]) == expected


def test_identical_candidates_nonfinite_and_config():
    cfg = KeyStoneConfig()
    assert int(select_index(torch.zeros(4, 16, 7), cfg)[0]) == 0
    for x in (torch.zeros(4, 7, 7), torch.zeros(3, 16, 7), torch.full((4, 16, 7), float("nan"))):
        with pytest.raises(ValueError):
            select_index(x, cfg)
    for bad in (
        {"num_samples": 1},
        {"num_samples": 2},
        {"num_samples": True},
        {"num_clusters": 4},
        {"num_clusters": 1},
        {"unimodal_tau": float("nan")},
        {"enabled": 1},
        {"unimodal_tau": True},
        {"unimodal_tau": -1},
        {"selection_horizon": 0},
        {"unknown": 1},
    ):
        with pytest.raises(ValueError):
            KeyStoneConfig.parse(bad)


class SpySampler:
    num_timesteps = 10

    def ddim_sample_loop(self, model, shape, noise=None, model_kwargs=None, **kwargs):
        self.input = noise.clone()
        self.z = model_kwargs["z"].clone()
        self.output = noise * 0.25 + model_kwargs["z"][:, :, :1]
        return self.output


def fake_host():
    sampler = SpySampler()
    net = SimpleNamespace(training=False)
    action_model = SimpleNamespace(ddim_diffusion=sampler, net=net)
    return SimpleNamespace(
        model=SimpleNamespace(action_model=action_model),
        parameters=SimpleNamespace(num_ddim_steps=10),
        query_diagnostics={},
    ), sampler


@pytest.mark.parametrize("cfg", [False, True])
def test_sampler_pairing_one_context_and_detach(cfg):
    host, sampler = fake_host()
    method = KeyStone(host, KeyStoneConfig())
    rows = 2 if cfg else 1
    z = torch.zeros(rows, 1, 4096)
    z[0] = 0.2
    if cfg:
        z[1] = -0.4
    kw = {"z": z}
    if cfg:
        kw["cfg_scale"] = 1.5
    with torch.inference_mode():
        outcomes = []
        for _ in range(2):
            method.reset(None)
            torch.manual_seed(7)
            noise = torch.randn(1, 16, 7).expand(rows, -1, -1)
            method.before_query({})
            result = sampler.ddim_sample_loop(
                None, noise.shape, noise=noise, model_kwargs=kw, clip_denoised=False
            )
            method.after_query(result)
            assert result.shape == noise.shape
            assert sampler.input.shape == (4 * rows, 16, 7)
            assert not torch.equal(sampler.input[0], sampler.input[1])
            assert torch.equal(sampler.z[:4], z[:1].expand(4, -1, -1))
            index = host.query_diagnostics["action_selection_index"]
            assert torch.equal(result[0], sampler.output[index])
            if cfg:
                assert torch.equal(sampler.input[:4], sampler.input[4:])
                assert torch.equal(sampler.z[4:], z[1:2].expand(4, -1, -1))
                assert torch.equal(result[1], sampler.output[4 + index])
            outcomes.append(result.clone())
        torch.testing.assert_close(*outcomes, rtol=0, atol=0)
    method.close()
    method.close()
    assert "ddim_sample_loop" not in sampler.__dict__
    assert not hasattr(sampler, "_evb_keystone")
    with pytest.raises(RuntimeError):
        method.before_query({})


def test_disabled_and_unsupported_options():
    host, sampler = fake_host()
    method = KeyStone(host, KeyStoneConfig(enabled=False))
    assert not method.patches.saved
    method.close()
    method = KeyStone(host, KeyStoneConfig())
    noise = torch.zeros(1, 16, 7)
    kw = {"z": torch.zeros(1, 1, 4096)}
    with pytest.raises(RuntimeError):
        sampler.ddim_sample_loop(None, noise.shape, noise=noise, model_kwargs=kw)
    method.before_query({})
    with torch.inference_mode(), pytest.raises(ValueError, match="nonnative"):
        sampler.ddim_sample_loop(None, noise.shape, noise=noise, model_kwargs=kw, eta=1)
    method.reset(None)
    method.close()


def test_native_dit_batch_parity():
    pytest.importorskip("efficient_vla_backend_cogact")
    from action_model.models import DiT

    torch.manual_seed(9)
    net = DiT(hidden_size=24, depth=2, num_heads=3, future_action_window_size=15).eval()
    with torch.no_grad():
        net.final_layer.linear.weight.normal_(std=0.05)
    x, z = torch.randn(4, 16, 7), torch.randn(1, 1, 4096)
    uncond = net.z_embedder.uncondition.unsqueeze(0).expand(4, 1, -1)
    with torch.inference_mode():
        joined = net.forward_with_cfg(
            torch.cat((x, x)),
            torch.full((8,), 50),
            torch.cat((z.expand(4, -1, -1), uncond)),
            1.5,
        )[:4]
        separate = torch.cat(
            [
                net.forward_with_cfg(
                    torch.cat((x[i : i + 1], x[i : i + 1])),
                    torch.full((2,), 50),
                    torch.cat((z, uncond[:1])),
                    1.5,
                )[:1]
                for i in range(4)
            ]
        )
    torch.testing.assert_close(joined, separate, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_real_native_reset_disabled_detach_and_shared_vlm(model_path, trace_path, request):
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    host = CogACTBackend(model_path)
    request.addfinalizer(host.close)
    context, frames, _ = load_trace(trace_path, require_control_steps=True)
    host.reset(context)
    native = host.predict_chunk(frames[0])
    host.attach_method(KeyStone(host, KeyStoneConfig(enabled=False)))
    host.reset(context)
    np.testing.assert_array_equal(native, host.predict_chunk(frames[0]))
    host.detach_method()
    calls = []
    hook = host.model.vlm.register_forward_hook(lambda *args: calls.append(1))
    request.addfinalizer(hook.remove)
    host.attach_method(KeyStone(host, KeyStoneConfig()))
    results = []
    for _ in range(2):
        host.reset(context)
        calls.clear()
        results.append(host.predict_chunk(frames[0]))
        assert len(calls) == 1
        assert results[-1].shape == (16, 7)
        assert np.isfinite(results[-1]).all()
        assert host.query_diagnostics["action_candidates"] == 4
        assert host.query_diagnostics["action_head_batch"] == 8
        assert host.query_diagnostics["denoiser_calls"] == 10
    np.testing.assert_array_equal(*results)
    host.detach_method()
    host.reset(context)
    np.testing.assert_array_equal(native, host.predict_chunk(frames[0]))
