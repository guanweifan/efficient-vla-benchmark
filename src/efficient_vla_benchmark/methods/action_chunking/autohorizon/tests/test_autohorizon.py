from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from ..config import AutoHorizonConfig

torch = pytest.importorskip("torch")

from ..selection import select_prefix  # noqa: E402


@pytest.mark.parametrize(
    "bad",
    [
        {"enabled": 1},
        {"denoising_step": 0},
        {"run_length": True},
        {"max_horizon": 1.5},
        {"hold_threshold": 0},
        {"hold_threshold": float("nan")},
        {"entropy_quantile": -1},
        {"entropy_quantile": 1.1},
        {"unknown": 3},
    ],
)
def test_invalid_config(bad):
    with pytest.raises(ValueError):
        AutoHorizonConfig.parse(bad)


def numpy_reference(attention, config):
    def pointer(a):
        a = a / (a.sum(-1, keepdims=True) + 1e-12)
        mu = np.maximum.accumulate((a * np.arange(len(a))).sum(-1))
        clipped = a.clip(1e-12)
        entropy = -(clipped * np.log(clipped)).sum(-1) / np.log(len(a))
        reliable = entropy <= np.quantile(entropy, min(config.entropy_quantile, 0.999))
        holds = (np.diff(mu, prepend=0) < config.hold_threshold) & reliable
        matches = [
            i
            for i in range(len(a) - config.run_length + 1)
            if holds[i : i + config.run_length].all()
        ]
        return mu, matches[0] if matches else len(a) - 1

    a = attention / (attention.sum(-1, keepdims=True) + 1e-12)
    forward, stop_f = pointer(a)
    reverse, stop_r = pointer(a[::-1, ::-1])
    backward = (len(a) - 1 - reverse)[::-1]
    n_f = int(np.clip(np.floor(forward[stop_f]) + 1, 1, len(a)))
    n_b = int(np.clip(np.floor(backward[len(a) - 1 - stop_r]) + 1, 1, len(a)))
    joined = ((backward - forward) <= 1).any()
    return min(config.max_horizon, len(a) if n_f + n_b >= len(a) and joined else n_f)


@pytest.mark.parametrize("hold", [0.1, 0.3, 0.5])
@pytest.mark.parametrize("run", [1, 2, 3])
def test_bidirectional_selector_independent_reference(hold, run):
    rng = np.random.default_rng(23)
    config = AutoHorizonConfig(hold_threshold=hold, run_length=run)
    outputs = []
    for _ in range(40):
        matrix = rng.lognormal(0, 3, (16, 16))
        horizon, _ = select_prefix(torch.tensor(matrix), config)
        assert horizon == numpy_reference(matrix, config)
        outputs.append(horizon)
    assert len(set(outputs)) > 1  # Active selector, not a fixed-horizon proxy.


def test_invalid_attention_and_real_prefix_cap():
    for bad in [torch.zeros(16, 16), torch.ones(3, 4), torch.full((16, 16), float("nan"))]:
        with pytest.raises(ValueError):
            select_prefix(bad, AutoHorizonConfig())
    value, _ = select_prefix(torch.ones(16, 16), AutoHorizonConfig(max_horizon=8))
    assert 1 <= value <= 8


@pytest.mark.parametrize("qk_norm", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_attention_probe_preserves_forward_full_key_normalization_and_cfg(qk_norm, fused):
    pytest.importorskip("timm")
    from timm.models.vision_transformer import Attention

    from ..implementations.cogact.method import action_attention

    torch.manual_seed(22)
    attn = Attention(32, num_heads=4, qkv_bias=True, qk_norm=qk_norm).eval()
    attn.fused_attn = fused
    x = torch.randn(2, 17, 32)
    captured = []
    with torch.inference_mode():
        expected = attn(x)
        handle = attn.qkv.register_forward_hook(
            lambda module, inputs, output: captured.append(action_attention(output, attn))
        )
        actual = attn(x)
        handle.remove()
        packed = attn.qkv(x).reshape(2, 17, 3, 4, 8).permute(2, 0, 3, 1, 4)
        q, k, _ = packed.unbind(0)
        q, k = attn.q_norm(q), attn.k_norm(k)
        full = ((q * attn.scale) @ k.transpose(-1, -2)).softmax(-1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(captured[0], full[0, :, 1:, 1:].mean(0), rtol=0, atol=0)
    assert not torch.allclose(captured[0], full[:, :, 1:, 1:].mean((0, 1)))
    assert (captured[0].sum(-1) < 1).all()  # Conditioning key still owns probability mass.


def test_only_selected_denoising_call_and_cleanup():
    pytest.importorskip("timm")
    from timm.models.vision_transformer import Attention

    from ..implementations.cogact.api import build

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attention(32, num_heads=4)

        def forward(self, x):
            return x + self.attn(x)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block(), Block()])

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x[:, 1:, :7]

    net = Net().eval()
    host = SimpleNamespace(
        model_access=SimpleNamespace(action_denoiser=net),
        parameters=SimpleNamespace(num_ddim_steps=10),
        query_diagnostics={},
    )
    x = torch.randn(2, 17, 32)
    with torch.inference_mode():
        native = net(x)
        method = build(host, AutoHorizonConfig(), None)
        for _ in range(2):
            method.reset(None)
            method.before_query({})
            for call in range(10):
                result = net(x)
                torch.testing.assert_close(result, native, rtol=0, atol=0)
                assert len(method.maps) == (0 if call < 2 else 2)
            method.after_query(result[0])
            assert method.maps == {}
            assert 1 <= method.select_execution_horizon(result[0], 8) <= 16
        method.close()
        method.close()
        torch.testing.assert_close(net(x), native, rtol=0, atol=0)
    assert "forward" not in net.__dict__
    assert all(not block.attn.qkv._forward_hooks for block in net.blocks)
    with pytest.raises(RuntimeError):
        method.before_query({})
    for config in [
        replace(AutoHorizonConfig(), denoising_step=11),
        replace(AutoHorizonConfig(), max_horizon=17),
    ]:
        with pytest.raises(ValueError):
            build(host, config, None)


@pytest.mark.gpu
def test_real_model_exact_prediction_rng_control_reset_and_detach(model_path, trace_path, request):
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from efficient_vla_benchmark.control.chunk import ChunkPolicy
    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..implementations.cogact.api import build

    host = CogACTBackend(model_path)
    request.addfinalizer(host.close)
    context, frames, _ = load_trace(trace_path, require_control_steps=True)
    host.reset(context)
    native = [host.predict_chunk(frame) for frame in frames[:3]]
    native_rng = torch.cuda.get_rng_state().clone()
    for enabled in (False, True):
        method = build(host, AutoHorizonConfig(enabled=enabled), None)
        host.attach_method(method)
        for _ in range(2):
            host.reset(context)
            for i, frame in enumerate(frames[:3]):
                actions = host.predict_chunk(frame)
                np.testing.assert_array_equal(actions, native[i])
                horizon = host.select_execution_horizon(actions.copy(), 8)
                assert 1 <= horizon <= 16
            assert torch.equal(native_rng, torch.cuda.get_rng_state())
        if enabled:
            policy = ChunkPolicy(host, 8)
            policy.reset(context)
            decisions = [policy.step(frame) for frame in frames[:32]]
            assert sum(d.queried for d in decisions) == policy.query_count
            assert all(1 <= d.diagnostics["selected_execution_horizon"] <= 16 for d in decisions)
        host.detach_method()
        assert "forward" not in host.model_access.action_denoiser.__dict__
        assert all(not b.attn.qkv._forward_hooks for b in host.model_access.action_denoiser.blocks)
    host.reset(context)
    np.testing.assert_array_equal(host.predict_chunk(frames[0]), native[0])
