from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from efficient_vla_benchmark.methods.quantization.holoq.config import HoloQConfig
from efficient_vla_benchmark.methods.quantization.holoq.gptq import mse_scales, quantize
from efficient_vla_benchmark.methods.quantization.holoq.rotation import (
    build,
    rotate,
    transformed_weight,
)


def test_config_rejects_unknowns_and_invalid_geometry():
    for values in (
        {"unknown": 1},
        {"block_size": 63},
        {"calibration_samples": 5},
        {"activation_percentile": float("nan")},
        {"gptq_block_size": True},
    ):
        with pytest.raises(ValueError):
            HoloQConfig.parse(values)


@torch.inference_mode()
def test_composite_transform_preserves_full_precision_map():
    torch.manual_seed(91)
    x, w = torch.randn(19, 64), torch.randn(32, 64)
    for row_restore in (False, True):
        t = build(w, 16, row_restore)
        xr = rotate(x[:, t["permutation"]], t["input"])
        wr = transformed_weight(w, t)
        y = F.linear(xr, wr)
        if row_restore:
            y = rotate(y, t["output"])
        torch.testing.assert_close(y, F.linear(x, w), atol=3e-5, rtol=3e-5)


@torch.inference_mode()
def test_gptq_retains_each_block_scale_and_column_error_propagation():
    torch.manual_seed(33)
    w, x = torch.randn(17, 64), torch.randn(256, 64)
    w[:, 32:] *= 19
    codes, scales = quantize(w, x, 32, 0.01)
    result = codes.float() * scales.repeat_interleave(32, 1)
    oracle = []
    for start in (0, 32):
        block = w[:, start : start + 32].clone()
        h = x[:, start : start + 32].T @ x[:, start : start + 32] / len(x)
        h = (h + h.T) / 2
        h.diagonal().add_(0.01 * h.diagonal().mean())
        inv = torch.cholesky_inverse(torch.linalg.cholesky(h))
        s = mse_scales(block)
        columns = []
        for c in range(32):
            q = (block[:, c] / s).round().clamp(-8, 7) * s
            error = (block[:, c] - q) / inv[c, c]
            columns.append(q)
            if c + 1 < 32:
                block[:, c + 1 :] -= error[:, None] * inv[c, c + 1 :][None, :]
        oracle.append(torch.stack(columns, 1))
    torch.testing.assert_close(result, torch.cat(oracle, 1), atol=0, rtol=0)
    assert scales.shape == (17, 2)
    assert not torch.equal(scales[:, 0], scales[:, 1])


def test_step_trackers_are_instance_local_and_detachable():
    from efficient_vla_benchmark.methods.quantization.holoq.implementations.cogact.step import (
        DDIMStepTracker,
    )

    def fake():
        return SimpleNamespace(
            ddim_diffusion=SimpleNamespace(
                timestep_map=[0, 10], original_num_steps=100, betas=[0.01, 0.02]
            ),
            noise_schedule="test",
            diffusion_steps=100,
            net=SimpleNamespace(t_embedder=torch.nn.Identity()),
        )

    a, b = fake(), fake()
    first, second = DDIMStepTracker(a, 2), DDIMStepTracker(b, 2)
    a.net.t_embedder(torch.tensor([10]))
    assert first.current_step == 0 and second.current_step is None
    b.net.t_embedder(torch.tensor([0]))
    assert first.current_step == 0 and second.current_step == 1
    first.reset()
    assert first.current_step is None and second.current_step == 1
    first.close()
    first.close()
    assert not a.net.t_embedder._forward_pre_hooks
    second.close()


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_packed_composite_a4_matches_dense_oracle(dtype):
    from efficient_vla_benchmark.methods.quantization.holoq.packed import HoloQLinear

    torch.manual_seed(12)
    w = torch.randn(64, 128, device="cuda") * 0.02
    t = build(w, 32, True)
    q = torch.randint(-8, 8, w.shape, dtype=torch.int8)
    s = torch.rand(64, 4) * 0.003 + 0.001
    scale_table = torch.rand(2, 128) * 0.1 + 0.02
    e = dict(
        codes=q,
        weight_scales=s,
        weight_group_size=32,
        transform=t,
        native_dtype=str(dtype),
        activation_steps=scale_table,
        bias=torch.randn(64) * 0.01,
        side="dit_attention",
    )
    row = [0]
    op = HoloQLinear(e, lambda: row[0]).cuda()
    x = torch.randn(33, 128, device="cuda", dtype=dtype)
    for idx in range(2):
        row[0] = idx
        xr = rotate(x[:, t["permutation"].cuda()], t["input"].cuda())
        step = scale_table[idx].to(x)
        xr = (xr / step).round().clamp(-8, 7) * step
        wr = (q.float() * s.repeat_interleave(32, 1)).to(x)
        ref = rotate(F.linear(xr, wr), t["output"].cuda()) + e["bias"].to(x)
        actual = op(x)
        tol = 0.01 if dtype == torch.bfloat16 else 2e-5
        torch.testing.assert_close(actual, ref, atol=tol, rtol=tol)
    assert op.linear.kernel_invocations == 2
    assert op.linear.packed_weight.numel() == w.numel() // 2
    row[0] = None
    with pytest.raises(ValueError):
        op(x)


def test_invalid_composite_transform_is_rejected():
    from efficient_vla_benchmark.methods.quantization.holoq.rotation import validate_transform

    t = build(torch.randn(32, 64), 16, True)
    validate_transform(t, 64, 32, True)
    t["permutation"][0] = t["permutation"][1]
    with pytest.raises(ValueError):
        validate_transform(t, 64, 32, True)
    t = build(torch.randn(32, 64), 16, True)
    t["input"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        validate_transform(t, 64, 32, True)
