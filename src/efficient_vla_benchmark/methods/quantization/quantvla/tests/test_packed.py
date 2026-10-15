from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from efficient_vla_benchmark.methods.quantization.quantvla.packed import (
    BlockRotation,
    PackedDuQuantLinear,
)


def test_block_rotation_rejects_geometry():
    with pytest.raises(ValueError):
        BlockRotation([torch.ones(4, 3)])


@pytest.mark.gpu
@torch.inference_mode()
def test_identity_input_operations_and_precast_rotations_are_exact():
    torch.manual_seed(113)
    k, n = 128, 64
    t = SimpleNamespace(
        smoothing=torch.ones(k, device="cuda"),
        permutation=torch.arange(k, device="cuda"),
        input_rotations=[torch.linalg.qr(torch.randn(32, 32, device="cuda")).Q for _ in range(4)],
        output_rotations=[torch.linalg.qr(torch.randn(32, 32, device="cuda")).Q for _ in range(2)],
        row_restore=True,
    )
    op = PackedDuQuantLinear(
        torch.randint(-8, 8, (n, k), device="cuda", dtype=torch.int8),
        torch.rand(n, device="cuda") * 0.01 + 0.001,
        torch.ones(k, device="cuda"),
        None,
        t,
    ).cuda()
    x = torch.randn(33, k, device="cuda", dtype=torch.bfloat16)
    expected = op(x)
    op.input_rotation.bfloat16()
    op.output_rotation.bfloat16()
    assert op.identity_smoothing and op.identity_permutation
    torch.testing.assert_close(op(x), expected, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_duquant_complete_transform_and_packed_equivalence(dtype):
    torch.manual_seed(71)
    device = "cuda"
    k, n, b = 128, 64, 32
    rotations = [torch.linalg.qr(torch.randn(b, b, device=device)).Q for _ in range(k // b)]
    output_rotations = [torch.linalg.qr(torch.randn(b, b, device=device)).Q for _ in range(n // b)]
    smoothing = torch.rand(k, device=device) + 0.5
    permutation = torch.randperm(k, device=device)
    t = SimpleNamespace(
        smoothing=smoothing,
        permutation=permutation,
        input_rotations=rotations,
        output_rotations=output_rotations,
        row_restore=True,
    )
    q = torch.randint(-8, 8, (n, k), dtype=torch.int8, device=device)
    ws = torch.rand(n, device=device) * 0.02 + 0.001
    amax = torch.rand(k, device=device) * 2 + 1
    bias = torch.randn(n, device=device, dtype=dtype) * 0.01
    op = PackedDuQuantLinear(q, ws, amax, bias, t).cuda()
    x = torch.randn(34, k, device=device, dtype=dtype)
    ref = (x * smoothing.to(dtype)).index_select(-1, permutation)
    ref = torch.cat(
        [ref[:, i * b : (i + 1) * b] @ r.to(dtype) for i, r in enumerate(rotations)], -1
    )
    step = (amax.to(dtype) / 127).clamp_min(1e-8)
    ref = (ref / step).round().clamp(-127, 127) * step
    weight = q.to(dtype) * ws.to(dtype)[:, None]
    ref = F.linear(ref, weight)
    ref = (
        torch.cat(
            [ref[:, i * b : (i + 1) * b] @ r.to(dtype) for i, r in enumerate(output_rotations)], -1
        )
        + bias
    )
    actual = op(x)
    tol = 0.02 if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(actual, ref, atol=tol, rtol=tol)
    assert op.linear.kernel_invocations == 1
    assert op.linear.packed_weight.numel() == n * k // 2


@pytest.mark.gpu
@torch.inference_mode()
def test_full_precision_transform_identity():
    torch.manual_seed(19)
    x = torch.randn(17, 128, device="cuda")
    w = torch.randn(64, 128, device="cuda")
    s = torch.rand(128, device="cuda") + 0.5
    p = torch.randperm(128, device="cuda")
    rin = [torch.linalg.qr(torch.randn(32, 32, device="cuda")).Q for _ in range(4)]
    rout = [torch.linalg.qr(torch.randn(32, 32, device="cuda")).Q for _ in range(2)]
    xr = BlockRotation(rin)((x * s).index_select(-1, p))
    wr = BlockRotation(rin)((w / s).index_select(1, p))
    wr = torch.cat([r @ wr[i * 32 : (i + 1) * 32] for i, r in enumerate(rout)])
    out = BlockRotation(rout)(F.linear(xr, wr))
    torch.testing.assert_close(out, F.linear(x, w), atol=3e-5, rtol=3e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("frozen_step,row_restore", [(True, True), (False, False)])
@torch.inference_mode()
def test_fp16_engine_matches_explicit_unfused_complete_transform(dtype, frozen_step, row_restore):
    torch.manual_seed(817)
    k = n = 256
    b = 64
    rotations = [
        torch.linalg.qr(torch.randn(b, b, device="cuda")).Q.to(dtype) for _ in range(k // b)
    ]
    outputs = [torch.linalg.qr(torch.randn(b, b, device="cuda")).Q.to(dtype) for _ in range(n // b)]
    transform = SimpleNamespace(
        smoothing=torch.rand(k, device="cuda") + 0.5,
        permutation=torch.randperm(k, device="cuda"),
        input_rotations=rotations,
        output_rotations=outputs,
        row_restore=row_restore,
    )
    codes = torch.randint(-8, 8, (n, k), dtype=torch.int8, device="cuda")
    weight_scale = torch.rand(n, device="cuda") * 0.02 + 0.003
    step = torch.rand(k, device="cuda") * 0.02 + 0.005
    bias = torch.randn(n, device="cuda", dtype=dtype) * 0.01
    op = PackedDuQuantLinear(
        codes, weight_scale, step, bias, transform, frozen_step=frozen_step, runtime="tinychat_fp16"
    )
    input_reference, output_reference = BlockRotation(rotations), BlockRotation(outputs)
    for m in (1, 16, 17, 34, 279, 280):
        x = torch.randn(1, m, k, device="cuda", dtype=dtype)
        rotated = input_reference(
            (x * transform.smoothing.to(dtype)).index_select(-1, transform.permutation)
        )
        grid = step.to(dtype) if frozen_step else (step.to(dtype) / 127).clamp_min(1e-8)
        quant = (rotated / grid).round().clamp(-128 if frozen_step else -127, 127).to(torch.int8)
        prepared = (quant.to(dtype) * grid).half()
        expected = op.linear.kernel(prepared).to(dtype)
        if row_restore:
            expected = output_reference(expected)
        expected = expected + bias
        torch.testing.assert_close(op(x), expected, atol=0, rtol=0)
    saved = {key: value.clone() for key, value in op.state_dict().items()}
    expected = op(x)
    op.load_state_dict(saved, strict=True)
    torch.testing.assert_close(op(x), expected, atol=0, rtol=0)
    assert op(x[:, :0]).shape == (1, 0, n)
    assert op.linear.execution_metadata()["activation_scale_axis"] == "input_channel"
