import json

import pytest
import torch

from efficient_vla_benchmark.core.provenance import digest_file
from efficient_vla_benchmark.model_ops.quantization.mixed_linear import MixedPrecisionW4Linear
from efficient_vla_benchmark.model_ops.quantization.mixed_runtime import (
    ABI_VERSION,
    CUTLASS_COMMIT,
    CUTLASS_HEADERS_SHA256,
    MixedW4Runtime,
    kernel_source,
    read_runtime,
)


def test_runtime_identity_rejects_tampering_before_loading(tmp_path):
    for name in ("mixed_w4.so", "CUTLASS_LICENSE.txt"):
        (tmp_path / name).write_text("fixture only; never loaded")
    manifest = dict(
        schema_version=1,
        abi_version=ABI_VERSION,
        engine="mixed_w4_cutlass_sm80",
        cutlass_commit=CUTLASS_COMMIT,
        cutlass_headers_sha256=CUTLASS_HEADERS_SHA256,
        kernel_source_sha256=digest_file(kernel_source()),
        files={
            name: digest_file(tmp_path / name) for name in ("mixed_w4.so", "CUTLASS_LICENSE.txt")
        },
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert read_runtime(tmp_path) == manifest
    (tmp_path / "mixed_w4.so").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        read_runtime(tmp_path)
    manifest["abi_version"] = 100
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="version"):
        read_runtime(tmp_path)
    with pytest.raises(ValueError, match="explicit"):
        read_runtime(None)


def test_rejects_bad_geometry_scales_and_missing_runtime():
    q, s = torch.zeros(24, 32, dtype=torch.int8), torch.ones(24)
    for codes, scales in (
        (q[:23], s[:23]),
        (q[:, :31], s),
        (q, s * 0),
        (q, s * float("nan")),
        (q, s * float("inf")),
    ):
        with pytest.raises(ValueError):
            MixedPrecisionW4Linear(codes, scales, runtime=None)
    with pytest.raises(ValueError, match="validated"):
        MixedPrecisionW4Linear(q, s, runtime=None)


def integer_reference(x, q, scales, bits):
    high = 2 ** (bits - 1) - 1
    xs = x.float().abs().amax(1).clamp_min(1e-5) / high
    xq = (x.float() / xs[:, None]).round().clamp(-high, high).to(torch.int8)
    padding = max(32, ((len(x) + 7) // 8) * 8) - len(x)
    sums = torch._int_mm(torch.nn.functional.pad(xq, (0, 0, 0, padding)), q.T.contiguous())[
        : len(x)
    ]
    return (sums.float() * xs[:, None] * scales.bfloat16().float()[None, :]).bfloat16()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "m,n,k",
    [(m, 64, 128) for m in (1, 7, 8, 16, 17, 32, 33, 34, 63, 64, 280, 511, 512, 513, 593)]
    + [(17, 24, 32), (280, 4096, 4096), (593, 11008, 4096), (280, 4096, 11008)],
)
@torch.inference_mode()
def test_four_precisions_reference_reload_stream_and_single_bank(runtime_path, m, n, k):
    torch.manual_seed(17203)
    runtime = MixedW4Runtime(runtime_path)
    q = torch.randint(-8, 8, (n, k), device="cuda", dtype=torch.int8)
    scales = torch.rand(n, device="cuda") * 0.0004 + 0.0001
    op = MixedPrecisionW4Linear(q, scales, runtime=runtime).cuda()
    x = torch.randn(m, k * 2, device="cuda", dtype=torch.bfloat16)[:, ::2]
    x[0] = 0
    state = {name: tensor.clone() for name, tensor in op.state_dict().items()}
    assert set(state) == {"packed_weight", "weight_scale"}
    assert op.packed_weight.numel() == n * k // 2
    pointer = op.packed_weight.data_ptr()
    for bits in (16, 2, 4, 8, 2, 16):
        op.set_active_bits(bits)
        actual = op(x)
        if bits == 16:
            reference = torch.nn.functional.linear(x, q.bfloat16() * scales.bfloat16()[:, None])
            torch.testing.assert_close(actual, reference, atol=0.0005, rtol=0.01)
        else:
            reference = integer_reference(x, q, scales, bits)
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        torch.testing.assert_close(op(x), actual, atol=0, rtol=0)
        assert op.packed_weight.data_ptr() == pointer
        assert all(torch.equal(v, op.state_dict()[name]) for name, v in state.items())
    assert op.calls == {2: 4, 4: 2, 8: 2, 16: 4}
    restored = MixedPrecisionW4Linear(torch.zeros_like(q), scales * 2, runtime=runtime).cuda()
    restored.load_state_dict(state)
    for bits in (2, 4, 8, 16):
        op.set_active_bits(bits)
        restored.set_active_bits(bits)
        torch.testing.assert_close(op(x), restored(x), atol=0, rtol=0)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            streamed = op(x)
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(streamed, op(x), atol=0, rtol=0)
        assert op(x[:0]).shape == (0, n)
        torch.testing.assert_close(op(x[0]), op(x[:1])[0], atol=0, rtol=0)
    for bits in (True, 0, 3, 4.0, 32):
        with pytest.raises(ValueError):
            op.set_active_bits(bits)
    for value in (x.cpu(), x.float(), x[:, :-1]):
        with pytest.raises(ValueError):
            op(value)


@pytest.mark.gpu
@pytest.mark.parametrize("bits", [2, 4, 8])
@torch.inference_mode()
def test_activation_codes_scales_and_rounding(bits):
    from efficient_vla_benchmark.model_ops.quantization.mixed_kernel import prepare_tokens
    from efficient_vla_benchmark.model_ops.quantization.packed import unpack_signed

    high = 2 ** (bits - 1) - 1
    x = torch.zeros(2, 128, device="cuda", dtype=torch.bfloat16)
    x[0, :9] = torch.tensor([high, -high, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0], device="cuda")
    packed = torch.empty(2, 128 // (8 // bits), device="cuda", dtype=torch.uint8)
    scales = torch.empty(2, device="cuda")
    prepare_tokens[(2,)](x, packed, scales, 2, 128, bits, 128)
    reference_scales = x.float().abs().amax(1).clamp_min(1e-5) / high
    expected = (x.float() / reference_scales[:, None]).round().clamp(-high, high).to(torch.int8)
    torch.testing.assert_close(scales, reference_scales, atol=0, rtol=0)
    torch.testing.assert_close(unpack_signed(packed, bits, 128), expected, atol=0, rtol=0)
