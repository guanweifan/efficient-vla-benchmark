import pytest
import torch


@pytest.mark.gpu
@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("activation", [False, True])
@torch.inference_mode()
def test_sequence_tiles_match_original_reduction(bits, dtype, activation):
    import triton

    from efficient_vla_benchmark.model_ops.quantization.kernel import packed_gemm
    from efficient_vla_benchmark.model_ops.quantization.linear import PackedLinear

    torch.manual_seed(29)
    n, k = 65, 97
    limit = 1 << (bits - 1)
    codes = torch.randint(-limit, limit, (n, k), dtype=torch.int8, device="cuda")
    scales = torch.rand(n, 4, device="cuda") * 0.02 + 0.001
    step = torch.rand(k, device="cuda") * 0.02 + 0.001 if activation else None
    bias = torch.randn(n, device="cuda", dtype=dtype) * 0.01
    op = PackedLinear(
        codes,
        scales,
        bits=bits,
        group_size=32,
        bias=bias,
        activation_step=step,
        activation_qmin=-128,
        round_weight_scale=activation,
    ).cuda()
    for rows in (31, 32, 63, 64, 65, 97):
        inputs = torch.randn(rows, k, device="cuda", dtype=dtype)
        qinputs = inputs
        astep = scales
        if activation:
            astep = step.to(dtype)
            qinputs = (inputs / astep).round().clamp(-128, 127).to(torch.int8)
        reference = torch.empty(rows, n, device="cuda", dtype=dtype)
        packed_gemm[(triton.cdiv(rows, 16), triton.cdiv(n, 32))](
            qinputs,
            op.packed_weight,
            op.weight_scale,
            astep,
            bias,
            reference,
            rows,
            n,
            k,
            bits,
            32,
            4,
            activation,
            activation,
            True,
            16,
            32,
            32,
            num_warps=4,
        )
        torch.testing.assert_close(op(inputs), reference, atol=0, rtol=0)
