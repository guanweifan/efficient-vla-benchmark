import pytest
import torch

from efficient_vla_benchmark.model_ops.quantization.tinychat import TinyChatLinear, reference_module


def test_invalid_geometry_rejected_without_extension():
    with pytest.raises(ValueError):
        TinyChatLinear(torch.zeros(128, 128), torch.ones(128, 1))
    with pytest.raises(ValueError):
        TinyChatLinear(torch.zeros(129, 128, dtype=torch.int8), torch.ones(129, 1))
    with pytest.raises(ValueError):
        TinyChatLinear(torch.full((128, 128), -9, dtype=torch.int8), torch.ones(128, 1))
    with pytest.raises(ValueError):
        TinyChatLinear(torch.zeros(128, 128, dtype=torch.int8), torch.full((128, 1), 1e10))


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 7, 8, 16, 17, 32, 33, 34, 280, 593])
@torch.inference_mode()
def test_author_dispatch_roundtrip_and_reload(dtype, rows):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(127)
    codes = torch.randint(-8, 8, (256, 256), dtype=torch.int8, device="cuda")
    scales = (torch.rand(256, 2, device="cuda") * 0.01 + 0.001).half()
    bias = torch.randn(256, device="cuda").half()
    module = TinyChatLinear(codes, scales, bias)
    x = torch.randn(1, rows, 256, device="cuda", dtype=dtype)
    actual = module(x)
    direct = reference_module().WQLinear(4, 128, 256, 256, True, "cuda")
    direct.load_state_dict(module.packed.state_dict())
    if 8 <= rows <= 32:
        padded = x.new_zeros((33, 256), dtype=torch.float16)
        padded[:rows].copy_(x.half().reshape(rows, 256))
        expected = direct(padded)[:rows].reshape_as(actual).to(dtype)
    else:
        expected = direct(x.half()).to(dtype)
    assert torch.equal(actual, expected)
    dense_weight = (codes.float() * scales.float().repeat_interleave(128, 1)).half()
    dense = torch.nn.functional.linear(x.half(), dense_weight, bias).to(dtype)
    relative = (actual.float() - dense.float()).abs().mean() / dense.float().abs().mean()
    assert relative < 0.012
    assert torch.equal(actual, module(x))
    reload = TinyChatLinear(codes.cpu(), scales.cpu(), bias.cpu()).cuda()
    reload.load_state_dict(module.state_dict())
    assert torch.equal(actual, reload(x))
    assert module(x[:, :0]).shape == (1, 0, 256)
    assert not list(module.parameters())
    assert all(t.shape != codes.shape for t in module.buffers())
    with pytest.raises(ValueError):
        module(x.cpu())
    with pytest.raises(ValueError):
        module(x.float())


@pytest.mark.gpu
@pytest.mark.parametrize("rows", [8, 17, 32, 34])
@torch.inference_mode()
def test_short_dispatch_independent_of_recycled_semaphore_storage(rows):
    torch.manual_seed(918)
    n, k = 3072, 768
    codes = torch.randint(-8, 8, (n, k), dtype=torch.int8, device="cuda")
    scales = torch.rand(n, k // 128, device="cuda") * 0.01 + 0.003
    module = TinyChatLinear(codes, scales)
    x = torch.randn(1, rows, k, device="cuda", dtype=torch.float16)
    expected = module(x).clone()
    for _ in range(20):
        # A prior allocation can leave the author's split-K semaphore value 1.
        scratch = torch.ones(((rows + 15) // 16 * (n // 128),), device="cuda", dtype=torch.int32)
        torch.cuda.synchronize()
        del scratch
        torch.testing.assert_close(module(x), expected, atol=0, rtol=0)
    assert torch.isfinite(expected).all()
    assert module.execution_metadata()["short_row_dispatch"].endswith("single_split_then_slice")
