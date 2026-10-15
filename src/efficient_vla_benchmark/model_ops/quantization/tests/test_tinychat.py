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
@pytest.mark.parametrize("rows", [1, 7, 8, 32, 280, 593])
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
    assert torch.equal(actual, direct(x.half()).to(dtype))
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
