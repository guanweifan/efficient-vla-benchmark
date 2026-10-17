import pytest
import torch

from efficient_vla_benchmark.model_ops.frozen_graph import FrozenOpGraph, GraphPool
from efficient_vla_benchmark.model_ops.quantization.tinychat import TinyChatLinear
from efficient_vla_benchmark.model_ops.quantization.tinychat_runtime import (
    TinyChatStreamRuntime,
    derive_sources,
    read_runtime,
)


def test_missing_runtime_and_invalid_frozen_inputs(tmp_path):
    with pytest.raises(ValueError, match="runtime-path"):
        read_runtime(None)
    with pytest.raises(FileNotFoundError):
        derive_sources(tmp_path, tmp_path / "derived")
    graph = FrozenOpGraph(lambda x: x + 1, GraphPool())
    with pytest.raises(ValueError, match="CUDA"):
        graph(torch.ones(2))
    assert not graph.metadata()["captured"]
    graph.clear()
    graph.clear()


@pytest.mark.gpu
@torch.inference_mode()
def test_stream_engine_graph_values_replay_reload(runtime_path):
    engine = TinyChatStreamRuntime(runtime_path)
    torch.manual_seed(183)
    codes = torch.randint(-8, 8, (256, 256), dtype=torch.int8, device="cuda")
    scales = torch.rand(256, 2, device="cuda") * 0.01 + 0.001
    op = TinyChatLinear(codes, scales).cuda()
    for rows in (1, 7, 8, 17, 32, 34, 280):
        x = torch.randn(1, rows, 256, device="cuda", dtype=torch.float16)
        op.stream_runtime = None
        reference = op(x)
        op.stream_runtime = engine
        torch.testing.assert_close(op(x), reference, atol=0, rtol=0)
        pool = GraphPool()
        graph = FrozenOpGraph(op, pool)
        expected = op(x)
        retained = graph(x)
        torch.testing.assert_close(retained, expected, atol=0, rtol=0)
        # Changed values, replay after an unrelated graph, and retained outputs.
        other = FrozenOpGraph(lambda a, b: a * b + 1, pool)
        other(x, torch.ones(256, device="cuda", dtype=x.dtype))
        for seed in (31, 94, 28):
            torch.manual_seed(seed)
            x.copy_(torch.randn_like(x))
            torch.testing.assert_close(graph(x), op(x), atol=0, rtol=0)
        torch.testing.assert_close(retained, expected, atol=0, rtol=0)
        with pytest.raises(ValueError, match="geometry"):
            graph(x.repeat(1, 2, 1) if rows == 1 else x[:, :1])
        graph.clear()
        saved = {k: v.clone() for k, v in op.state_dict().items()}
        op.load_state_dict(saved)
        torch.testing.assert_close(graph(x), op(x), atol=0, rtol=0)
        graph.clear()
        other.clear()


@pytest.mark.gpu
@torch.inference_mode()
def test_graph_copies_dynamic_scales_without_capturing_old_values():
    pool = GraphPool()

    def op(x, scale):
        return (x / scale).round().clamp(-8, 7) * scale

    graph = FrozenOpGraph(op, pool)
    x = torch.randn(2, 17, 768, device="cuda")
    saved = []
    for value in (0.03, 0.2, 0.05, 0.03):
        scale = torch.full((768,), value, device="cuda")
        result = graph(x, scale)
        torch.testing.assert_close(result, op(x, scale), atol=0, rtol=0)
        saved.append(result)
    assert not torch.equal(saved[0], saved[1])
    torch.testing.assert_close(saved[0], saved[-1], atol=0, rtol=0)
    assert graph.replays == 4
