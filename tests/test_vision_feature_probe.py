"""Optional tensor probe: native outputs/gradients and instance lifetime stay intact."""

import pytest

torch = pytest.importorskip("torch")

from efficient_vla_benchmark.model_ops.vision import VisionFeatureProbe  # noqa: E402


def test_snapshot_is_detached_independent_and_preserves_native_output():
    block = torch.nn.Identity()
    probe = VisionFeatureProbe(block)
    x = torch.ones(1, 5, 4, requires_grad=True)
    output = block(x)
    assert output is x
    snapshots = probe.take()
    assert len(snapshots) == 1 and not snapshots[0].requires_grad
    output.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    with torch.no_grad():
        output.add_(2)
    assert snapshots[0].eq(1).all()
    assert probe.take() == []
    probe.close()


def test_view_order_clear_idempotent_close_and_instance_isolation():
    block = torch.nn.Identity()
    first, second = VisionFeatureProbe(block), VisionFeatureProbe(block)
    block(torch.ones(1, 5, 4))
    block(torch.full((1, 5, 4), 2.0))
    assert [float(x[0, 0, 0]) for x in first.take()] == [1.0, 2.0]
    assert len(second.features) == 2
    second.clear()
    first.close()
    first.close()
    block(torch.ones(1, 5, 4))
    assert first.take() == [] and len(second.take()) == 1
    second.close()
    assert len(block._forward_hooks) == 0


@pytest.mark.parametrize("shape", [(4,), (4, 5), (1, 2, 3, 4)])
def test_invalid_output_rejected_without_retaining_it(shape):
    block = torch.nn.Identity()
    probe = VisionFeatureProbe(block)
    try:
        with pytest.raises(ValueError, match="batched token"):
            block(torch.ones(*shape))
        assert not probe.features
    finally:
        probe.close()
