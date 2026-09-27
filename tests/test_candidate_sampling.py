import pytest

torch = pytest.importorskip("torch")

from efficient_vla_benchmark.model_ops.candidate_sampling import expand_candidates  # noqa: E402


@pytest.mark.parametrize("using_cfg", [False, True])
@pytest.mark.parametrize("count", [1, 4, 20])
def test_exact_native_draw_order_and_branch_pairing(using_cfg, count):
    rows = 2 if using_cfg else 1
    native = torch.arange(112, dtype=torch.float32).reshape(1, 16, 7).expand(rows, -1, -1)
    condition = torch.arange(rows * 12, dtype=torch.float32).reshape(rows, 1, 12)
    torch.manual_seed(43)
    # Independent pre-refactor implementation, including discarded native RNG state.
    extra = torch.randn((count - 1, 16, 7)).to(native.dtype)
    expected = torch.cat((native[:1], extra))
    expected_condition = condition[:1].expand(count, 1, 12)
    if using_cfg:
        expected = torch.cat((expected, expected))
        expected_condition = torch.cat((expected_condition, condition[1:2].expand(count, 1, 12)))
    state = torch.get_rng_state().clone()
    torch.manual_seed(43)
    actual, actual_condition = expand_candidates(native, condition, count, using_cfg)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_condition, expected_condition, rtol=0, atol=0)
    assert torch.equal(state, torch.get_rng_state())


def test_invalid_pairing_and_geometry():
    for count in (0, True):
        with pytest.raises(ValueError):
            expand_candidates(torch.zeros(1, 16, 7), torch.zeros(1, 1, 4), count, False)
    with pytest.raises(ValueError, match="paired"):
        expand_candidates(torch.randn(2, 16, 7), torch.zeros(2, 1, 4), 4, True)
    with pytest.raises(ValueError):
        expand_candidates(torch.zeros(2, 16, 7), torch.zeros(2, 1, 4), 4, False)
