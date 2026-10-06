"""Independent full-map reductions for compact attention consumers."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from efficient_vla_benchmark.model_ops.attention_statistics import (  # noqa: E402
    AttentionStatisticsRequest,
    attention_statistics,
)


@pytest.mark.parametrize("rows", [(0, 3), (5, 8)])
def test_complete_entropy_and_text_rectangle_match_numpy(rows):
    p = np.random.default_rng(71).uniform(0, 1, (8, 13)).astype(np.float32)
    p[0] = 0
    request = AttentionStatisticsRequest(*rows, 1, 9)
    result = attention_statistics(torch.tensor(p), request)
    norm = p / (p.sum(-1, keepdims=True) + 1e-10)
    expected = -(norm * np.log(norm + 1e-10)).sum(-1).mean()
    assert float(result.entropy) == pytest.approx(float(expected), rel=1e-6)
    np.testing.assert_array_equal(result.text_attention.numpy(), p[rows[0] : rows[1], 1:9])
    assert result.text_attention.is_contiguous()
    assert result.text_attention.untyped_storage().nbytes() == result.text_attention.numel() * 4


def test_reject_bad_statistics_bounds():
    with pytest.raises(ValueError):
        attention_statistics(torch.zeros(8, 13), AttentionStatisticsRequest(4, 4, 1, 9))
