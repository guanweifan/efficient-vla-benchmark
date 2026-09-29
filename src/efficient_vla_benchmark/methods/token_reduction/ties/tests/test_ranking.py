import itertools

import numpy as np
import pytest
import torch

from ..config import TIESConfig
from ..ranking import (
    UndefinedRankCorrelation,
    adjacent_kendall,
    attention_fraction,
    rank_consistency,
)
from ..selection import select_visual


def test_kendall_matches_common_identity_pair_oracle():
    rng = np.random.default_rng(7)
    for _ in range(80):
        first, second = rng.permutation(15)[:10].tolist(), rng.permutation(15)[:10].tolist()
        common = set(first) & set(second)
        concordance = []
        for a, b in itertools.combinations(common, 2):
            concordance.append(
                np.sign((first.index(a) - first.index(b)) * (second.index(a) - second.index(b)))
            )
        assert adjacent_kendall(first, second) == pytest.approx(np.mean(concordance))
    assert adjacent_kendall([8, 4, 2], [8, 4, 2]) == 1
    assert adjacent_kendall([8, 4, 2], [2, 4, 8]) == -1
    assert rank_consistency([[8, 4, 2], [2, 4, 8], [2, 4, 8]]) == 0
    with pytest.raises(UndefinedRankCorrelation):
        adjacent_kendall([0, 1], [1, 2])
    with pytest.raises(ValueError):
        adjacent_kendall([1, 1], [1, 2])


def test_hard_soft_equations_and_revision_sign():
    q = [-0.2, 0.1, 0.6]
    for tau, expected in [(-1, 1), (-0.2, 1), (0.2, 0.5), (0.6, 0), (1, 0)]:
        assert attention_fraction(tau, q, "soft") == pytest.approx(expected)
    assert attention_fraction(0.1, q, "hard") == 1
    assert attention_fraction(0.1001, q, "hard") == 0
    with pytest.raises(ValueError):
        attention_fraction(0.3, [0.3, 0.3, 0.3], "soft")


def test_selection_budget_no_duplicates_uniform_rng_isolation():
    scores = torch.arange(10, dtype=torch.float32)
    before = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(17)
    first, stats = select_visual(scores, 0.5, 0.6, generator)
    assert first.sum() == 5 and bool(first[7:].all())
    assert stats == dict(attention_tokens=3, uniform_tokens=2)
    generator.manual_seed(17)
    assert torch.equal(first, select_visual(scores, 0.5, 0.6, generator)[0])
    assert torch.equal(before, torch.get_rng_state())
    top, _ = select_visual(scores, 0.5, 1, generator)
    assert top.nonzero().flatten().tolist() == [5, 6, 7, 8, 9]
    uniform, stats = select_visual(scores, 0.5, 0, generator)
    assert uniform.sum() == 5 and stats["uniform_tokens"] == 5


@pytest.mark.parametrize(
    "settings",
    [
        dict(ranking_layers=1),
        dict(ranking_topk=1),
        dict(refresh_steps=0),
        dict(keep_ratio=float("nan")),
        dict(keep_ratio=True),
        dict(mode="heuristic"),
        dict(calibration_samples=99),
        dict(enabled=1),
    ],
)
def test_invalid_config(settings):
    with pytest.raises(ValueError):
        TIESConfig.parse(settings)
