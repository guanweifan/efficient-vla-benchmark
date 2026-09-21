import numpy as np
import pytest

from efficient_vla_benchmark.methods.token_reduction.team_vla.config import TEAMVLAConfig


@pytest.mark.parametrize(
    "bad",
    [
        {"enabled": 1},
        {"context_ratio": -1},
        {"context_ratio": float("nan")},
        {"expansion_kernel": 2},
        {"density_threshold": -1},
        {"merging_layer": -1},
        {"merge_tokens": 0},
        {"rms_epsilon": 0},
        {"seed": -1},
    ],
)
def test_invalid_config(bad):
    with pytest.raises(ValueError):
        TEAMVLAConfig.parse(bad)


def test_expansion_and_interval_sampling():
    import torch

    from ..implementations.openvla_oft.method import early_selection, expand_seeds

    generator = torch.Generator().manual_seed(2)
    seeds = torch.zeros(2, 5, 5, dtype=torch.bool)
    seeds[:, 2, 2] = True
    # Density == threshold neither dense nor sparse: strict paper inequalities.
    torch.testing.assert_close(expand_seeds(seeds, 3, 1, generator), seeds)
    expanded = expand_seeds(seeds, 3, 0, generator)
    assert expanded.all()  # Convolution support, followed by dense neighborhood fill.
    sparse = expand_seeds(seeds, 3, 2, generator)
    assert torch.all(sparse >= seeds) and int(sparse.sum()) > 2
    cfg = TEAMVLAConfig(context_ratio=1)
    assert early_selection(torch.randn(2, 16, 8), torch.randn(3, 8), cfg, generator).all()


def test_soft_merge_matches_independent_scalar_weighted_sum():
    import torch

    from ..implementations.openvla_oft.method import soft_merge

    torch.manual_seed(8)
    visual, queries = torch.randn(12, 8), torch.randn(3, 8)
    ids, merged = soft_merge(visual, queries, 4, 1e-6)
    source = visual[ids]
    targets = [i for i in range(12) if i not in ids.tolist()]
    sums, mass = source.clone(), torch.ones(4)
    for i in targets:
        scores = []
        for anchor in source:
            a = anchor / torch.sqrt(anchor.square().mean() + 1e-6)
            b = visual[i] / torch.sqrt(visual[i].square().mean() + 1e-6)
            scores.append(torch.dot(a, b) / 8**0.5)
        weights = torch.stack(scores).softmax(0)
        sums += weights[:, None] * visual[i]
        mass += weights
    torch.testing.assert_close(merged, sums / mass[:, None])
    ids, identity = soft_merge(visual, queries, 12, 1e-6)
    torch.testing.assert_close(identity, visual, rtol=0, atol=0)


def test_dense_expansion_matches_explicit_neighborhood_union():
    import torch

    from ..implementations.openvla_oft.method import expand_seeds

    seeds = torch.zeros(2, 7, 7, dtype=torch.bool)
    seeds[0, 0, 0] = seeds[0, 1, 1] = seeds[0, 5, 5] = True
    seeds[1, 3, 3] = True
    expected = seeds.clone()
    # tau=1 has no sparse branch: enumerate Eq. 3's density then dilation.
    for view in range(2):
        for row in range(7):
            for col in range(7):
                ys, xs = (
                    slice(max(row - 1, 0), min(row + 2, 7)),
                    slice(max(col - 1, 0), min(col + 2, 7)),
                )
                if seeds[view, ys, xs].sum() > 1:
                    expected[view, ys, xs] = True
    torch.testing.assert_close(expand_seeds(seeds, 3, 1, torch.Generator()), expected)
    assert expected[1].sum() == 1


def test_early_selection_uses_local_rng_and_resets_reproducibly():
    import torch

    from ..implementations.openvla_oft.method import early_selection

    torch.manual_seed(12)
    visual, text = torch.randn(2, 16, 8), torch.randn(4, 8)
    state = torch.random.get_rng_state().clone()
    a = early_selection(
        visual, text, TEAMVLAConfig(density_threshold=2), torch.Generator().manual_seed(9)
    )
    b = early_selection(
        visual, text, TEAMVLAConfig(density_threshold=2), torch.Generator().manual_seed(9)
    )
    assert torch.equal(a, b)
    assert torch.equal(torch.random.get_rng_state(), state)


def test_soft_merge_normalizes_by_received_mass_not_source_count():
    import torch

    from ..implementations.openvla_oft.method import soft_merge

    # A single surviving source receives unit mass from EVERY target.
    visual = torch.tensor([[4.0, 0.0], [1.0, 1.0], [0.0, 2.0], [-1.0, 1.0]])
    ids, merged = soft_merge(visual, torch.tensor([[1.0, 0.0]]), 1, 1e-6)
    assert ids.tolist() == [0]
    torch.testing.assert_close(merged[0], visual.mean(0))


@pytest.mark.gpu
def test_real_oft_merge_and_reset(model_path, trace_path, request):
    from efficient_vla_backend_openvla_oft.backend import OpenVLAOFTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..implementations.openvla_oft.method import TEAMVLA

    backend = OpenVLAOFTBackend(model_path)
    request.addfinalizer(backend.close)
    context, samples, _ = load_trace(trace_path)
    backend.reset(context)
    expected = backend.predict_chunk(samples[0])
    for params in ({"enabled": False}, {"context_ratio": 1, "merge_tokens": 512}):
        backend.attach_method(TEAMVLA(backend, TEAMVLAConfig.parse(params)))
        backend.reset(context)
        np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
        backend.detach_method()
    method = TEAMVLA(backend, TEAMVLAConfig())
    backend.attach_method(method)
    backend.reset(context)
    actual = backend.predict_chunk(samples[0])
    assert actual.shape == (8, 7) and np.isfinite(actual).all()
    assert backend.query_diagnostics["visual_tokens_final"] == 80
    assert 128 <= backend.query_diagnostics["visual_tokens_early"] <= 512
    sizes = backend.query_diagnostics["layer_sequence_lengths"]
    assert sizes[0] == sizes[15] and sizes[16] == sizes[31] and sizes[0] > sizes[16]
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(samples[0]), actual)
    backend.detach_method()
    np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
