"""Independent bipartite/mass/attention references and native-grid contracts."""

import importlib
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from timm.models.vision_transformer import Attention, VisionTransformer

from ..config import ToMeConfig
from ..merging import attention_with_mass, bipartite_groups, weighted_merge
from ..method import ToMeCore


def reference_groups(keys, removal, prefix):
    batch, length, _ = keys.shape
    r = min(removal, (length - prefix) // 2)
    if r == 0:
        return torch.arange(length).expand(batch, -1)
    output = []
    for example in keys:
        normalized = example.float() / example.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
        proposals = []
        for left in range(0, length, 2):
            if left < prefix:
                continue
            values = [
                (float(normalized[left] @ normalized[right]), right)
                for right in range(1, length, 2)
                if right >= prefix
            ]
            value, target = max(values, key=lambda item: (item[0], -item[1]))
            proposals.append((value, left, target))
        proposals.sort(key=lambda item: (-item[0], item[1]))
        survivors = [x[1] for x in proposals[r:]]
        if prefix:
            survivors.sort()
        survivors = (
            list(range(prefix)) + survivors + [i for i in range(1, length, 2) if i >= prefix]
        )
        remap = {token: i for i, token in enumerate(survivors)}
        for _, source, target in proposals[:r]:
            remap[source] = remap[target]
        output.append([remap[i] for i in range(length)])
    return torch.tensor(output)


@pytest.mark.parametrize("prefix", [0, 1, 5])
@pytest.mark.parametrize("length", [6, 8, 17])
@pytest.mark.parametrize("removal", [0, 1, 4, 100])
def test_matching_mass_and_unmerge_reference(prefix, length, removal):
    torch.manual_seed(101)
    keys = torch.randn(2, length, 7)
    actual, count = bipartite_groups(keys, removal, prefix)
    expected = reference_groups(keys, removal, prefix)
    assert torch.equal(actual, expected)
    features = torch.randn(2, length, 11)
    mass = torch.arange(1, length + 1).float().expand(2, -1)
    merged, summed_mass = weighted_merge(features, mass, actual, count)
    for batch in range(2):
        for group in range(count):
            members = expected[batch] == group
            total = mass[batch, members].sum()
            assert summed_mass[batch, group] == total
            reference = (features[batch, members] * mass[batch, members, None]).sum(0) / total
            torch.testing.assert_close(merged[batch, group], reference)
        torch.testing.assert_close(summed_mass[batch].sum(), mass[batch].sum())
    if prefix:
        assert torch.equal(actual[:, :prefix], torch.arange(prefix).expand(2, -1))
        assert (actual[:, prefix:] >= prefix).all()
        torch.testing.assert_close(merged[:, :prefix], features[:, :prefix])


def test_exact_ties_are_stable_and_zero_metric_is_finite():
    groups, count = bipartite_groups(torch.ones(1, 6, 4), 2, 0)
    assert groups.tolist() == [[1, 1, 1, 2, 0, 3]]
    zeros, _ = bipartite_groups(torch.zeros(1, 6, 4), 2, 0)
    assert torch.equal(groups, zeros) and count == 4


@pytest.mark.parametrize("fused", [False, True])
def test_proportional_attention_matches_dense_equation(fused):
    torch.manual_seed(53)
    attention = Attention(32, num_heads=4, qkv_bias=True).eval()
    attention.fused_attn = fused
    hidden = torch.randn(2, 7, 32)
    mass = torch.tensor([[1, 2, 1, 4, 2, 3, 1], [1, 1, 1, 7, 2, 1, 1]]).float()
    qkv = attention.qkv(hidden).reshape(2, 7, 3, 4, 8).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    logits = (q @ k.transpose(-1, -2)) / 8**0.5 + mass.log()[:, None, None, :]
    expected = (logits.softmax(-1) @ v).transpose(1, 2).reshape(2, 7, 32)
    expected = attention.proj(expected)
    actual, metric = attention_with_mass(attention, hidden, mass)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(metric, k.mean(1))


def tiny_encoder(class_token, registers=0):
    encoder = VisionTransformer(
        img_size=32,
        patch_size=8,
        embed_dim=32,
        depth=4,
        num_heads=4,
        num_classes=0,
        class_token=class_token,
        reg_tokens=registers,
        global_pool="token" if class_token else "avg",
    ).eval()
    native = partial(encoder.get_intermediate_layers, n={2})
    encoder.forward = lambda x: native(x)[0]
    return encoder


@pytest.mark.parametrize("class_token,registers", [(False, 0), (True, 0), (True, 4)])
@torch.inference_mode()
def test_full_vision_equations_before_mlp_and_original_patch_unmerge(class_token, registers):
    torch.manual_seed(75)
    encoder = tiny_encoder(class_token, registers)
    images = torch.randn(2, 3, 32, 32)
    hidden = encoder.norm_pre(encoder.patch_drop(encoder._pos_embed(encoder.patch_embed(images))))
    membership = torch.arange(hidden.shape[1]).expand(2, -1)
    mass = torch.ones(hidden.shape[:2])
    for block in encoder.blocks[:-1]:
        normalized = block.norm1(hidden)
        batch, length, channels = hidden.shape
        q, k, v = block.attn.qkv(normalized).reshape(batch, length, 3, 4, 8).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-1, -2) / 8**0.5 + mass.log()[:, None, None]
        attention = (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, length, channels)
        hidden = hidden + block.ls1(block.attn.proj(attention))
        groups = reference_groups(k.mean(1), 2, encoder.num_prefix_tokens)
        merged, masses = [], []
        for batch_index in range(batch):
            current, current_mass = [], []
            for group in range(length - 2):
                members = groups[batch_index] == group
                total = mass[batch_index, members].sum()
                current_mass.append(total)
                current.append(
                    (hidden[batch_index, members] * mass[batch_index, members, None]).sum(0) / total
                )
            merged.append(torch.stack(current))
            masses.append(torch.stack(current_mass))
        hidden, mass = torch.stack(merged), torch.stack(masses)
        hidden = hidden + block.ls2(block.mlp(block.norm2(hidden)))
        membership = groups.gather(1, membership)
    prefix = encoder.num_prefix_tokens
    expected = hidden.gather(1, membership[..., None].expand(-1, -1, hidden.shape[-1]))[:, prefix:]
    method = ToMeCore(
        SimpleNamespace(query_diagnostics={}), ToMeConfig(tokens_per_layer=2), {"vision": encoder}
    )
    try:
        actual = encoder(images)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    finally:
        method.close()


@torch.inference_mode()
def test_penultimate_grid_restore_lifecycle_and_instance_isolation():
    torch.manual_seed(31)
    encoders = {"dino": tiny_encoder(True), "siglip": tiny_encoder(False)}
    backend = SimpleNamespace(query_diagnostics={})
    images = torch.randn(2, 3, 32, 32)
    native = {name: model(images) for name, model in encoders.items()}
    forwards = {name: model.forward for name, model in encoders.items()}
    original_blocks = {
        name: ["forward" in b.__dict__ for b in model.blocks] for name, model in encoders.items()
    }
    for config in (ToMeConfig(enabled=False), ToMeConfig(tokens_per_layer=0)):
        method = ToMeCore(backend, config, encoders)
        assert not method.patches.saved
        for name, model in encoders.items():
            torch.testing.assert_close(model(images), native[name], atol=0, rtol=0)
        method.close()
    method = ToMeCore(backend, ToMeConfig(tokens_per_layer=2), encoders)
    method.before_query({})
    actual = {name: model(images) for name, model in encoders.items()}
    method.after_query(None)
    assert not method.state
    for name in encoders:
        assert actual[name].shape == (2, 16, 32)
        assert method.vision_lengths[name] == [[16, 14, 12, 10]]
        assert method.retained[name] == [10]
    method.reset(None)
    for name, model in encoders.items():
        torch.testing.assert_close(model(images), actual[name], atol=0, rtol=0)
    # An invalid input clears transient membership even when native patchify fails.
    with pytest.raises(AssertionError):
        encoders["dino"](torch.zeros(1, 3, 31, 31))
    assert not method.state
    method.close()
    method.close()
    for name, model in encoders.items():
        assert model.forward is forwards[name]
        assert ["forward" in b.__dict__ for b in model.blocks] == original_blocks[name]
        assert not hasattr(model, "_evb_tome")
        torch.testing.assert_close(model(images), native[name], atol=0, rtol=0)


@pytest.mark.parametrize(
    "parameters",
    [
        {"enabled": 1},
        {"tokens_per_layer": True},
        {"tokens_per_layer": -1},
        {"tokens_per_layer": 1.5},
        {"tokens_per_layer": 129},
        {"unknown": 1},
    ],
)
def test_invalid_configuration(parameters):
    with pytest.raises(ValueError):
        ToMeConfig.parse(parameters)


@pytest.mark.gpu
@pytest.mark.parametrize("backend_id", ["openvla_oft", "cogact"])
def test_real_native_noop_active_grid_reset_and_detach(backend_id, model_path, trace_path, request):
    module = pytest.importorskip(f"efficient_vla_backend_{backend_id}.backend")
    from efficient_vla_benchmark.evaluation.traces import load_trace

    build = importlib.import_module(
        f"efficient_vla_benchmark.methods.token_reduction.tome.implementations.{backend_id}.api"
    ).build
    cls = getattr(module, "CogACTBackend" if backend_id == "cogact" else "OpenVLAOFTBackend")
    backend = cls(model_path)
    request.addfinalizer(backend.close)
    context, observations, _ = load_trace(trace_path)
    backend.reset(context)
    native = backend.predict_chunk(observations[0])
    native_rng = torch.cuda.get_rng_state().clone()
    for config in (ToMeConfig(enabled=False), ToMeConfig(tokens_per_layer=0)):
        backend.attach_method(build(backend, config, None))
        backend.reset(context)
        np.testing.assert_array_equal(backend.predict_chunk(observations[0]), native)
        backend.detach_method()
    method = build(backend, ToMeConfig(), None)
    backend.attach_method(method)
    backend.reset(context)
    actual = backend.predict_chunk(observations[0])
    assert actual.shape == native.shape and np.isfinite(actual).all()
    assert torch.equal(torch.cuda.get_rng_state(), native_rng)
    diagnostics = backend.query_diagnostics
    for name, lengths in diagnostics["tome_vision_layer_patch_counts"].items():
        for counts in lengths:
            assert counts[0] == 256
            assert counts == [256 - 4 * i for i in range(len(counts))]
        assert diagnostics["tome_vision_retained"][name] == [counts[-1] for counts in lengths]
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(observations[0]), actual)
    backend.detach_method()
    method.close()
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(observations[0]), native)
