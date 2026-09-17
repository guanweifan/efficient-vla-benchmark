import pytest
import torch
from efficient_vla_backend_openvla_oft.decoder import (
    DecoderAdapter,
    DecoderStrategy,
    IndexedKVStore,
    TokenLayout,
)
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel


def tiny_decoder():
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    config._attn_implementation = "sdpa"
    return LlamaModel(config).eval()


def tiny_layout(length):
    return TokenLayout(length, patches_per_view=2, action_tokens=2)


@torch.inference_mode()
def test_noop_decoder_matches_native_and_restores_instance():
    torch.manual_seed(0)
    model, other = tiny_decoder(), tiny_decoder()
    data = torch.randn(1, 12, 32)
    mask = torch.ones(1, 12)
    mask[:, -1] = 0
    expected = model(inputs_embeds=data, attention_mask=mask).last_hidden_state
    original_class = type(other.layers[0].self_attn)
    adapter = DecoderAdapter(model, DecoderStrategy(), tiny_layout)
    actual = model(inputs_embeds=data, attention_mask=mask).last_hidden_state
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
    assert type(other.layers[0].self_attn) is original_class
    with pytest.raises(ValueError, match="Only one"):
        DecoderAdapter(model, DecoderStrategy(), tiny_layout)
    adapter.close()
    assert type(model.layers[0].self_attn) is original_class
    torch.testing.assert_close(
        model(inputs_embeds=data, attention_mask=mask).last_hidden_state, expected
    )


@torch.inference_mode()
def test_pruning_preserves_positions_and_action_readout():
    class Prune(DecoderStrategy):
        def after_layer(self, layer, state, attention):
            if layer == 0:
                return state.positions != 2

        def probes(self, layer, state):
            return {"action": state.positions[state.positions >= state.layout.action_start]}

    model = tiny_decoder()
    adapter = DecoderAdapter(model, Prune(), tiny_layout)
    output = model(inputs_embeds=torch.randn(1, 12, 32), output_hidden_states=True)
    assert output.last_hidden_state.shape == (1, 12, 32)
    assert output.hidden_states[-1].shape == (1, 12, 32)
    assert torch.count_nonzero(output.last_hidden_state[:, 2]) == 0
    assert adapter.state.positions.tolist() == [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert model.layers[2].self_attn.last_attention["action"].shape == (3, 11)
    assert torch.isfinite(output.last_hidden_state[:, tiny_layout(12).readout_start :]).all()
    adapter.close()


@torch.inference_mode()
def test_nonvisual_pruning_is_rejected():
    class Invalid(DecoderStrategy):
        def before_layer(self, layer, state):
            return state.positions != 0

    model = tiny_decoder()
    adapter = DecoderAdapter(model, Invalid(), tiny_layout)
    with pytest.raises(ValueError, match="non-visual"):
        model(inputs_embeds=torch.randn(1, 12, 32))
    adapter.close()


def test_indexed_cache_overwrites_without_growing_and_resets():
    cache = IndexedKVStore()
    keys = torch.arange(12).reshape(1, 1, 4, 3).float()
    values = keys + 100
    cache.update(0, keys, values, torch.arange(4), 4)
    updated, _ = cache.update(
        0, keys[:, :, [0, 3]] + 1, values[:, :, [0, 3]], torch.tensor([0, 3]), 4
    )
    torch.testing.assert_close(updated[:, :, [1, 2]], keys[:, :, [1, 2]])
    torch.testing.assert_close(updated[:, :, [0, 3]], keys[:, :, [0, 3]] + 1)
    assert updated.shape == keys.shape
    cache.clear()
    with pytest.raises(ValueError, match="complete frame"):
        cache.update(0, keys[:, :, :2], values[:, :, :2], torch.arange(2), 4)


def test_oft_layout_keeps_native_readout_offset():
    layout = TokenLayout(601)
    assert layout.visual_end == 513 and layout.text_start == 514
    assert layout.text_end == 544 and layout.readout_start == 543


@torch.inference_mode()
def test_indexed_reuse_preserves_dense_output_for_identical_frames():
    class Reuse(DecoderStrategy):
        def __init__(self):
            self.store = IndexedKVStore()
            self.warm = False

        def before_layer(self, layer, state):
            if self.warm and layer == 1:
                return state.positions != 2

        def key_positions(self, layer, state):
            return torch.arange(state.layout.length, device=state.positions.device)

        def update_kv(self, layer, keys, values, state):
            return self.store.update(layer, keys, values, state.positions, state.layout.length)

        def finish(self, state):
            self.warm = True

    torch.manual_seed(3)
    model = tiny_decoder()
    inputs = torch.randn(1, 12, 32)
    mask = torch.ones(1, 12)
    reference = model(inputs_embeds=inputs, attention_mask=mask).last_hidden_state
    adapter = DecoderAdapter(model, Reuse(), tiny_layout)
    cold = model(inputs_embeds=inputs, attention_mask=mask).last_hidden_state
    warm = model(inputs_embeds=inputs, attention_mask=mask).last_hidden_state
    torch.testing.assert_close(cold, reference, atol=1e-6, rtol=0)
    torch.testing.assert_close(warm[:, 5:], reference[:, 5:], atol=1e-6, rtol=0)
    adapter.close()


@pytest.mark.parametrize("attention_backend", ["eager", "sdpa"])
@torch.inference_mode()
def test_full_attention_probe_matches_reference_and_does_not_recompute(
    attention_backend, monkeypatch
):
    from transformers.models.llama.modeling_llama import LlamaAttention

    class FullProbe(DecoderStrategy):
        def probes(self, layer, state):
            return {"full": state.positions}

    torch.manual_seed(31)
    model = tiny_decoder()
    original = model.layers[0].self_attn
    hidden = torch.randn(1, 12, 32)
    positions = torch.arange(12).unsqueeze(0)
    mask = torch.zeros(1, 1, 12, 12)
    mask[..., -1] = torch.finfo(hidden.dtype).min
    expected, probabilities, _ = LlamaAttention.forward(
        original, hidden, attention_mask=mask, position_ids=positions, output_attentions=True
    )
    strategy = FullProbe()
    strategy.attention_backend = attention_backend
    adapter = DecoderAdapter(model, strategy, tiny_layout)
    from efficient_vla_backend_openvla_oft.decoder import DecoderState

    adapter.state = DecoderState(tiny_layout(12), positions[0])
    if attention_backend == "eager":

        def forbidden(*args, **kwargs):
            raise AssertionError("Eager output must reuse probabilities, not call SDPA")

        monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", forbidden)
    output, _, _ = model.layers[0].self_attn(hidden, attention_mask=mask, position_ids=positions)
    torch.testing.assert_close(output, expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(
        model.layers[0].self_attn.last_attention["full"],
        probabilities.mean(dim=1)[0],
        atol=1e-7,
        rtol=0,
    )
    adapter.close()


@torch.inference_mode()
def test_probe_outside_active_range_is_rejected():
    class InvalidProbe(DecoderStrategy):
        def probes(self, layer, state):
            return {"invalid": state.positions.new_tensor([state.layout.length + 1])}

    model = tiny_decoder()
    adapter = DecoderAdapter(model, InvalidProbe(), tiny_layout)
    with pytest.raises(ValueError, match="removed query"):
        model(inputs_embeds=torch.randn(1, 12, 32))
    adapter.close()
