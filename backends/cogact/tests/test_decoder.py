"""Native tiny-Llama references for causal masks, positions and sparse readout."""

from types import SimpleNamespace

import pytest
import torch
from efficient_vla_backend_cogact.decoder import CausalTokenAdapter, TokenLayout, TokenStrategy
from transformers import LlamaConfig, LlamaForCausalLM


def make_model():
    torch.manual_seed(17)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    config._attn_implementation = "sdpa"
    return LlamaForCausalLM(config).eval()


def layout(length):
    return TokenLayout(length, patches_per_view=4)


class FixedSelection(TokenStrategy):
    def __init__(self, selected=(1, 3), layer=0, after=False):
        self.selected, self.layer, self.after = selected, layer, after
        self.final_positions = None

    def select(self, index, state):
        if index != self.layer:
            return None
        keep = ~state.layout.visual_mask(state.positions)
        for original in self.selected:
            keep |= state.positions == original
        return keep

    def before_layer(self, index, state):
        return None if self.after else self.select(index, state)

    def after_layer(self, index, state):
        return self.select(index, state) if self.after else None

    def finish(self, state):
        self.final_positions = state.positions.clone()


@torch.inference_mode()
def reference(model, embeddings, strategy, attention_mask):
    """Independent explicit causal reference, not the adapter's mask helper."""
    hidden = embeddings.clone()
    positions = torch.arange(hidden.shape[1])

    def select():
        nonlocal hidden, positions
        keep = (positions == 0) | (positions >= 5)
        for original in strategy.selected:
            keep |= positions == original
        hidden, positions = hidden[:, keep], positions[keep]

    for index, layer in enumerate(model.model.layers):
        if index == strategy.layer and not strategy.after:
            select()
        permitted = positions[None, :] <= positions[:, None]
        if attention_mask is not None:
            permitted &= attention_mask[0, positions][None, :].bool()
        mask = hidden.new_zeros((1, 1, len(positions), len(positions)))
        mask.masked_fill_(~permitted[None, None], torch.finfo(hidden.dtype).min)
        hidden = layer(
            hidden,
            attention_mask=mask,
            position_ids=positions[None],
            use_cache=False,
            cache_position=positions,
        )[0]
        if index == strategy.layer and strategy.after:
            select()
    return model.model.norm(hidden), positions


@pytest.mark.parametrize("use_cache", [False, True])
def test_full_retention_hooks_preserve_native_prefill_and_readout(use_cache):
    model = make_model()
    embeddings = torch.randn(1, 12, 32)
    with torch.inference_mode():
        original = model(inputs_embeds=embeddings, use_cache=use_cache, output_hidden_states=True)
        original_attentions = [layer.self_attn for layer in model.model.layers]
        adapter = CausalTokenAdapter(model.model, FixedSelection((1, 2, 3, 4)), layout)
        actual = model(inputs_embeds=embeddings, use_cache=use_cache, output_hidden_states=True)
        torch.testing.assert_close(original.logits, actual.logits, atol=0, rtol=0)
        torch.testing.assert_close(
            original.hidden_states[-1], actual.hidden_states[-1], atol=0, rtol=0
        )
        assert [layer.self_attn for layer in model.model.layers] == original_attentions
        assert adapter.state is None
        adapter.close()
        adapter.close()
        torch.testing.assert_close(
            model(inputs_embeds=embeddings).logits, original.logits, atol=0, rtol=0
        )


@pytest.mark.parametrize("after", [False, True])
@pytest.mark.parametrize("pruning_layer", [0, 2])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("use_cache", [False, True])
def test_sparse_rows_match_independent_reference(after, pruning_layer, padded, use_cache):
    model = make_model()
    embeddings = torch.randn(1, 12, 32)
    mask = torch.ones(1, 12, dtype=torch.long) if padded else None
    if mask is not None:
        mask[0, 7] = 0
    strategy = FixedSelection(layer=pruning_layer, after=after)
    expected, positions = reference(model, embeddings, strategy, mask)
    adapter = CausalTokenAdapter(model.model, strategy, layout)
    with torch.inference_mode():
        actual = model(
            inputs_embeds=embeddings,
            attention_mask=mask,
            use_cache=use_cache,
            output_hidden_states=True,
        )
    torch.testing.assert_close(actual.hidden_states[-1], expected, rtol=1e-5, atol=1e-6)
    assert strategy.final_positions.tolist() == positions.tolist()
    assert strategy.final_positions[-1].item() == 11
    assert actual.hidden_states[-1].shape == (1, 10, 32)
    assert adapter.state is None
    adapter.close()


def test_future_text_cannot_change_earlier_visual_outputs():
    model = make_model()
    embeddings = torch.randn(1, 12, 32)
    changed = embeddings.clone()
    changed[:, 5:] *= -9
    adapter = CausalTokenAdapter(model.model, FixedSelection(), layout)
    with torch.inference_mode():
        a = model(inputs_embeds=embeddings, output_hidden_states=True).hidden_states[-1]
        b = model(inputs_embeds=changed, output_hidden_states=True).hidden_states[-1]
    torch.testing.assert_close(a[:, :3], b[:, :3], rtol=0, atol=0)
    assert not torch.equal(a[:, -1], b[:, -1])
    adapter.close()


@pytest.mark.parametrize("removed", [0, 5, 11])
def test_nonvisual_tokens_cannot_be_removed_and_error_state_clears(removed):
    class Bad(TokenStrategy):
        def before_layer(self, index, state):
            keep = torch.ones_like(state.positions, dtype=torch.bool)
            keep[removed] = False
            return keep

    model = make_model()
    adapter = CausalTokenAdapter(model.model, Bad(), layout)
    with pytest.raises(ValueError, match="nonvisual"):
        model(inputs_embeds=torch.randn(1, 12, 32))
    assert adapter.state is None
    adapter.close()


def test_cached_decode_is_rejected_and_close_removes_ownership():
    model = make_model()
    embeddings = torch.randn(1, 12, 32)
    with torch.inference_mode():
        past = model(inputs_embeds=embeddings, use_cache=True).past_key_values
    adapter = CausalTokenAdapter(model.model, FixedSelection(), layout)
    with pytest.raises(ValueError, match="cached decoding"):
        model(inputs_embeds=embeddings, past_key_values=past)
    assert adapter.state is None
    with pytest.raises(ValueError, match="Only one"):
        CausalTokenAdapter(model.model, TokenStrategy(), layout)
    adapter.close()
    assert not hasattr(model.model, "_evb_token_adapter")


def test_model_access_validates_paired_grid():
    from efficient_vla_backend_cogact.model_access import CogACTModelAccess

    vision = SimpleNamespace(
        dino_featurizer=SimpleNamespace(patch_embed=SimpleNamespace(num_patches=256)),
        siglip_featurizer=SimpleNamespace(patch_embed=SimpleNamespace(num_patches=196)),
    )
    model = SimpleNamespace(
        vlm=SimpleNamespace(
            vision_backbone=vision,
            llm_backbone=SimpleNamespace(llm=SimpleNamespace(model=make_model().model)),
        )
    )
    with pytest.raises(ValueError, match="aligned"):
        CogACTModelAccess(model)
