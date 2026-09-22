"""Independent tiny-model evidence for causal indexed KV storage."""

import pytest
import torch
from efficient_vla_backend_cogact.decoder import TokenLayout
from efficient_vla_backend_cogact.reuse import CausalReuseAdapter, ReuseStrategy
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel

from efficient_vla_benchmark.model_ops.decoder import IndexedKVStore


class FixedReuse(ReuseStrategy):
    def __init__(self, mode):
        self.attention_backend = mode
        self.store = IndexedKVStore()
        self.warm = False
        self.maps = []

    def start(self, state):
        self.full = torch.arange(state.layout.length)
        self.maps = []

    def before_layer(self, layer, state):
        if self.warm and layer == 1:
            return ~torch.isin(state.positions, torch.tensor([1, 3]))

    def key_positions(self, layer, state):
        return self.full

    def update_kv(self, layer, keys, values, state):
        return self.store.update(layer, keys, values, state.positions, state.layout.length)

    def probes(self, layer, state):
        return {"all": state.positions, "last": state.positions[-1:]}

    def after_layer(self, layer, state, attention):
        self.maps.append((state.positions.clone(), attention["all"].clone()))
        torch.testing.assert_close(attention["last"], attention["all"][-1:], atol=1e-7, rtol=0)

    def finish(self, state):
        self.last = state.positions.clone()

    def abort(self):
        self.store.clear()


def tiny():
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


@pytest.mark.parametrize("mode", ["eager", "sdpa"])
@pytest.mark.parametrize("padding", [False, True])
@torch.inference_mode()
def test_original_causal_rows_fixed_storage_and_native_readout(mode, padding):
    torch.manual_seed(3)
    decoder = tiny()
    data = torch.randn(1, 12, 32)
    mask = torch.ones(1, 12)
    if padding:
        mask[0, 9] = 0
    native = decoder(inputs_embeds=data, attention_mask=mask, use_cache=True).last_hidden_state
    originals = [layer.self_attn for layer in decoder.layers]
    strategy = FixedReuse(mode)
    adapter = CausalReuseAdapter(decoder, strategy, lambda n: TokenLayout(n, patches_per_view=4))
    try:
        cold = decoder(inputs_embeds=data, attention_mask=mask, use_cache=True).last_hidden_state
        torch.testing.assert_close(cold, native, atol=1e-6, rtol=0)
        strategy.warm = True
        saved = strategy.store.keys[2].clone()
        warm = decoder(inputs_embeds=data, attention_mask=mask).last_hidden_state
        torch.testing.assert_close(warm, native[:, strategy.last], atol=1e-6, rtol=0)
        torch.testing.assert_close(
            strategy.store.keys[2][:, :, [1, 3]], saved[:, :, [1, 3]], atol=0, rtol=0
        )
        assert all(k.shape[2] == 12 for k in strategy.store.keys.values())
        for positions, probabilities in strategy.maps:
            assert not probabilities[torch.arange(12)[None, :] > positions[:, None]].any()
            if padding:
                assert not probabilities[:, 9].any()
        assert adapter.state is None and strategy.last[-1] == 11
        assert decoder(inputs_embeds=data).past_key_values is None
    finally:
        adapter.close()
    assert not strategy.store.keys
    assert all(layer.self_attn is original for layer, original in zip(decoder.layers, originals))
    restored = decoder(inputs_embeds=data, attention_mask=mask).last_hidden_state
    torch.testing.assert_close(restored, native, atol=0, rtol=0)
