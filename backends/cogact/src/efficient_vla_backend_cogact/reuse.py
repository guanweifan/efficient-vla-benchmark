"""Indexed cross-frame KV reuse with original causal query/key positions.

The one-token native cognition-generation call does not need an autoregressive
append cache. This adapter substitutes bounded per-frame storage, retaining the
native Llama blocks/norms and the protected last cognition row.
"""

import torch

from efficient_vla_benchmark.model_ops.decoder import InspectableAttention

from .decoder import CausalTokenAdapter, TokenLayout, TokenStrategy


class ReuseStrategy(TokenStrategy):
    attention_backend = "sdpa"

    def key_positions(self, layer, state):
        return state.positions

    def update_kv(self, layer, keys, values, state):
        return keys, values

    def probes(self, layer, state):
        return {}

    def after_layer(self, layer, state, attention):
        return None


class CausalReuseAdapter(CausalTokenAdapter):
    def __init__(self, decoder, strategy, layout_factory=TokenLayout):
        if strategy.attention_backend not in {"sdpa", "eager"}:
            raise ValueError("Unsupported attention inspection backend")
        self.original_attention = []
        self.causal_sdpa = False
        super().__init__(decoder, strategy, layout_factory)
        try:
            for index, layer in enumerate(decoder.layers):
                self.original_attention.append(layer.self_attn)
                layer.self_attn = InspectableAttention(layer.self_attn, self, index)
        except Exception:
            self.close()
            raise

    def _start(self, module, args, kwargs):
        super()._start(module, args, kwargs)
        # Native predict_action generates only one token and reads the final
        # hidden cognition row. No external/append cache is needed or accepted.
        return args, dict(kwargs, use_cache=False, output_attentions=False, return_dict=True)

    def _before(self, index):
        def hook(module, args, kwargs):
            state = self.state
            state.hidden = args[0]
            hidden = self._select(args[0], self.strategy.before_layer(index, state))
            state.hidden = hidden
            positions = state.positions
            keys = self.strategy.key_positions(index, state)
            mask = kwargs.get("attention_mask")
            self.causal_sdpa = (
                mask is None and positions.numel() == keys.numel() == state.layout.length
            )
            if mask is None:
                future = keys[None, :] > positions[:, None]
                mask = hidden.new_zeros((1, 1, positions.numel(), keys.numel()))
                mask.masked_fill_(future[None, None], torch.finfo(hidden.dtype).min)
            else:
                if mask.ndim != 4 or mask.shape[-2] != state.layout.length:
                    raise ValueError("Expected each original query's native causal/padding mask")
                mask = mask.index_select(-2, positions).index_select(-1, keys)
            self.layer_sequence_lengths.append(positions.numel())
            return (hidden, *args[1:]), dict(
                kwargs,
                attention_mask=mask,
                position_ids=positions[None],
                cache_position=positions,
                use_cache=False,
                past_key_value=None,
            )

        return hook

    def _after(self, index):
        def hook(module, args, kwargs, output):
            state = self.state
            state.hidden = output[0]
            try:
                keep = self.strategy.after_layer(index, state, module.self_attn.last_attention)
                hidden = self._select(output[0], keep)
                state.hidden = hidden
                return (hidden, *output[1:])
            finally:
                module.self_attn.last_attention = {}

        return hook

    def close(self):
        super().close()
        for layer, original in zip(self.decoder.layers, self.original_attention):
            layer.self_attn = original
        self.original_attention.clear()
        self.causal_sdpa = False
