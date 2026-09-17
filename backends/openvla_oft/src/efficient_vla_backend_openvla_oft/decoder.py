"""Instance-local OFT decoder extension points; no global Transformers patches.

The pinned OFT decoder uses bidirectional SDPA. Methods may change which rows
are computed, inspect selected attention rows, and supply an indexed KV store.
Original sequence positions remain authoritative for RoPE and action readout.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


@dataclass(frozen=True)
class TokenLayout:
    length: int
    patches_per_view: int = 256
    views: int = 2
    proprio_tokens: int = 1
    action_tokens: int = 56

    def __post_init__(self):
        if self.text_end <= self.text_start:
            raise ValueError("The multimodal sequence must contain language tokens")

    @property
    def visual_end(self):
        return 1 + self.patches_per_view * self.views

    @property
    def text_start(self):
        return self.visual_end + self.proprio_tokens

    @property
    def text_end(self):
        return self.length - self.action_tokens - 1

    @property
    def action_start(self):
        return self.text_end

    @property
    def readout_start(self):
        # Preserve the pinned checkpoint's native L1-head slice, including its
        # one-token offset from the action-placeholder range.
        return self.length - self.action_tokens - 2

    def visual_mask(self, positions):
        return (positions >= 1) & (positions < self.visual_end)

    def view_mask(self, positions, view):
        start = 1 + view * self.patches_per_view
        return (positions >= start) & (positions < start + self.patches_per_view)


@dataclass
class DecoderState:
    layout: TokenLayout
    positions: torch.Tensor


class DecoderStrategy:
    """Method-owned decisions. Returned keep masks refer to current query rows."""

    attention_backend = "sdpa"

    def start(self, state: DecoderState) -> None:
        pass

    def before_layer(self, layer: int, state: DecoderState):
        return None

    def probes(self, layer: int, state: DecoderState) -> dict[str, torch.Tensor]:
        """Named original-position query IDs; probabilities retain all key columns."""
        return {}

    def key_positions(self, layer: int, state: DecoderState):
        return state.positions

    def update_kv(self, layer: int, keys, values, state: DecoderState):
        return keys, values

    def after_layer(self, layer: int, state: DecoderState, attention: dict):
        return None

    def finish(self, state: DecoderState) -> None:
        pass


class IndexedKVStore:
    """Overwrite current rows, retain skipped rows; never append across frames."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.keys: dict[int, torch.Tensor] = {}
        self.values: dict[int, torch.Tensor] = {}

    def update(self, layer: int, keys, values, positions, full_length: int):
        if layer not in self.keys:
            if keys.shape[2] != full_length:
                raise ValueError("A KV layer must be initialized with a complete frame")
            expected = torch.arange(full_length, device=positions.device)
            if not torch.equal(positions, expected):
                raise ValueError("Initial KV positions must cover the complete sequence")
            self.keys[layer] = keys.clone()
            self.values[layer] = values.clone()
        else:
            if self.keys[layer].shape[2] != full_length:
                raise ValueError("Sequence layout changed without a cache reset")
            self.keys[layer].index_copy_(2, positions, keys)
            self.values[layer].index_copy_(2, positions, values)
        return self.keys[layer], self.values[layer]


class InspectableAttention(nn.Module):
    """Native SDPA probes, or shared eager probabilities for full-map consumers."""

    def __init__(self, original, owner, layer):
        super().__init__()
        self.q_proj = original.q_proj
        self.k_proj = original.k_proj
        self.v_proj = original.v_proj
        self.o_proj = original.o_proj
        self.rotary_emb = original.rotary_emb
        self.num_heads = original.num_heads
        self.num_key_value_heads = original.num_key_value_heads
        self.num_key_value_groups = original.num_key_value_groups
        self.head_dim = original.head_dim
        self.hidden_size = original.hidden_size
        self.owner = owner  # Plain object, not another registered nn.Module.
        self.layer = layer
        self.last_attention = {}
        self.train(original.training)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        **kwargs,
    ):
        if self.training or output_attentions or use_cache or past_key_value is not None:
            raise ValueError("The method decoder supports inference and its own probes/cache only")
        batch, length, _ = hidden_states.shape
        queries = (
            self.q_proj(hidden_states)
            .view(batch, length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        keys = (
            self.k_proj(hidden_states)
            .view(batch, length, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        values = (
            self.v_proj(hidden_states)
            .view(batch, length, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        cos, sin = self.rotary_emb(values, position_ids)
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)
        state, strategy = self.owner.state, self.owner.strategy
        keys, values = strategy.update_kv(self.layer, keys, values, state)
        keys = repeat_kv(keys, self.num_key_value_groups)
        values = repeat_kv(values, self.num_key_value_groups)
        if attention_mask is not None:
            queries, keys, values = queries.contiguous(), keys.contiguous(), values.contiguous()
        self.last_attention = {}
        probes = strategy.probes(self.layer, state)
        probabilities = None
        if strategy.attention_backend == "eager":
            # Same execution pattern as the reference LlamaAttention: compute
            # probabilities once, then reuse them for both P @ V and statistics.
            logits = torch.matmul(queries, keys.transpose(2, 3)) / self.head_dim**0.5
            if attention_mask is not None:
                logits = logits + attention_mask
            probabilities = F.softmax(logits, dim=-1, dtype=torch.float32).to(queries.dtype)
            output = torch.matmul(probabilities, values)
        for name, requested in probes.items():
            if requested.numel() == 0:
                self.last_attention[name] = queries.new_empty(
                    (0, keys.shape[2]), dtype=torch.float32
                )
                continue
            # Identity is a safe full-row request, not a disabled validity check.
            # It avoids a redundant GPU search/gather/equality synchronization.
            if requested is state.positions:
                row_indices = None
            else:
                row_indices = torch.searchsorted(state.positions, requested).clamp_max(
                    state.positions.numel() - 1
                )
                if not torch.equal(state.positions[row_indices], requested):
                    raise ValueError("An attention probe requested a removed query token")
            if probabilities is not None:
                selected = (
                    probabilities
                    if row_indices is None
                    else probabilities.index_select(2, row_indices)
                )
            else:
                probe_queries = (
                    queries if row_indices is None else queries.index_select(2, row_indices)
                )
                logits = torch.matmul(probe_queries, keys.transpose(2, 3)) / self.head_dim**0.5
                if attention_mask is not None:
                    logits = logits + attention_mask[:, :, :1, :]
                selected = F.softmax(logits, dim=-1, dtype=torch.float32).to(queries.dtype)
            self.last_attention[name] = selected.mean(dim=1, dtype=torch.float32)[0]
        if probabilities is None:
            output = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        output = output.transpose(1, 2).contiguous().view(batch, length, self.hidden_size)
        return self.o_proj(output), None, None


class DecoderAdapter:
    """Attach reversible hooks and attention modules to one LlamaModel instance."""

    def __init__(self, decoder, strategy: DecoderStrategy, layout_factory=TokenLayout):
        if strategy.attention_backend not in {"sdpa", "eager"}:
            raise ValueError("Unsupported inspectable attention backend")
        if getattr(decoder, "_evb_adapter", None) is not None:
            raise ValueError("Only one decoder strategy may be attached")
        if decoder.config._attn_implementation != "sdpa" or decoder.training:
            raise ValueError("The method decoder requires an eval-mode OFT SDPA model")
        self.decoder, self.strategy, self.layout_factory = decoder, strategy, layout_factory
        self.state = None
        self.original_attention = []
        self.handles = []
        decoder._evb_adapter = self
        self.handles.append(decoder.register_forward_pre_hook(self._start, with_kwargs=True))
        for index, layer in enumerate(decoder.layers):
            self.original_attention.append(layer.self_attn)
            layer.self_attn = InspectableAttention(layer.self_attn, self, index)
            self.handles.append(
                layer.register_forward_pre_hook(self._before(index), with_kwargs=True)
            )
            self.handles.append(layer.register_forward_hook(self._after(index), with_kwargs=True))
        self.handles.append(decoder.register_forward_hook(self._finish, with_kwargs=True))

    def _start(self, module, args, kwargs):
        embeds = kwargs.get("inputs_embeds")
        if embeds is None or embeds.shape[0] != 1:
            raise ValueError("Method decoding requires batch-one multimodal inputs_embeds")
        if kwargs.get("past_key_values") is not None or kwargs.get("output_attentions"):
            raise ValueError("External decoder caches/full-attention outputs are unsupported")
        layout = self.layout_factory(embeds.shape[1])
        self.state = DecoderState(layout, torch.arange(layout.length, device=embeds.device))
        self.strategy.start(self.state)
        kwargs = dict(kwargs, use_cache=False, output_attentions=False, return_dict=True)
        return args, kwargs

    def _select(self, hidden, keep):
        if keep is None:
            return hidden
        if keep.dtype != torch.bool or keep.shape != self.state.positions.shape:
            raise ValueError("A token selection must be a boolean mask of current rows")
        protected = ~self.state.layout.visual_mask(self.state.positions)
        if not bool(keep[protected].all()):
            raise ValueError("Methods may not remove non-visual tokens")
        self.state.positions = self.state.positions[keep]
        return hidden[:, keep, :]

    def _before(self, index):
        def hook(module, args, kwargs):
            hidden = self._select(args[0], self.strategy.before_layer(index, self.state))
            positions = self.state.positions
            keys = self.strategy.key_positions(index, self.state)
            mask = kwargs.get("attention_mask")
            if mask is not None:
                # Native OFT broadcasts the final causal-mask row to every query.
                # Here the key and query sets may differ because of indexed reuse.
                mask = mask[:, :, -1:, : self.state.layout.length].index_select(-1, keys)
                mask = mask.expand(-1, -1, positions.numel(), -1)
            kwargs = dict(
                kwargs,
                position_ids=positions.unsqueeze(0),
                cache_position=positions,
                attention_mask=mask,
                use_cache=False,
                past_key_value=None,
            )
            return (hidden, *args[1:]), kwargs

        return hook

    def _after(self, index):
        def hook(module, args, kwargs, output):
            keep = self.strategy.after_layer(index, self.state, module.self_attn.last_attention)
            return (self._select(output[0], keep), *output[1:])

        return hook

    def _finish(self, module, args, kwargs, output):
        self.strategy.finish(self.state)
        if self.state.positions.numel() != self.state.layout.length:
            full = output.last_hidden_state.new_zeros(
                (1, self.state.layout.length, output.last_hidden_state.shape[-1])
            )
            full.index_copy_(1, self.state.positions, output.last_hidden_state)
            output.last_hidden_state = full
            if output.hidden_states is not None:
                output.hidden_states = (*output.hidden_states[:-1], full)
        return output

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for layer, original in zip(self.decoder.layers, self.original_attention):
            layer.self_attn = original
        if getattr(self.decoder, "_evb_adapter", None) is self:
            del self.decoder._evb_adapter
