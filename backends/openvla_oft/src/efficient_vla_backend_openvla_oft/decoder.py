"""Instance-local OFT decoder extension points; no global Transformers patches.

The pinned OFT decoder uses bidirectional SDPA. Methods may change which rows
are computed, inspect selected attention rows, and supply an indexed KV store.
Original sequence positions remain authoritative for RoPE and action readout.
"""

from dataclasses import dataclass

import torch

from efficient_vla_benchmark.model_ops.decoder import IndexedKVStore as IndexedKVStore
from efficient_vla_benchmark.model_ops.decoder import InspectableAttention as InspectableAttention


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
    # Current layer input/output, before the corresponding strategy callback.
    # A strategy must not retain this full tensor across queries.
    hidden: torch.Tensor | None = None


@dataclass(frozen=True)
class TokenReduction:
    """Select rows and optionally replace retained visual features for merging.

    keep refers to CURRENT rows. visual_features has shape [retained_visual, D]
    in their original-position order; nonvisual features cannot be replaced.
    The surviving positions remain the anchors for RoPE and the native readout.
    """

    keep: torch.Tensor
    visual_features: torch.Tensor | None = None


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

    def _select(self, hidden, selection):
        replacement = None
        if isinstance(selection, TokenReduction):
            keep, replacement = selection.keep, selection.visual_features
        else:
            keep = selection
        if keep is None:
            return hidden
        if keep.dtype != torch.bool or keep.shape != self.state.positions.shape:
            raise ValueError("A token selection must be a boolean mask of current rows")
        protected = ~self.state.layout.visual_mask(self.state.positions)
        if not bool(keep[protected].all()):
            raise ValueError("Methods may not remove non-visual tokens")
        self.state.positions = self.state.positions[keep]
        selected = hidden[:, keep, :]
        if replacement is not None:
            visual_ids = self.state.layout.visual_mask(self.state.positions).nonzero().flatten()
            if (
                replacement.shape != (visual_ids.numel(), hidden.shape[-1])
                or replacement.device != hidden.device
                or replacement.dtype != hidden.dtype
            ):
                raise ValueError("Merged features must match the retained visual rows and dtype")
            selected.index_copy_(1, visual_ids, replacement.unsqueeze(0))
        return selected

    def _before(self, index):
        def hook(module, args, kwargs):
            self.state.hidden = args[0]
            hidden = self._select(args[0], self.strategy.before_layer(index, self.state))
            self.state.hidden = hidden
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
            self.state.hidden = output[0]
            keep = self.strategy.after_layer(index, self.state, module.self_attn.last_attention)
            hidden = self._select(output[0], keep)
            self.state.hidden = hidden
            return (hidden, *output[1:])

        return hook

    def _finish(self, module, args, kwargs, output):
        self.strategy.finish(self.state)
        self.state.hidden = None
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
        self.state = None
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for layer, original in zip(self.decoder.layers, self.original_attention):
            layer.self_attn = original
        if getattr(self.decoder, "_evb_adapter", None) is self:
            del self.decoder._evb_adapter
