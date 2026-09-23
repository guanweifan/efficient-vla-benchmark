"""Causal token selection for native CogACT prefill, without replacing attention.

Only visual rows may be removed. Native attention modules, caches, generation,
normalization and the final cognition readout remain in use. This API deliberately
does not advertise cross-frame KV reuse or autoregressive cached decoding.
"""

from dataclasses import dataclass

import torch

from efficient_vla_benchmark.model_ops.token_reduction import TokenReduction


@dataclass(frozen=True)
class TokenLayout:
    length: int
    patches_per_view: int = 256
    views: int = 1

    def __post_init__(self):
        if self.views != 1 or self.patches_per_view < 1 or self.length <= self.visual_end + 1:
            raise ValueError("Expected one visual view followed by prompt and cognition tokens")

    @property
    def visual_end(self):
        return 1 + self.patches_per_view

    @property
    def cognition_position(self):
        return self.length - 1

    def visual_mask(self, positions):
        return (positions >= 1) & (positions < self.visual_end)

    def visual_count(self, retained_length):
        return retained_length - (self.length - self.patches_per_view)


@dataclass
class DecoderState:
    layout: TokenLayout
    positions: torch.Tensor
    hidden: torch.Tensor | None = None


class TokenStrategy:
    def start(self, state):
        pass

    def before_layer(self, layer, state):
        return None

    def after_layer(self, layer, state):
        return None

    def finish(self, state):
        pass

    def abort(self):
        pass


class CausalTokenAdapter:
    """Reversible instance hooks; no custom attention kernel or OFT dependency."""

    def __init__(self, decoder, strategy, layout_factory=TokenLayout):
        if decoder.training or decoder.config._attn_implementation != "sdpa":
            raise ValueError("Token pruning requires native eval-mode causal SDPA")
        if getattr(decoder, "_evb_token_adapter", None) is not None:
            raise ValueError("Only one token strategy may be attached")
        self.decoder, self.strategy, self.layout_factory = decoder, strategy, layout_factory
        self.state = None
        self.layer_sequence_lengths = []
        self.handles = []
        decoder._evb_token_adapter = self
        try:
            self.handles.append(decoder.register_forward_pre_hook(self._start, with_kwargs=True))
            for index, layer in enumerate(decoder.layers):
                self.handles.append(
                    layer.register_forward_pre_hook(self._before(index), with_kwargs=True)
                )
                self.handles.append(
                    layer.register_forward_hook(self._after(index), with_kwargs=True)
                )
            self.handles.append(
                decoder.register_forward_hook(self._finish, with_kwargs=True, always_call=True)
            )
        except Exception:
            self.close()
            raise

    def _start(self, module, args, kwargs):
        embeds = kwargs.get("inputs_embeds")
        if embeds is None or embeds.ndim != 3 or embeds.shape[0] != 1:
            raise ValueError("Token pruning requires batch-one multimodal inputs_embeds")
        if self.state is not None:
            raise RuntimeError("Nested token-adapted forwards are not supported")
        if kwargs.get("past_key_values") is not None:
            raise ValueError("Only fresh CogACT prefill is supported, not cached decoding")
        if kwargs.get("output_attentions"):
            raise ValueError("Full attention outputs are not part of the token-selection contract")
        layout = self.layout_factory(embeds.shape[1])
        positions = torch.arange(layout.length, device=embeds.device)
        supplied = kwargs.get("position_ids")
        if supplied is not None and not torch.equal(supplied, positions[None]):
            raise ValueError("Unexpected non-native input positions")
        self.state = DecoderState(layout, positions)
        self.layer_sequence_lengths = []
        self.strategy.start(self.state)

    def _select(self, hidden, selection):
        if isinstance(selection, TokenReduction):
            keep, replacement = selection.keep, selection.visual_features
        else:
            keep, replacement = selection, None
        if keep is None:
            return hidden
        state = self.state
        if (
            not isinstance(keep, torch.Tensor)
            or keep.dtype != torch.bool
            or keep.device != hidden.device
            or keep.shape != state.positions.shape
        ):
            raise ValueError("Selection must be a device-local boolean mask of current rows")
        visual = state.layout.visual_mask(state.positions)
        if not bool(keep[~visual].all()) or not bool(keep[visual].any()):
            raise ValueError("Keep all nonvisual tokens and at least one visual token")
        state.positions = state.positions[keep]
        selected = hidden[:, keep, :]
        if replacement is not None:
            visual_ids = state.layout.visual_mask(state.positions).nonzero().flatten()
            if (
                replacement.shape != (visual_ids.numel(), hidden.shape[-1])
                or replacement.device != hidden.device
                or replacement.dtype != hidden.dtype
                or not bool(torch.isfinite(replacement).all())
            ):
                raise ValueError("Merged features must match retained visual rows and dtype")
            selected.index_copy_(1, visual_ids, replacement.unsqueeze(0))
        return selected

    def _before(self, index):
        def hook(module, args, kwargs):
            state = self.state
            state.hidden = args[0]
            hidden = self._select(args[0], self.strategy.before_layer(index, state))
            state.hidden = hidden
            positions = state.positions
            mask = kwargs.get("attention_mask")
            if mask is not None:
                if mask.ndim != 4 or mask.shape[-2] != state.layout.length:
                    raise ValueError("Expected the native full-query causal/padding mask")
                if mask.shape[-1] < state.layout.length:
                    raise ValueError("Native mask does not cover the original visual sequence")
                # Preserve EACH query's causal visibility; never broadcast the final row.
                mask = mask.index_select(-2, positions).index_select(-1, positions)
            # With no explicit mask, native LlamaSdpaAttention uses a causal triangle.
            # Retained positions stay ordered, so this is the same visibility relation.
            kwargs = dict(
                kwargs,
                attention_mask=mask,
                position_ids=positions[None],
                cache_position=positions,
            )
            self.layer_sequence_lengths.append(hidden.shape[1])
            return (hidden, *args[1:]), kwargs

        return hook

    def _after(self, index):
        def hook(module, args, kwargs, output):
            self.state.hidden = output[0]
            hidden = self._select(output[0], self.strategy.after_layer(index, self.state))
            self.state.hidden = hidden
            return (hidden, *output[1:])

        return hook

    def _finish(self, module, args, kwargs, output):
        try:
            if output is not None and self.state is not None:
                self.strategy.finish(self.state)
            else:
                self.strategy.abort()
        finally:
            if self.state is not None:
                self.state.hidden = None
            self.state = None
        # No zero-padding or action-slot reconstruction: the protected cognition
        # token remains LAST and native generate/predict_action reads that token.

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if getattr(self.decoder, "_evb_token_adapter", None) is self:
            del self.decoder._evb_token_adapter
        self.state = None
        self.strategy.abort()
