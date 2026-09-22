"""FlashVLA ICS on the unmodified native causal CogACT attention output."""

from dataclasses import asdict

import torch
from efficient_vla_backend_cogact.decoder import TokenStrategy

from efficient_vla_benchmark.core.method import InferenceMethod

from ...scoring import information_scores


class FlashVLA(InferenceMethod, TokenStrategy):
    name = "flash_vla"

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.active = (
            config.enabled and config.tokens_per_view < backend.model_access.patches_per_view
        )
        self.adapter = self.handle = None
        self.reset(None)
        if self.active:
            decoder = backend.model_access.decoder
            if config.pruning_layer >= len(decoder.layers) - 1:
                raise ValueError("ICS must prune before at least one remaining decoder layer")
            self.adapter = backend.model_access.attach_token_strategy(self)
            try:
                self.handle = decoder.layers[
                    config.pruning_layer
                ].self_attn.o_proj.register_forward_hook(self._capture)
            except Exception:
                self.close()
                raise

    def _capture(self, module, args, output):
        self.attention_features = output

    def reset(self, context):
        self.attention_features = None

    def before_query(self, prepared):
        self.reset(None)
        self.backend.query_diagnostics = {
            "visual_tokens_final": self.backend.model_access.patches_per_view
        }

    def after_layer(self, layer, state):
        if layer != self.config.pruning_layer:
            return None
        if self.attention_features is None:
            raise RuntimeError("ICS did not observe the native attention output")
        visual = state.layout.visual_mask(state.positions)
        try:
            scores = information_scores(self.attention_features[0, visual], self.config.svd_driver)
            ids = visual.nonzero().flatten()[
                torch.argsort(scores, descending=True, stable=True)[: self.config.tokens_per_view]
            ]
            keep = ~visual
            keep[ids] = True
            return keep
        finally:
            self.attention_features = None

    def finish(self, state):
        self.backend.query_diagnostics.update(
            visual_tokens_final=state.layout.visual_count(state.positions.numel()),
            layer_sequence_lengths=list(self.adapter.layer_sequence_lengths),
        )

    def abort(self):
        self.attention_features = None

    def close(self):
        if self.handle:
            self.handle.remove()
            self.handle = None
        if self.adapter:
            self.adapter.close()
            self.adapter = None
        self.reset(None)

    def metadata(self):
        return {
            "scope": "ICS token-pruning core only; action reuse is NOT implemented",
            "implementation_basis": "paper equation: arXiv:2505.21200v1 section 3.2",
            "score": "L1 norm of U*singular_values from complete economy SVD",
            "feature_source": "native attention o_proj output before residual and MLP",
            "attention_context": "causal visual prefix; later language is not visible here",
            "position_policy": "original positions and native causal visibility",
            "svd_driver": self.config.svd_driver,
            "requires_training": False,
            "config": asdict(self.config),
        }
