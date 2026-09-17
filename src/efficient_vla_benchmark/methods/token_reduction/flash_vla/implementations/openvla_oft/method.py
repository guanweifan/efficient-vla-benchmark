"""Exact economy-SVD implementation of the 2025 FlashVLA information score."""

from dataclasses import asdict

import torch
from efficient_vla_backend_openvla_oft.decoder import DecoderAdapter, DecoderStrategy

from efficient_vla_benchmark.core.method import InferenceMethod


def information_scores(features):
    # Exact Eq. 2; no random/low-rank approximation, squared-energy substitution,
    # discarded singular directions, or hidden offline token-selection pass.
    kwargs = {"driver": "gesvd"} if features.is_cuda else {}
    left, singular, _ = torch.linalg.svd(features.float(), full_matrices=False, **kwargs)
    return (left * singular).abs().sum(-1)


class FlashVLA(InferenceMethod, DecoderStrategy):
    name = "flash_vla"

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.active = config.enabled and config.tokens_per_view < 256
        self.adapter = self.handle = None
        self.reset(None)
        if self.active:
            decoder = backend.model.language_model.model
            self.adapter = DecoderAdapter(decoder, self)
            self.handle = decoder.layers[
                config.pruning_layer
            ].self_attn.o_proj.register_forward_hook(self._capture)

    def _capture(self, module, args, output):
        self.attention_features = output

    def reset(self, context):
        self.attention_features = None

    def before_query(self, prepared):
        self.reset(None)
        self.backend.query_diagnostics = {}

    def start(self, state):
        self.layer_sizes = []

    def probes(self, layer, state):
        self.layer_sizes.append(state.positions.numel())
        return {}

    def after_layer(self, layer, state, attention):
        if layer != self.config.pruning_layer:
            return None
        if self.attention_features is None:
            raise RuntimeError("FlashVLA did not receive the actual attention output")
        keep = ~state.layout.visual_mask(state.positions)
        try:
            for view in range(state.layout.views):
                mask = state.layout.view_mask(state.positions, view)
                scores = information_scores(self.attention_features[0, mask])
                ids = mask.nonzero().flatten()[
                    torch.argsort(scores, descending=True, stable=True)[
                        : self.config.tokens_per_view
                    ]
                ]
                keep[ids] = True
        finally:
            self.attention_features = None
        return keep

    def finish(self, state):
        self.backend.query_diagnostics.update(
            visual_tokens_final=int(state.layout.visual_mask(state.positions).sum().item()),
            layer_sequence_lengths=self.layer_sizes,
        )

    def close(self):
        if self.handle:
            self.handle.remove()
        if self.adapter:
            self.adapter.close()
        self.reset(None)

    def metadata(self):
        return {
            "scope": "ICS token-pruning core only; FlashTrigger action reuse NOT implemented",
            "implementation_basis": "paper-based: arXiv:2505.21200v1 section 3.2",
            "config": asdict(self.config),
            "score": "L1 norm of U*singular_values from exact economy SVD, per camera",
            "feature_source": "attention o_proj output before residual and MLP",
            "requires_training": False,
        }
