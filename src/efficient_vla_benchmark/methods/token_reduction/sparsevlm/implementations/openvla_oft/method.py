from dataclasses import asdict

from efficient_vla_backend_openvla_oft.decoder import DecoderAdapter, DecoderStrategy

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.token_reduction import TokenReduction

from ...recycling import prune_and_recycle, text_raters


class SparseVLM(InferenceMethod, DecoderStrategy):
    name = "sparsevlm"

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.adapter = None
        self.reset(None)
        if config.enabled:
            decoder = backend.model.language_model.model
            if max(config.pruning_layers) >= len(decoder.layers) - 1:
                raise ValueError("SparseVLM pruning must leave a subsequent decoder layer")
            self.adapter = DecoderAdapter(decoder, self)

    def reset(self, context):
        self.raters = None
        self.layer_sizes, self.stages = [], []

    def before_query(self, prepared):
        self.reset(None)
        self.backend.query_diagnostics = {}

    def before_layer(self, layer, state):
        if layer == 0:
            visual = state.layout.visual_mask(state.positions)
            text = (state.positions >= state.layout.text_start) & (
                state.positions < state.layout.text_end
            )
            self.raters = state.positions[text][
                text_raters(state.hidden[0, visual], state.hidden[0, text])
            ]

    def probes(self, layer, state):
        self.layer_sizes.append(state.positions.numel())
        return {"raters": self.raters} if layer in self.config.pruning_layers else {}

    def after_layer(self, layer, state, attention):
        if layer not in self.config.pruning_layers:
            return
        visual = state.layout.visual_mask(state.positions)
        visual_ids = visual.nonzero().flatten()
        ids, features, stats = prune_and_recycle(
            state.hidden[0, visual],
            attention["raters"][:, visual],
            state.positions[visual],
            self.config,
        )
        self.stages.append({"layer": layer, **stats})
        keep = ~visual
        keep[visual_ids[ids]] = True
        return TokenReduction(keep, features)

    def finish(self, state):
        self.backend.query_diagnostics.update(
            visual_tokens_final=int(state.layout.visual_mask(state.positions).sum()),
            layer_sequence_lengths=self.layer_sizes,
            sparsevlm_stages=self.stages,
            text_raters=int(self.raters.numel()),
        )
        self.raters = None

    def close(self):
        if self.adapter:
            self.adapter.close()
            self.adapter = None
        self.reset(None)

    def metadata(self):
        return {
            "scope": "SparseVLM v1 equations 6-11: raters, rank adaptation, density recycling",
            "reference": "arXiv:2410.04417v1; not SparseVLM+ or fixed-budget v1.5",
            "reconstruction": "sum, not mean; latest-member native position per cluster",
            "preset": "fixed engineering rank scale/tolerance; no LIBERO success fitting",
            "text": "native language tokens only, excluding proprio/action placeholders",
            "attention": "bidirectional native OFT selected-rater probes",
            "requires_training": False,
            "config": asdict(self.config),
        }
