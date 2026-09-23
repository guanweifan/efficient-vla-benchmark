from dataclasses import asdict

from efficient_vla_backend_cogact.decoder import TokenStrategy

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.attention_probe import NativeAttentionProbe
from efficient_vla_benchmark.model_ops.depth import depth_patch_signals
from efficient_vla_benchmark.model_ops.token_reduction import TokenReduction

from ...merging import ProgressiveDepthMerge


class DepthCachePrimary(InferenceMethod, TokenStrategy):
    name = "depth_cache_primary"
    requires_control_history = True

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.core = ProgressiveDepthMerge(config)
        self.adapter = self.probe = None
        if config.enabled:
            decoder = backend.model_access.decoder
            if config.attention_layer >= len(decoder.layers):
                raise ValueError("Semantic attention layer outside decoder")
            self.adapter = backend.model_access.attach_token_strategy(self)
            try:
                self.probe = NativeAttentionProbe(
                    decoder.layers[config.attention_layer].self_attn,
                    self._request,
                    lambda: self.adapter.state.positions,
                )
            except Exception:
                self.close()
                raise

    def _request(self):
        if self.core.warming:
            state = self.adapter.state
            text = (state.positions >= state.layout.visual_end) & (
                state.positions < state.layout.cognition_position
            )
            return state.positions[text]

    def reset(self, context):
        self.core.reset()
        self.abort()

    def abort(self):
        if self.probe:
            self.probe.clear()

    def before_query(self, prepared):
        self.abort()
        self.backend.query_diagnostics = {}
        if not self.config.enabled:
            return
        observation = prepared["observation"]
        if observation.depth is None:
            raise ValueError("DepthCache requires measured metric depth")
        depth, edges = depth_patch_signals(
            observation.depth,
            self.backend.device,
            crop_area=1.0,
            edge_threshold=self.config.edge_threshold_m,
        )
        self.core.observe_depth(depth, edges)

    def before_layer(self, layer, state):
        if layer == 0:
            visual = state.layout.visual_mask(state.positions)
            anchors, features = self.core.reduce(state.hidden[0, visual])
            keep = ~visual
            keep[visual.nonzero().flatten()[anchors]] = True
            return TokenReduction(keep, features)

    def after_layer(self, layer, state):
        if self.core.warming and layer == self.config.attention_layer:
            try:
                self.core.collect_saliency(
                    self.probe.probabilities[:, state.layout.visual_mask(state.positions)].mean(0)
                )
            finally:
                self.probe.clear()

    def finish(self, state):
        self.backend.query_diagnostics.update(
            visual_tokens_final=state.layout.visual_count(len(state.positions)),
            layer_sequence_lengths=list(self.adapter.layer_sequence_lengths),
            **self.core.diagnostics(),
        )
        self.abort()

    def close(self):
        if self.probe:
            self.probe.close()
            self.probe = None
        if self.adapter:
            self.adapter.close()
            self.adapter = None
        self.reset(None)

    def metadata(self):
        return {
            "scope": "primary-view depth partitions, gradual merge, dual protection and recovery",
            "exclusions": "carrying-dependent scene reinitialization; no wrist view in this model",
            "reference": "arXiv:2603.10469v1 III-B1/B2/B3",
            "inputs": "metric simulator/RGB-D depth only; no grasp or object oracle",
            "semantic_signal": "native final-layer language-to-visual causal attention",
            "geometry": "native resize, no extra crop; nearest metric-depth resampling",
            "position_policy": "latest-member merge anchors preserve causal information order",
            "thresholds": "fixed explicit engineering defaults, not success-tuned author settings",
            "requires_training": False,
            "config": asdict(self.config),
        }
