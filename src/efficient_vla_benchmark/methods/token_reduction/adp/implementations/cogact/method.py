"""Released ADP Q/K pruning and observed-EEF gating on native CogACT."""

from dataclasses import asdict

from efficient_vla_backend_cogact.decoder import TokenStrategy

from efficient_vla_benchmark.core.method import InferenceMethod

from ...motion import MotionGate
from ...scoring import embedding_scores, view_quotas


class ADP(InferenceMethod, TokenStrategy):
    name = "adp"

    def __init__(self, backend, config):
        from .api import preflight

        preflight(config, None)
        self.backend, self.config = backend, config
        self.active = config.enabled and config.keep_ratio < 1
        self.requires_control_history = self.active and config.dynamic
        self.adapter = None
        self.reset(None)
        if self.active:
            self.adapter = backend.model_access.attach_token_strategy(self)

    def reset(self, context):
        self.gate = MotionGate(self.config)
        self.queries = self.pruned_queries = 0
        self.delta = 0.0
        self.should_prune = False

    def observe(self, observation):
        if self.requires_control_history:
            self.gate.observe(observation.proprio[:3])

    def before_query(self, prepared):
        if self.requires_control_history:
            if self.gate.last is None:
                raise RuntimeError("ADP requires observed EEF history from the common controller")
            if self.queries:
                self.delta = self.gate.close_window()
        self.should_prune = self.active and (not self.config.dynamic or self.gate.state)
        self.queries += 1
        self.pruned_queries += int(self.should_prune)
        self.backend.query_diagnostics = {
            "pruned": self.should_prune,
            "observed_motion": self.delta,
            "query_count": self.queries,
            "pruned_queries": self.pruned_queries,
            "visual_tokens_final": self.backend.model_access.patches_per_view,
        }

    def before_layer(self, layer, state):
        if layer != 0 or not self.should_prune:
            return None
        visual = state.layout.visual_mask(state.positions)
        # Match the author's all-non-action input embedding rule. CogACT has no
        # action placeholders: native BOS, prompt, separator and cognition-EOS
        # INPUT embeddings are included, never a contextualized action-query proxy.
        hidden = state.hidden
        scores = embedding_scores(
            self.backend.model_access.decoder.layers[0], hidden[:, ~visual], hidden[:, visual]
        )
        quota = view_quotas(
            round(scores.numel() * self.config.keep_ratio),
            self.config.view_weights,
            state.layout.patches_per_view,
        )[0]
        ids = visual.nonzero().flatten()[scores.topk(quota).indices]
        keep = ~visual
        keep[ids] = True
        return keep

    def finish(self, state):
        self.backend.query_diagnostics.update(
            visual_tokens_final=state.layout.visual_count(state.positions.numel()),
            layer_sequence_lengths=list(self.adapter.layer_sequence_lengths),
        )

    def close(self):
        if self.adapter:
            self.adapter.close()
            self.adapter = None
        self.reset(None)

    def metadata(self):
        return {
            "scope": "released layer-zero Q/K selection and observed-EEF motion gating",
            "source_commit": "d7094b09a4996847772c1fa975f09d863e1b759a",
            "scoring_layer": 0,
            "score_inputs": "all native nonvisual input embeddings; no action-attention proxy",
            "position_policy": "original positions and native causal visibility",
            "extra_observation_inputs": ["observed_eef_xyz"]
            if self.requires_control_history
            else [],
            "temporal_sampling": "executed chunk windows, not unexecuted predicted actions",
            "requires_training": False,
            "config": asdict(self.config),
        }
