"""TIES native prefix observation and amortized allocation; no future oracle."""

from dataclasses import asdict, replace

import torch

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.forward_patch import ForwardPatches

from .ranking import UndefinedRankCorrelation, attention_fraction, rank_consistency
from .selection import select_visual


class PrefixComplete(Exception):
    """Local sentinel: the requested native prefix has actually executed."""


class TIESStrategy(InferenceMethod):
    name = "ties"
    requires_control_history = True

    def __init__(self, backend, config, decoder, manifest=None, calibrating=False):
        if not config.pruning_layer < len(decoder.layers) or config.ranking_layers > len(
            decoder.layers
        ):
            raise ValueError("TIES layer settings exceed native decoder geometry")
        self.backend, self.config, self.decoder = backend, config, decoder
        self.calibrating = calibrating
        self.manifest = manifest
        self.active = calibrating or (config.enabled and config.keep_ratio < 1)
        self.adapter = None
        self.patches = ForwardPatches()
        self.native_probes = []
        self.reset(None)

    def install_prefix(self):
        if self.active and not self.calibrating:
            self.patches.wrap(self.decoder, self.wrap_forward)

    def reset(self, context):
        self.control_step = 0
        self.last_refresh = None
        self.tau = None
        self.fraction = None
        self.probing = False
        self.rankings = []
        self.generator = None
        self.stage = {}
        self.layer_lengths = []

    def before_query(self, prepared):
        self.rankings = []
        self.stage = {}
        self.layer_lengths = []
        self.backend.query_diagnostics = {}

    def after_action(self, action, remaining):
        self.control_step += 1
        return False

    def wrap_forward(self, original):
        def forward(*args, **kwargs):
            due = (
                self.last_refresh is None
                or self.control_step - self.last_refresh >= self.config.refresh_steps
            )
            if due:
                initial = self.adapter.state
                self.probing = True
                self.rankings = []
                self.stage["refresh"] = True
                try:
                    probe_kwargs = dict(
                        kwargs,
                        use_cache=False,
                        past_key_values=None,
                        output_hidden_states=False,
                        output_attentions=False,
                    )
                    original(*args, **probe_kwargs)
                    raise RuntimeError("TIES prefix sentinel was not reached")
                except PrefixComplete:
                    if len(self.rankings) != self.config.ranking_layers:
                        raise RuntimeError("Incomplete TIES native rank prefix")
                    try:
                        self.tau = rank_consistency(self.rankings)
                        self.fraction = attention_fraction(
                            self.tau, self.manifest["quantiles"], self.config.mode
                        )
                    except UndefinedRankCorrelation:
                        # The equation has no value for <2 common identities.
                        # Explicit conservative fallback, never an invented tau.
                        self.tau = self.fraction = None
                    self.last_refresh = self.control_step
                finally:
                    self.probing = False
                    self.adapter.state = replace(initial, hidden=None)
                    self.layer_lengths = []
                    if hasattr(self.adapter, "layer_sequence_lengths"):
                        self.adapter.layer_sequence_lengths = []
                    for probe in self.native_probes:
                        probe.clear()
                self.stage["extra_prefix_layers"] = self.config.ranking_layers
            else:
                self.stage.update(refresh=False, extra_prefix_layers=0)
            return original(*args, **kwargs)

        return forward

    def wants_scores(self, layer):
        if self.calibrating or self.probing:
            return layer < self.config.ranking_layers
        return layer == self.config.pruning_layer - 1 and self.fraction is not None

    def before_layer(self, layer, state):
        if not self.probing:
            self.layer_lengths.append(len(state.positions))
        return None

    def consume_attention(self, layer, state, probabilities):
        if self.wants_scores(layer):
            if probabilities is None or probabilities.shape != (
                len(state.positions),
                len(state.positions),
            ):
                raise RuntimeError("TIES requires every native query row and current key")
            visual = state.layout.visual_mask(state.positions)
            scores = probabilities.float().sum(dim=0)[visual]
            if self.calibrating or self.probing:
                if self.config.ranking_topk > len(scores):
                    raise ValueError("TIES ranking top-K exceeds native visual tokens")
                ids = state.positions[visual]
                order = scores.argsort(descending=True, stable=True)[: self.config.ranking_topk]
                self.rankings.append(ids[order].tolist())
            else:
                if self.generator is None:
                    self.generator = torch.Generator(device=scores.device).manual_seed(
                        self.config.uniform_seed
                    )
                chosen, counts = select_visual(
                    scores, self.config.keep_ratio, self.fraction, self.generator
                )
                self.stage.update(counts)
                keep = ~visual
                keep[visual.nonzero().flatten()[chosen]] = True
                return keep
        if self.probing and layer + 1 == self.config.ranking_layers:
            raise PrefixComplete
        return None

    def finish(self, state):
        if self.calibrating:
            if len(self.rankings) != self.config.ranking_layers:
                raise RuntimeError("Incomplete TIES calibration layers")
            self.tau = rank_consistency(self.rankings)
        self.backend.query_diagnostics.update(
            visual_tokens_final=int(state.layout.visual_mask(state.positions).sum()),
            layer_sequence_lengths=list(self.layer_lengths),
            ties=dict(
                self.stage,
                tau=self.tau,
                attention_fraction=self.fraction,
                control_step=self.control_step,
                undefined_tau_dense_fallback=self.tau is None,
            ),
        )

    def abort(self):
        for probe in self.native_probes:
            probe.clear()

    def close(self):
        self.patches.close()
        for probe in self.native_probes:
            probe.close()
        self.native_probes.clear()
        if self.adapter is not None:
            self.adapter.close()
            self.adapter = None
        self.reset(None)

    def metadata(self):
        return dict(
            config=asdict(self.config),
            reference="OpenReview jN98pgTpd8, 10-page 2026-05-26 PDF including Appendix A",
            scope=(
                "native inter-layer top-K Kendall; training quantiles; "
                "hard/soft attention-uniform allocation"
            ),
            conventions=[
                "sum all native query rows; mean heads",
                "stable rank ties",
                "isolated seeded uniform sampling without replacement",
                "prune after block2",
                "undefined tau retains dense tokens",
            ],
            refresh_unit="executed controls; extra prefix fully charged",
            calibration=None
            if self.manifest is None
            else {
                k: self.manifest[k]
                for k in ("statistics_sha256", "samples_sha256", "quantiles", "checkpoint_sha256")
            },
            implementation_basis="independent revision equations, not author-code parity",
        )
