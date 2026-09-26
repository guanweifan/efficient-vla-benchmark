"""Environment-triggered dynamic channels with interleaved pruning work."""

from dataclasses import asdict

import torch

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.forward_patch import ForwardPatches
from efficient_vla_benchmark.model_ops.indexed_linear import indexed_gate_up, indexed_linear

from .artifact import load_artifact
from .attention import sparse_attention
from .selection import SparsityTrigger, importance, select_channels


class EcoVLA(InferenceMethod):
    name = "eco_vla"
    requires_control_history = True

    def __init__(self, backend, decoder, vision, config, resources, backend_id, *, causal):
        self.backend, self.decoder, self.config = backend, decoder, config
        self.causal = causal
        self.active = config.enabled and config.keep_ratio < 1
        self.patches = ForwardPatches()
        self.handles, self.transposed = [], []
        self.buffers, self.pending = {}, None
        self.stream = None
        self.capacity = 0
        self.artifact_digest = None
        self.initial = {}
        self.trigger = SparsityTrigger(config.trigger_quantile, config.history_window)
        self.dense = True
        self.updates = 0
        if not self.active:
            return
        device = next(decoder.parameters()).device
        if device.type != "cuda" or decoder.training:
            raise ValueError("EcoVLA requires an eval-mode CUDA decoder")
        self.device = device
        self.dtype = next(decoder.parameters()).dtype
        self.depth = len(decoder.layers)
        self.head_dim = decoder.layers[0].self_attn.head_dim
        self.widths = {
            "mlp": decoder.layers[0].mlp.down_proj.in_features,
            "attention": decoder.layers[0].self_attn.o_proj.in_features,
        }
        for layer in decoder.layers:
            attn = layer.self_attn
            if attn.num_heads != attn.num_key_value_heads or attn.head_dim != self.head_dim:
                raise ValueError("EcoVLA initial bindings require uniform coupled MHA heads")
            if layer.mlp.down_proj.in_features != self.widths["mlp"]:
                raise ValueError("EcoVLA needs uniform layer widths for batched statistics")
            for linear in (
                attn.q_proj,
                attn.k_proj,
                attn.v_proj,
                attn.o_proj,
                layer.mlp.gate_proj,
                layer.mlp.up_proj,
                layer.mlp.down_proj,
            ):
                if linear.bias is not None or not linear.weight.is_contiguous():
                    raise ValueError("EcoVLA requires native contiguous bias-free linear weights")
        initial, self.artifact_digest = load_artifact(
            backend, decoder, resources.artifact_path, config, backend_id
        )
        self.initial = {name: value.to(device) for name, value in initial.items()}
        self.weight_norm = {}
        for name, attribute in (("mlp", "mlp.down_proj"), ("attention", "self_attn.o_proj")):
            # Eq. 11's norm is sqrt(sum W^4), not the Wanda L2 column norm.
            self.weight_norm[name] = torch.stack(
                [
                    layer.get_submodule(attribute)
                    .weight.detach()
                    .float()
                    .square()
                    .square()
                    .sum(0)
                    .sqrt()
                    for layer in decoder.layers
                ]
            )
        self.stream = torch.cuda.Stream(device=device)
        self.initial_indices = self._select(
            {name: self.weight_norm[name] * self.initial[name] for name in self.widths}
        )
        try:
            for index, layer in enumerate(decoder.layers):
                for linear in (layer.mlp.down_proj, layer.self_attn.o_proj):
                    linear.weight.data = linear.weight.data.t().contiguous().t()
                    self.transposed.append(linear)
                self.handles.append(
                    layer.mlp.down_proj.register_forward_pre_hook(self._capture("mlp", index))
                )
                self.handles.append(
                    layer.self_attn.o_proj.register_forward_pre_hook(
                        self._capture("attention", index)
                    )
                )
                self.patches.wrap(layer.mlp, self._mlp(index, layer.mlp))
                self.patches.wrap(layer.self_attn, self._attention(index, layer.self_attn))
            self.handles.append(vision.register_forward_hook(self._vision))
            self.handles.append(
                decoder.register_forward_pre_hook(self._begin_decoder, with_kwargs=True)
            )
            self.handles.append(decoder.register_forward_hook(self._end_decoder, always_call=True))
            self.reset(None)
        except Exception:
            self.close()
            raise

    def _select(self, scores):
        return {
            "mlp": select_channels(scores["mlp"], self.config.keep_ratio, alignment=64),
            "attention": select_channels(
                scores["attention"], self.config.keep_ratio, group_size=self.head_dim
            ),
        }

    def _join(self):
        if self.stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)

    def reset(self, context):
        if self.stream is not None:
            self.stream.synchronize()
        self.trigger.reset()
        self.pending = None
        self.dense = True
        self.updates = 0
        self.inflight = False
        if self.active:
            self.history = {name: value.clone() for name, value in self.initial.items()}
            self.indices = self.initial_indices

    def before_query(self, prepared):
        if not self.active:
            return
        if self.inflight:
            raise RuntimeError("Reset EcoVLA after a failed/incomplete policy query")
        self._join()
        self.inflight = True
        self.backend.query_diagnostics = {}

    def _vision(self, module, args, output):
        if not torch.is_tensor(output) or output.ndim != 3 or output.shape[0] != 1:
            raise ValueError("Expected native batch-one visual encoder token features")
        self.dense, similarity = self.trigger.update(output)
        self.backend.query_diagnostics.update(
            eco_dense_refresh=self.dense, eco_visual_similarity=similarity
        )

    def _begin_decoder(self, module, args, kwargs):
        if not self.dense:
            return
        hidden = kwargs.get("inputs_embeds")
        if hidden is None or hidden.shape[0] != 1:
            raise ValueError("EcoVLA requires native batch-one multimodal prefill")
        self.length = hidden.shape[1]
        if self.length > self.capacity:
            self._join()
            self.capacity = ((self.length + 63) // 64) * 64
            self.buffers = {
                name: torch.empty(
                    (self.depth, self.capacity, width), dtype=self.dtype, device=self.device
                )
                for name, width in self.widths.items()
            }
        self.captured = set()

    def _capture(self, name, index):
        def hook(module, args):
            if not self.dense:
                return
            value = args[0]
            if value.shape != (1, self.length, self.widths[name]):
                raise ValueError("Dense channel feature geometry changed inside a query")
            if (name, index) in self.captured:
                raise RuntimeError("Each projection must be captured once per dense prefill")
            self.captured.add((name, index))
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.stream):
                self.buffers[name][index, : self.length].copy_(value[0])
                value.record_stream(self.stream)

        return hook

    def _end_decoder(self, module, args, result):
        if result is None:
            self.pending = None
            return
        if not self.dense:
            return
        if len(self.captured) != self.depth * 2:
            raise RuntimeError("Incomplete dense statistics cannot update EcoVLA masks")
        # All copies are already ordered on the pruning stream. Batched metric
        # work now overlaps the native action head, never a served extra policy.
        with torch.cuda.stream(self.stream):
            scores, updated = {}, {}
            for name in self.widths:
                energy = self.buffers[name][:, : self.length].float().square().sum(1)
                scores[name], updated[name] = importance(
                    self.weight_norm[name],
                    self.history[name],
                    energy,
                    self.config.temporal_inertia,
                    self.config.history_momentum,
                )
            self.pending = updated, self._select(scores)

    def _mlp(self, index, module):
        def factory(original):
            def call(x):
                if self.dense:
                    return original(x)
                channels = self.indices["mlp"][index]
                middle = indexed_gate_up(
                    x, module.gate_proj.weight, module.up_proj.weight, channels
                )
                return indexed_linear(middle, module.down_proj.weight, columns=channels)

            return call

        return factory

    def _attention(self, index, module):
        def factory(original):
            def call(hidden_states, **kwargs):
                if self.dense:
                    return original(hidden_states, **kwargs)
                return sparse_attention(
                    module,
                    hidden_states,
                    self.indices["attention"][index],
                    causal=self.causal,
                    **kwargs,
                )

            return call

        return factory

    def after_query(self, actions):
        if not self.active:
            return
        self._join()
        if self.dense:
            if self.pending is None:
                raise RuntimeError("Dense EcoVLA query did not produce complete statistics")
            self.history, self.indices = self.pending
            self.pending = None
            self.updates += 1
        self.inflight = False
        self.backend.query_diagnostics.update(
            eco_updates=self.updates,
            eco_mlp_channels=self.widths["mlp"] if self.dense else self.indices["mlp"].shape[1],
            eco_attention_heads=(
                self.widths["attention"] if self.dense else self.indices["attention"].shape[1]
            )
            // self.head_dim,
        )

    def close(self):
        if self.stream is not None:
            self.stream.synchronize()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.patches.close()
        for linear in self.transposed:
            linear.weight.data = linear.weight.data.contiguous()
        self.transposed.clear()
        self.buffers.clear()
        self.pending = None
        self.trigger.reset()
        self.stream = None

    def metadata(self):
        return {
            "scope": "EcoVLA adaptive channel/head pruning with interleaved execution",
            "reference": "arXiv:2602.00780v1 Sections 3-4, equations 5-11",
            "implementation": "independent paper-based implementation; no author-code parity claim",
            "granularity": "coupled FFN channels and whole attention heads; no layer skipping",
            "kernel": "indexed Triton GEMM; fused gate/up/SiLU; column-major output weights",
            "mask_timing": "trigger frame dense; next query uses new mask; all pruning work timed",
            "port_conventions": (
                "first two queries dense; partial history then fixed window; "
                "group L2 head scores; FFN budget rounded upward to 64"
            ),
            "calibration_artifact_sha256": self.artifact_digest,
            "requires_training": False,
            "exclusions": "composition with token pruning or caching, other VLA families",
            "config": asdict(self.config),
        }
