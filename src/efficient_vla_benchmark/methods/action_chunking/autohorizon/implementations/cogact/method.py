"""Observe native DiT action attention without replacing its fused forward."""

from dataclasses import asdict

import torch

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.forward_patch import ForwardPatches

from ...selection import select_prefix


def action_attention(qkv, attention):
    """Softmax over ALL keys before cropping action rows/keys and averaging heads."""
    batch, tokens, channels = qkv.shape
    if batch not in (1, 2) or tokens != 17 or channels % (3 * attention.num_heads):
        raise ValueError("Expected one CogACT observation with condition plus 16 action tokens")
    head_dim = channels // (3 * attention.num_heads)
    packed = qkv[:1].reshape(1, tokens, 3, attention.num_heads, head_dim)
    query, key, _ = packed.permute(2, 0, 3, 1, 4).unbind(0)
    query, key = attention.q_norm(query), attention.k_norm(key)
    full = ((query * attention.scale) @ key.transpose(-2, -1)).softmax(-1)
    return full[0, :, 1:, 1:].mean(0)


class AutoHorizon(InferenceMethod):
    name = "autohorizon"
    requires_control_history = True

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.net = backend.model_access.action_denoiser
        self.patches = ForwardPatches()
        self.handles = []
        self.closed = False
        self.native_steps = backend.parameters.num_ddim_steps
        if (
            self.net.training
            or config.denoising_step > self.native_steps
            or config.max_horizon > 16
            or config.run_length > 16
        ):
            raise ValueError("Unsupported native AutoHorizon geometry or training mode")
        for block in self.net.blocks:
            attn = block.attn
            if any(not hasattr(attn, name) for name in ("qkv", "q_norm", "k_norm", "scale")):
                raise ValueError("AutoHorizon requires the native joint-QKV attention")
        self.reset(None)
        if config.enabled:
            try:
                self.patches.wrap(self.net, self._count)
                for index, block in enumerate(self.net.blocks):
                    self.handles.append(
                        block.attn.qkv.register_forward_hook(self._capture(index, block.attn))
                    )
            except Exception:
                self.close()
                raise

    def _count(self, original):
        def forward(*args, **kwargs):
            if not self.query_open or torch.is_grad_enabled():
                raise RuntimeError("AutoHorizon requires a nonnested inference policy query")
            self.calls += 1
            self.capture_active = self.calls == self.config.denoising_step
            try:
                return original(*args, **kwargs)
            finally:
                self.capture_active = False

        return forward

    def _capture(self, index, attention):
        def hook(module, inputs, output):
            if self.capture_active:
                if index in self.maps:
                    raise RuntimeError("Duplicate DiT attention capture")
                self.maps[index] = action_attention(output.detach(), attention)

        return hook

    def reset(self, context):
        self.maps = {}
        self.calls = 0
        self.capture_active = False
        self.query_open = False
        self.horizon = None
        self.diagnostics = {}

    def before_query(self, prepared):
        if self.closed or self.query_open:
            raise RuntimeError("Closed or unfinished AutoHorizon query")
        self.reset(None)
        self.query_open = True
        self.backend.query_diagnostics = {}

    def after_query(self, actions):
        if self.config.enabled:
            if self.calls != self.native_steps or len(self.maps) != len(self.net.blocks):
                raise RuntimeError("Incomplete native attention capture")
            # The reference divides by total denoising calls before row
            # renormalization. Retain that convention, including its epsilon.
            matrix = torch.stack(list(self.maps.values())).mean(0) / self.calls
            self.horizon, self.diagnostics = select_prefix(matrix, self.config)
            self.backend.query_diagnostics.update(
                denoiser_calls=self.calls,
                attention_probe_step=self.config.denoising_step,
                attention_probe_layers=len(self.maps),
            )
        self.maps = {}
        self.query_open = False

    def select_execution_horizon(self, actions, default_horizon):
        if not self.config.enabled:
            return default_horizon
        if self.horizon is None or self.horizon > len(actions):
            raise RuntimeError("AutoHorizon needs a completed native prediction")
        return self.horizon

    def control_diagnostics(self):
        return dict(
            horizon_policy="adaptive" if self.config.enabled else "native",
            **self.diagnostics,
        )

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.patches.close()
        self.reset(None)
        self.closed = True

    def metadata(self):
        return dict(
            scope="AutoHorizon bidirectional action-attention soft pointers",
            implementation_basis="pinned author bidirectional selector and third-step capture",
            reference_revision="c7504f1756109103f2cfcc2e23f1b1a23841c885",
            config=asdict(self.config),
            requires_training=False,
            execution_horizon_policy="adaptive" if self.config.enabled else "native",
            attention="conditional DiT row; full-key softmax then action crop and head/layer mean",
            sampler="unchanged native DDIM and CFG",
            excluded=["flow-trained checkpoint claims", "extra denoiser calls"],
        )
