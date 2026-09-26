"""Section III-B: selective timm attention updates with full native residual/MLP."""

import math
from dataclasses import asdict

import torch
import torch.nn.functional as F

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.forward_patch import ForwardPatches


class TemporalVisualReuse(InferenceMethod):
    name = "temporal_redundancy_visual"

    def __init__(self, backend, config, encoders, views):
        self.backend, self.config = backend, config
        self.encoders, self.views = encoders, views
        self.active = config.enabled and config.update_ratio < 1
        self.closed = False
        self.patches = ForwardPatches()
        self.handles = []
        self.reset(None)
        if self.active:
            try:
                for name, encoder in encoders.items():
                    if encoder.training or hasattr(encoder, "_evb_temporal_reuse"):
                        raise ValueError("Expected an unused eval-mode native vision encoder")
                    encoder._evb_temporal_reuse = self
                    self.patches.wrap(encoder, lambda original, n=name: self._encoder(original, n))
                    for i, block in enumerate(encoder.blocks):
                        attn = block.attn
                        if not all(
                            hasattr(attn, k)
                            for k in ("qkv", "q_norm", "k_norm", "proj", "fused_attn")
                        ):
                            raise ValueError("Unsupported native timm attention")
                        self.patches.wrap(
                            attn,
                            lambda original, a=attn, n=name, i=i: self._attention(
                                original, a, n, i
                            ),
                        )
            except Exception:
                self.close()
                raise

    def _encoder(self, original, name):
        def forward(*args, **kwargs):
            if not self.query_open or torch.is_grad_enabled() or self.encoders[name].training:
                self.reset(None)
                raise RuntimeError("Temporal reuse requires an inference policy query")
            view = self.calls[name]
            if view >= self.views:
                self.reset(None)
                raise RuntimeError("Unexpected camera count")
            self.current[name] = view
            try:
                result = original(*args, **kwargs)
                self.calls[name] += 1
                return result
            except Exception:
                self.reset(None)
                raise

        return forward

    @staticmethod
    def _split(attn, qkv):
        b, n, total = qkv.shape
        q, k, v = (
            qkv.reshape(b, n, 3, attn.num_heads, total // (3 * attn.num_heads))
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        return attn.q_norm(q), attn.k_norm(k), v

    def _attention(self, original, attn, name, layer):
        def forward(x):
            view = self.current[name]
            key = (name, view, layer)
            if x.ndim != 3 or x.shape[0] != 1:
                raise ValueError("Temporal visual reuse supports batch one")
            if self.query % 2 == 0:
                captured = []
                handle = attn.qkv.register_forward_hook(
                    lambda module, args, output: captured.append(output)
                )
                try:
                    out = original(x)
                finally:
                    handle.remove()
                if len(captured) != 1:
                    raise RuntimeError("Native attention QKV boundary changed")
                _, k, v = self._split(attn, captured[0])
                self.cache[key] = (k.detach(), v.detach(), out.detach())
                self.signatures[key] = (tuple(x.shape), x.dtype, x.device)
                self.computed_rows += x.shape[1]
                return out
            if key not in self.cache:
                raise RuntimeError("Temporal visual reuse missing dense reference")
            old_k, old_v, old_out = self.cache[key]
            if self.signatures[key] != (tuple(x.shape), x.dtype, x.device):
                raise RuntimeError("Visual geometry changed across queries")
            if layer == 0:
                c = x.shape[-1]
                # Only K for all rows is needed to choose the dynamic patches.
                bias = None if attn.qkv.bias is None else attn.qkv.bias[c : 2 * c]
                k_all = F.linear(x, attn.qkv.weight[c : 2 * c], bias)
                k_all = attn.k_norm(
                    k_all.reshape(1, x.shape[1], attn.num_heads, -1).permute(0, 2, 1, 3)
                )
                a = k_all.permute(0, 2, 1, 3).flatten(2).float()
                b = old_k.permute(0, 2, 1, 3).flatten(2).float()
                prefix = self.encoders[name].num_prefix_tokens
                count = math.ceil((x.shape[1] - prefix) * self.config.update_ratio)
                score = F.cosine_similarity(a, b, dim=-1)[0, prefix:]
                ids = torch.argsort(score, stable=True)[:count] + prefix
                ids = torch.cat((torch.arange(prefix, device=x.device), ids)).sort().values
                self.selected[name, view] = ids
            ids = self.selected[name, view]
            q, k, v = self._split(attn, attn.qkv(x.index_select(1, ids)))
            full_k, full_v = old_k.clone(), old_v.clone()
            full_k.index_copy_(2, ids, k)
            full_v.index_copy_(2, ids, v)
            if attn.fused_attn:
                output = F.scaled_dot_product_attention(q, full_k, full_v, dropout_p=0.0)
            else:
                weights = attn.attn_drop(((q * attn.scale) @ full_k.transpose(-2, -1)).softmax(-1))
                output = weights @ full_v
            output = output.transpose(1, 2).reshape(1, ids.numel(), x.shape[-1])
            output = attn.proj_drop(attn.proj(output))
            result = old_out.clone()
            result.index_copy_(1, ids, output)
            self.computed_rows += ids.numel()
            self.reused_rows += x.shape[1] - ids.numel()
            return result

        return forward

    def reset(self, context):
        self.cache, self.current, self.selected, self.signatures = {}, {}, {}, {}
        self.query = 0
        self.query_open = False
        self.calls = {name: 0 for name in self.encoders}

    def before_query(self, prepared):
        if self.closed or self.query_open:
            raise RuntimeError("Closed or unfinished temporal reuse query")
        self.query_open = True
        self.calls = {name: 0 for name in self.encoders}
        self.selected = {}
        self.computed_rows = self.reused_rows = 0
        self.backend.query_diagnostics = {}

    def after_query(self, actions):
        if self.active:
            if any(count != self.views for count in self.calls.values()):
                self.reset(None)
                raise RuntimeError("Native vision camera traversal incomplete")
            self.backend.query_diagnostics.update(
                vision_attention_computed_rows=self.computed_rows,
                vision_attention_reused_rows=self.reused_rows,
                vision_cache_refresh=self.query % 2 == 0,
                vision_cache_entries=len(self.cache),
            )
        self.query += 1
        self.query_open = False

    def close(self):
        self.patches.close()
        for encoder in self.encoders.values():
            if getattr(encoder, "_evb_temporal_reuse", None) is self:
                del encoder._evb_temporal_reuse
        self.reset(None)
        self.closed = True

    def metadata(self):
        return dict(
            scope="Reducing Temporal Redundancy section III-B visual attention reuse core",
            excluded=["two-step flow sampling", "trained flow calibration adapter"],
            implementation_basis="paper equations 3-5; independent timm binding",
            schedule="first and alternate policy queries dense; intervening queries selective",
            selection="first-block normalized-key cosine, per encoder/view, stable patch ties",
            protected="all prefix/register tokens; native positions, residuals and full-token MLP",
            cache_lifetime="one two-query pair; isolated per encoder/camera/layer",
            requires_training=False,
            config=asdict(self.config),
        )
