"""Read selected native Llama attention rows without changing its forward result."""

import torch
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


class NativeAttentionProbe:
    """Instance-local, fresh-prefill Q/K inspection with native RoPE and visibility.

    ``request`` returns original query positions (or None), and ``positions``
    returns the ordered original positions of the current sequence. Probabilities
    are normalized over ALL current keys, then averaged across heads in FP32.
    No V projection, output kernel, cache or action readout is replaced.
    """

    def __init__(self, attention, request, positions, *, per_head=False, gravity_correction=False):
        self.attention, self.request, self.positions = attention, request, positions
        self.per_head, self.gravity_correction = per_head, gravity_correction
        self.handles = []
        self.clear()
        try:
            self.handles = [
                attention.register_forward_pre_hook(self._start, with_kwargs=True),
                attention.q_proj.register_forward_hook(self._query),
                attention.k_proj.register_forward_hook(self._key),
                attention.register_forward_hook(self._finish, with_kwargs=True, always_call=True),
            ]
        except Exception:
            self.close()
            raise

    def clear(self):
        self.requested = self.q = self.k = self.probabilities = None

    def _start(self, module, args, kwargs):
        self.clear()
        requested = self.request()
        if requested is None or requested.numel() == 0:
            return
        if kwargs.get("past_key_value") is not None:
            # Native generate supplies an initially empty cache. Inspection is
            # valid only before this layer has appended any previous positions.
            cache = kwargs["past_key_value"]
            if cache.get_seq_length(module.layer_idx) != 0:
                raise ValueError("Native attention probes require fresh prefill")
        self.requested = requested

    def _query(self, module, args, output):
        if self.requested is not None:
            self.q = output

    def _key(self, module, args, output):
        if self.requested is not None:
            self.k = output

    def _finish(self, module, args, kwargs, output):
        try:
            if output is None or self.requested is None:
                return
            if self.q is None or self.k is None or self.q.shape[0] != 1:
                raise ValueError("Expected native batch-one Q/K projections")
            positions = self.positions()
            rows = torch.searchsorted(positions, self.requested).clamp_max(len(positions) - 1)
            if not torch.equal(positions[rows], self.requested):
                raise ValueError("Probe requested a removed query position")
            length = len(positions)
            q = self.q.view(1, length, module.num_heads, module.head_dim).transpose(1, 2)
            k = self.k.view(1, length, module.num_key_value_heads, module.head_dim).transpose(1, 2)
            cos, sin = module.rotary_emb(k, kwargs["position_ids"])
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k = repeat_kv(k, module.num_key_value_groups)
            logits = q.index_select(2, rows) @ k.transpose(2, 3) / module.head_dim**0.5
            mask = kwargs.get("attention_mask")
            if mask is None:
                allowed = positions[None, :] <= self.requested[:, None]
                logits = logits.masked_fill(~allowed[None, None], torch.finfo(logits.dtype).min)
            else:
                logits = logits + mask.index_select(-2, rows)
            probabilities = F.softmax(logits, dim=-1, dtype=torch.float32).to(q.dtype)
            if self.gravity_correction:
                # Evaluate the requested rectangle of the author's full ones-RoPE
                # correction. Normalization still includes ALL current keys.
                from transformers.models.llama.modeling_llama import rotate_half

                ones = torch.ones_like(q)
                rotated = ones * cos.unsqueeze(1) + rotate_half(ones) * sin.unsqueeze(1)
                gravity = rotated.index_select(2, rows) @ rotated.transpose(2, 3)
                gravity = gravity / module.head_dim**0.5
                probabilities = probabilities / gravity
            self.probabilities = (
                probabilities[0] if self.per_head else probabilities.mean(1, dtype=torch.float32)[0]
            )
        finally:
            self.q = self.k = self.requested = None

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clear()
