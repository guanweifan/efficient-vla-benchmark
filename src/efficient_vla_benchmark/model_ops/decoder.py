"""Optional Llama attention inspection and bounded indexed KV storage."""

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


class IndexedKVStore:
    """Overwrite current rows, retain skipped rows; never append across frames."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.keys: dict[int, torch.Tensor] = {}
        self.values: dict[int, torch.Tensor] = {}

    def update(self, layer: int, keys, values, positions, full_length: int):
        if layer not in self.keys:
            if keys.shape[2] != full_length:
                raise ValueError("A KV layer must be initialized with a complete frame")
            expected = torch.arange(full_length, device=positions.device)
            if not torch.equal(positions, expected):
                raise ValueError("Initial KV positions must cover the complete sequence")
            self.keys[layer] = keys.clone()
            self.values[layer] = values.clone()
        else:
            if self.keys[layer].shape[2] != full_length:
                raise ValueError("Sequence layout changed without a cache reset")
            self.keys[layer].index_copy_(2, positions, keys)
            self.values[layer].index_copy_(2, positions, values)
        return self.keys[layer], self.values[layer]


class InspectableAttention(nn.Module):
    """Native SDPA probes, or shared eager probabilities for full-map consumers."""

    def __init__(self, original, owner, layer):
        super().__init__()
        self.q_proj = original.q_proj
        self.k_proj = original.k_proj
        self.v_proj = original.v_proj
        self.o_proj = original.o_proj
        self.rotary_emb = original.rotary_emb
        self.num_heads = original.num_heads
        self.num_key_value_heads = original.num_key_value_heads
        self.num_key_value_groups = original.num_key_value_groups
        self.head_dim = original.head_dim
        self.hidden_size = original.hidden_size
        self.owner = owner  # Plain object, not another registered nn.Module.
        self.layer = layer
        self.last_attention = {}
        self.train(original.training)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        **kwargs,
    ):
        if self.training or output_attentions or use_cache or past_key_value is not None:
            raise ValueError("The method decoder supports inference and its own probes/cache only")
        batch, length, _ = hidden_states.shape
        queries = (
            self.q_proj(hidden_states)
            .view(batch, length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        keys = (
            self.k_proj(hidden_states)
            .view(batch, length, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        values = (
            self.v_proj(hidden_states)
            .view(batch, length, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        cos, sin = self.rotary_emb(values, position_ids)
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)
        state, strategy = self.owner.state, self.owner.strategy
        keys, values = strategy.update_kv(self.layer, keys, values, state)
        keys = repeat_kv(keys, self.num_key_value_groups)
        values = repeat_kv(values, self.num_key_value_groups)
        if attention_mask is not None:
            queries, keys, values = queries.contiguous(), keys.contiguous(), values.contiguous()
        self.last_attention = {}
        probes = strategy.probes(self.layer, state)
        probabilities = None
        if strategy.attention_backend == "eager":
            # Same execution pattern as the reference LlamaAttention: compute
            # probabilities once, then reuse them for both P @ V and statistics.
            logits = torch.matmul(queries, keys.transpose(2, 3)) / self.head_dim**0.5
            if attention_mask is not None:
                logits = logits + attention_mask
            probabilities = F.softmax(logits, dim=-1, dtype=torch.float32).to(queries.dtype)
            output = torch.matmul(probabilities, values)
        for name, requested in probes.items():
            if requested.numel() == 0:
                self.last_attention[name] = queries.new_empty(
                    (0, keys.shape[2]), dtype=torch.float32
                )
                continue
            # Identity is a safe full-row request, not a disabled validity check.
            # It avoids a redundant GPU search/gather/equality synchronization.
            if requested is state.positions:
                row_indices = None
            else:
                row_indices = torch.searchsorted(state.positions, requested).clamp_max(
                    state.positions.numel() - 1
                )
                if not torch.equal(state.positions[row_indices], requested):
                    raise ValueError("An attention probe requested a removed query token")
            if probabilities is not None:
                selected = (
                    probabilities
                    if row_indices is None
                    else probabilities.index_select(2, row_indices)
                )
            else:
                probe_queries = (
                    queries if row_indices is None else queries.index_select(2, row_indices)
                )
                logits = torch.matmul(probe_queries, keys.transpose(2, 3)) / self.head_dim**0.5
                if attention_mask is not None:
                    logits = logits + (
                        attention_mask
                        if row_indices is None
                        else attention_mask.index_select(-2, row_indices)
                    )
                selected = F.softmax(logits, dim=-1, dtype=torch.float32).to(queries.dtype)
            self.last_attention[name] = selected.mean(dim=1, dtype=torch.float32)[0]
        if probabilities is None:
            output = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=None if getattr(self.owner, "causal_sdpa", False) else attention_mask,
                dropout_p=0.0,
                is_causal=getattr(self.owner, "causal_sdpa", False),
            )
        output = output.transpose(1, 2).contiguous().view(batch, length, self.hidden_size)
        return self.o_proj(output), None, None
