"""Selected coupled heads with native RoPE, masks and fresh-prefill cache semantics."""

from torch.nn import functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from efficient_vla_benchmark.model_ops.indexed_linear import indexed_linear


def sparse_attention(
    module,
    hidden_states,
    channels,
    *,
    causal,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
    **kwargs,
):
    if module.training or output_attentions or kwargs:
        raise ValueError("EcoVLA supports eval-mode native Llama SDPA fresh prefill only")
    if hidden_states.shape[0] != 1 or module.num_heads != module.num_key_value_heads:
        raise ValueError("EcoVLA initial binding requires batch-one coupled MHA heads")
    batch, length, _ = hidden_states.shape
    heads = channels.numel() // module.head_dim
    q, k, v = [
        indexed_linear(hidden_states, p.weight, rows=channels)
        .view(batch, length, heads, module.head_dim)
        .transpose(1, 2)
        for p in (module.q_proj, module.k_proj, module.v_proj)
    ]
    cos, sin = module.rotary_emb(v, position_ids)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    if past_key_value is not None:
        if past_key_value.get_seq_length(module.layer_idx):
            raise ValueError("Cross-query autoregressive caches are unsupported")
        k, v = past_key_value.update(
            k, v, module.layer_idx, {"sin": sin, "cos": cos, "cache_position": cache_position}
        )
    mask = attention_mask
    if mask is not None:
        mask = mask[..., : k.shape[-2]]
        if not causal:
            mask = mask[:, :, -1:, :].expand(-1, -1, length, -1)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    result = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=causal and mask is None and length > 1
    )
    result = result.transpose(1, 2).contiguous().reshape(batch, length, channels.numel())
    return indexed_linear(result, module.o_proj.weight, columns=channels), None, past_key_value
