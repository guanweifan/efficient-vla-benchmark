"""Model-independent released layer-zero Q/K and quota rules; imported only by selected runtimes."""


def embedding_scores(layer, text, visual):
    """Released ADP scorer: RMSNorm, pretrained Q/K, raw head-averaged logits.

    No softmax, no RoPE and no hidden full-model probe. All selection stays on GPU.
    """
    attn = layer.self_attn
    q = attn.q_proj(layer.input_layernorm(text))
    k = attn.k_proj(layer.input_layernorm(visual))
    heads, width = attn.num_heads, attn.head_dim
    if attn.num_key_value_heads != heads:
        raise ValueError("The released ADP scorer requires matching Q/K head counts")
    q = q.view(1, -1, heads, width).transpose(1, 2)
    k = k.view(1, -1, heads, width).transpose(1, 2)
    return ((q @ k.transpose(-1, -2)) * width**-0.5).mean((0, 1, 2))


def view_quotas(total, weights, patches):
    # Match the reference's rounding, min-one and per-camera clamp. Very small
    # budgets are raised to the camera count to prevent its nonterminating loop.
    total = max(len(weights), total)
    normalized = [w / sum(weights) for w in weights]
    quotas = [max(1, round(total * w)) for w in normalized]
    order = sorted(range(len(weights)), key=lambda i: normalized[i], reverse=True)
    difference = total - sum(quotas)
    while difference:
        for i in order:
            if difference > 0:
                quotas[i] += 1
                difference -= 1
            elif difference < 0 and quotas[i] > 1:
                quotas[i] -= 1
                difference += 1
            if not difference:
                break
    return [min(q, patches) for q in quotas]
