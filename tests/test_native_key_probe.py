import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from efficient_vla_benchmark.model_ops.key_probe import NativeKeyProbe


def test_key_probe_matches_independent_native_rotation_without_changing_logits():
    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config).eval()
    embeddings = torch.randn(1, 7, 32)
    with torch.inference_mode():
        expected = model(inputs_embeds=embeddings, use_cache=False).logits
    attention = model.model.layers[0].self_attn
    captured = []
    handle = attention.k_proj.register_forward_hook(
        lambda module, args, output: captured.append(output)
    )
    probe = NativeKeyProbe(attention)
    try:
        with torch.inference_mode():
            actual = model(inputs_embeds=embeddings, use_cache=False).logits
            keys = captured[0].view(1, 7, 2, 8).transpose(1, 2)
            positions = torch.arange(7)[None]
            cos, sin = attention.rotary_emb(keys, positions)
            _, rotated = apply_rotary_pos_emb(keys, keys, cos, sin)
        torch.testing.assert_close(probe.keys, rotated.transpose(1, 2).reshape(7, -1))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert probe.projected is None
        probe.clear()
        assert probe.keys is None
    finally:
        handle.remove()
        probe.close()
        probe.close()
    with torch.inference_mode():
        torch.testing.assert_close(
            model(inputs_embeds=embeddings, use_cache=False).logits, expected, rtol=0, atol=0
        )
