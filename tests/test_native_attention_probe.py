"""Independent eager reference for read-only native SDPA inspection."""

from copy import deepcopy
from importlib import import_module

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

NativeAttentionProbe = import_module(
    "efficient_vla_benchmark.model_ops.attention_probe"
).NativeAttentionProbe


@pytest.mark.parametrize("padding", [False, True])
def test_native_probe_matches_eager_and_preserves_output(padding):
    torch.manual_seed(12)
    cfg = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    cfg._attn_implementation = "sdpa"
    model = transformers.LlamaForCausalLM(cfg).eval()
    eager_cfg = deepcopy(cfg)
    eager_cfg._attn_implementation = "eager"
    eager = transformers.LlamaForCausalLM(eager_cfg).eval()
    eager.load_state_dict(model.state_dict())
    hidden = torch.randn(1, 9, 32)
    positions = torch.tensor([0, 1, 3, 4, 7, 8, 9, 11, 12])
    requested = positions[torch.tensor([2, 8])]
    mask = torch.ones(1, 9, dtype=torch.long)
    if padding:
        mask[:, 1] = 0
    inputs = dict(
        inputs_embeds=hidden, position_ids=positions[None], attention_mask=mask, use_cache=False
    )
    with torch.inference_mode():
        expected = model(**inputs).logits
        reference = eager(**inputs, output_attentions=True).attentions[0]
    module = model.model.layers[0].self_attn
    original = module.forward
    probe = NativeAttentionProbe(module, lambda: requested, lambda: positions)
    try:
        with torch.inference_mode():
            actual = model(**inputs).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(probe.probabilities, reference[0, :, [2, 8]].mean(0))
        assert probe.probabilities[0, 3:].count_nonzero() == 0
        assert probe.q is None and probe.k is None
        assert module.forward == original
    finally:
        probe.close()
        probe.close()
    with torch.inference_mode():
        torch.testing.assert_close(model(**inputs).logits, expected, rtol=0, atol=0)


def test_idle_probe_and_exception_clear_transient_state():
    cfg = transformers.LlamaConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2
    )
    cfg._attn_implementation = "sdpa"
    model = transformers.LlamaForCausalLM(cfg).eval()
    probe = NativeAttentionProbe(
        model.model.layers[0].self_attn, lambda: None, lambda: torch.arange(4)
    )
    try:
        with torch.inference_mode():
            model(inputs_embeds=torch.randn(1, 4, 16), use_cache=False)
        assert probe.probabilities is None and probe.q is None and probe.k is None
        probe.request = lambda: torch.tensor([7])
        with torch.inference_mode(), pytest.raises(ValueError, match="removed"):
            model(inputs_embeds=torch.randn(1, 4, 16), use_cache=False)
        assert probe.q is None and probe.k is None
    finally:
        probe.close()
