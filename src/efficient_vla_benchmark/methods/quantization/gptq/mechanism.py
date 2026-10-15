"""Sequential Llama projection groups and native-input capture, without padding."""

import torch
from torch import nn

GROUPS = (
    ("self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"),
    ("self_attn.o_proj",),
    ("mlp.up_proj", "mlp.gate_proj"),
    ("mlp.down_proj",),
)
PROJECTIONS = tuple(name for group in GROUPS for name in group)
CONTRACT = "gptq_dynamic_g128_true_sequential_native_bf16_calibration_v1"


def validate_layer(layer):
    for name in PROJECTIONS:
        module = layer.get_submodule(name)
        if (
            not isinstance(module, nn.Linear)
            or module.weight.dtype != torch.bfloat16
            or module.in_features % 128
            or module.out_features % 128
        ):
            raise ValueError("GPTQ requires fresh aligned native BF16 Llama projections")
    for group in GROUPS:
        if len({layer.get_submodule(n).in_features for n in group}) != 1:
            raise ValueError("Invalid shared GPTQ input group")


def move_arguments(arguments, device):
    return {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in arguments.items()
    }


def native_arguments(kwargs):
    allowed = {
        "attention_mask",
        "position_ids",
        "cache_position",
        "past_key_value",
        "output_attentions",
        "use_cache",
        "position_embeddings",
    }
    if set(kwargs) - allowed:
        raise ValueError("Unrecognized native decoder argument")
    cache = kwargs.get("past_key_value")
    if cache is not None and (not hasattr(cache, "get_seq_length") or cache.get_seq_length() != 0):
        raise ValueError("GPTQ capture requires fresh prefill, not populated KV cache")
    result = dict(kwargs, past_key_value=None, use_cache=False, output_attentions=False)
    if result.get("position_embeddings") is not None:
        raise ValueError("Externally supplied rotary embeddings are not validated")
    return move_arguments(result, "cpu")


class CapturedInput(Exception):
    """Intentional early exit, distinct from unrelated model failures."""


@torch.inference_mode()
def capture_inputs(backend, observations, layers):
    samples, identities = [], []

    def capture(module, args, kwargs):
        if len(args) != 1 or args[0].ndim != 3 or args[0].shape[0] != 1:
            raise ValueError("GPTQ capture requires one native multimodal observation")
        samples.append((args[0].detach().cpu(), native_arguments(kwargs)))
        raise CapturedInput

    handle = layers[0].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        for context, observation, identity in observations:
            backend.reset(context)
            before = len(samples)
            try:
                backend.predict_chunk(observation)
            except CapturedInput:
                pass
            if len(samples) != before + 1:
                raise ValueError("Native GPTQ embedding capture did not execute exactly once")
            identities.append(identity)
    finally:
        handle.remove()
    return samples, identities
