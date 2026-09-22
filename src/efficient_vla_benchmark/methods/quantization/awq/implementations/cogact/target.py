"""CogACT decoder access without changing vision, cognition or DiT sampling."""

from ...target import DecoderTarget


def decoder_target(backend):
    layers = backend.model_access.decoder.layers
    if len(layers) != 32:
        raise ValueError("CogACT AWQ requires the validated 32-layer Llama decoder")
    return DecoderTarget(
        backend_id="cogact",
        layers=layers,
        device=backend.device,
        precision="bfloat16",
        checkpoint_sha256=backend._model_manifest["sha256"],
        causal=True,
        offload=backend.model.cpu,
    )
