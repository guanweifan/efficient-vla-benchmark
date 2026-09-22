"""OFT's bidirectional decoder and auxiliary-module offloading."""

from ...target import DecoderTarget


def decoder_target(backend):
    def offload():
        backend.model.cpu()
        backend.action_head.cpu()
        backend.proprio_projector.cpu()

    return DecoderTarget(
        backend_id="openvla_oft",
        layers=backend.model.language_model.model.layers,
        device=backend.device,
        precision=backend.precision,
        checkpoint_sha256=backend._model_manifest["sha256"],
        causal=False,
        offload=offload,
    )
