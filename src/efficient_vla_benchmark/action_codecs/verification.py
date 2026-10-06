"""Discrete-only independent-decoder audit, outside measured latency."""

import io

import numpy as np

from .core import Context, Packet, actions_checked


def independent_decode(encoder, decoder, actions, context, *, atol=1e-6):
    """Decoder must be a fresh independently loaded instance, not a copy of encoder."""
    if encoder is decoder:
        raise ValueError("Independent decoder instance required")
    before = (encoder.metadata(), decoder.metadata())
    packet = encoder.encode(actions.copy(), context)
    expected = actions_checked(encoder.decode(packet, context), len(actions))
    stream = io.BytesIO()
    np.savez(
        stream,
        ids=packet.ids,
        vocabulary=packet.vocabulary,
        horizon=packet.horizon,
    )
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as archive:
        restored = Packet(archive["ids"], int(archive["vocabulary"]), int(archive["horizon"]))
    fresh_context = Context(context.suite, context.initial_state)
    actual = actions_checked(decoder.decode(restored, fresh_context), len(actions))
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0)
    if before != (encoder.metadata(), decoder.metadata()):
        raise ValueError("Codec metadata changed during independent decode")
    return {
        "tokens": packet.token_count,
        "nominal_bits": packet.nominal_bits,
        "max_independent_decode_difference": float(np.max(np.abs(actual - expected))),
    }
