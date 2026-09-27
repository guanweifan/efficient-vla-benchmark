import os

import numpy as np
import pytest


def test_released_actioncodec_native_blocks():
    assets, directory = os.environ.get("EVB_ACTIONCODEC_ASSETS"), os.environ.get("EVB_CODEC_CORPUS")
    if not assets or not directory:
        pytest.skip("Explicit released weights and corpus required")
    from efficient_vla_benchmark.action_codecs.corpus import Corpus
    from efficient_vla_benchmark.action_codecs.implementations.actioncodec.adapter import (
        ReleasedActionCodec,
    )

    corpus = Corpus(directory)
    codec = ReleasedActionCodec(
        assets=assets,
        calibration={},
        corpus=corpus,
        device=os.environ.get("EVB_CODEC_DEVICE", "cpu"),
    )
    actions, context = corpus.window(corpus.rows("evaluation")[0])
    packet = codec.encode(actions, context)
    expected = [
        codec.model.encode(x[None], embodiment_ids=codec.embodiment)[0]
        for x in np.split(codec.normalized(actions), 2)
    ]
    np.testing.assert_array_equal(packet.ids, expected)
    assert packet.token_count == 32
    decoded = codec.decode(packet, context)
    codec.encode(corpus.window(corpus.rows("evaluation")[1])[0], context)
    np.testing.assert_array_equal(codec.decode(packet, context), decoded)
    assert decoded.shape == (40, 7) and np.isfinite(decoded).all()
    with pytest.raises(ValueError, match="whole native"):
        codec.encode(actions[:19], context)
    codec.close()
    codec.close()
