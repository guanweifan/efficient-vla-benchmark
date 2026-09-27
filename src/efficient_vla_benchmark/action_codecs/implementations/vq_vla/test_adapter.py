import os

import numpy as np
import pytest


def test_released_vq_vla_codes_and_detached_decode():
    assets = os.environ.get("EVB_VQ_VLA_ASSETS")
    source = os.environ.get("EVB_VQ_VLA_SOURCES")
    directory = os.environ.get("EVB_CODEC_CORPUS")
    if not all((assets, source, directory)):
        pytest.skip("Explicit frozen weights, source and corpus required")
    import torch

    from efficient_vla_benchmark.action_codecs.corpus import Corpus
    from efficient_vla_benchmark.action_codecs.implementations.vq_vla.adapter import VqVla
    from efficient_vla_benchmark.action_codecs.normalization import fit_quantiles, normalize

    corpus = Corpus(directory)
    codec = VqVla(
        assets=assets,
        sources=source,
        corpus=corpus,
        calibration={"corpus_sha256": corpus.sha256, "quantiles": fit_quantiles(corpus)},
        device=os.environ.get("EVB_CODEC_DEVICE", "cpu"),
    )
    action, context = corpus.window(corpus.rows("evaluation")[0])
    packet = codec.encode(action, context)
    native_input = normalize(action[:5], codec.bounds[context.suite], signed_gripper=False)
    native = codec.model.get_code(torch.as_tensor(native_input[None], device=codec.device))
    np.testing.assert_array_equal(packet.ids[0], native[0].cpu().numpy())
    assert packet.token_count == 32
    decoded = codec.decode(packet, context)
    codec.encode(corpus.window(corpus.rows("evaluation")[1])[0], context)
    np.testing.assert_array_equal(codec.decode(packet, context), decoded)
    assert decoded.shape == (40, 7) and np.isfinite(decoded).all()
    codec.close()
    codec.close()
