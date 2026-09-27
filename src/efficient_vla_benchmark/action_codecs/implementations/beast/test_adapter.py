import os

import numpy as np
import pytest


def test_native_beast_parity_and_frozen_calibration():
    assets, directory = os.environ.get("EVB_BEAST_ASSETS"), os.environ.get("EVB_CODEC_CORPUS")
    if not assets or not directory:
        pytest.skip("Explicit processor and frozen corpus required")
    import torch

    from efficient_vla_benchmark.action_codecs.corpus import Corpus
    from efficient_vla_benchmark.action_codecs.implementations.beast.adapter import Beast

    corpus = Corpus(directory)
    codec = Beast(
        assets=assets,
        corpus=corpus,
        device=os.environ.get("EVB_CODEC_DEVICE", "cpu"),
        calibration={"corpus_sha256": corpus.sha256},
    )
    original = codec.metadata()
    actions, context = corpus.window(corpus.rows("evaluation")[0])
    expected = (
        codec.tokenizer.encode_discrete(
            torch.as_tensor(actions[None], device=codec.device), update_bounds=False
        )[0]
        .cpu()
        .numpy()
    )
    packet = codec.encode(actions, context)
    np.testing.assert_array_equal(packet.ids, expected)
    reconstructed = codec.decode(packet, context)
    b, other_context = corpus.window(corpus.rows("evaluation")[1])
    codec.encode(b, other_context)
    np.testing.assert_allclose(codec.decode(packet, context), reconstructed, rtol=0, atol=0)
    direct = codec.tokenizer.decode_discrete(torch.tensor(expected[None], device=codec.device))
    np.testing.assert_array_equal(reconstructed, direct[0].cpu().numpy())
    assert codec.metadata() == original and packet.token_count == 70
    codec.close()
    codec.close()
