import os

import numpy as np
import pytest

from efficient_vla_benchmark.action_codecs.core import Context


def test_command_embedding_round_trip_without_future_states():
    pytest.importorskip("scipy")
    from efficient_vla_benchmark.action_codecs.implementations.x_tokenizer.representation import (
        command_path,
        observed_reference,
        raw_commands,
    )

    rng = np.random.default_rng(91)
    actions = rng.uniform(-0.9, 0.9, (40, 7)).astype(np.float32)
    actions[:, 6] = np.where(actions[:, 6] > 0, 1, -1)
    context = Context("libero_goal", np.array([0.2, -0.1, 1.0, 3.1, 0.1, -0.1, 0.04, -0.04]))
    path, reference = command_path(actions, context)
    assert reference[9] == pytest.approx(1)
    np.testing.assert_allclose(raw_commands(path, context), actions, atol=3e-6)
    np.testing.assert_array_equal(reference, observed_reference(context))
    path[:, 3:9] = 0
    with pytest.raises(ValueError, match="Degenerate"):
        raw_commands(path, context)


def test_released_x_tokenizer_and_frozen_statistics():
    assets, sources = (
        os.environ.get("EVB_X_TOKENIZER_ASSETS"),
        os.environ.get("EVB_X_TOKENIZER_SOURCES"),
    )
    directory = os.environ.get("EVB_CODEC_CORPUS")
    if not all((assets, sources, directory)):
        pytest.skip("Explicit released weights, sources and corpus required")
    from efficient_vla_benchmark.action_codecs.corpus import Corpus
    from efficient_vla_benchmark.action_codecs.implementations.x_tokenizer.adapter import (
        XTokenizerCodec,
    )
    from efficient_vla_benchmark.action_codecs.implementations.x_tokenizer.representation import (
        command_path,
    )

    corpus = Corpus(directory)
    codec = XTokenizerCodec(
        assets=assets,
        sources=sources,
        corpus=corpus,
        calibration={"corpus_sha256": corpus.sha256},
        device=os.environ.get("EVB_CODEC_DEVICE", "cpu"),
    )
    meta = codec.metadata()
    action, context = corpus.window(corpus.rows("evaluation")[0])
    packet = codec.encode(action, context)
    path, reference = command_path(action, context)
    expected = codec.tokenizer.encode_from_absolute(
        path[None],
        reference[None, None],
        robot_type="Franka",
        dof_mask=codec.mask(40),
        token_order="time_major",
    )
    np.testing.assert_array_equal(packet.ids, expected[0].cpu().numpy())
    assert packet.token_count == 40
    decoded = codec.decode(packet, context)
    other, other_context = corpus.window(corpus.rows("evaluation")[1])
    codec.encode(other, other_context)
    np.testing.assert_array_equal(codec.decode(packet, context), decoded)
    assert meta == codec.metadata() and np.isfinite(decoded).all()
    codec.close()
    codec.close()
