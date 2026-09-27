import json

import numpy as np
import pytest

from efficient_vla_benchmark.action_codecs.core import Context, Packet, actions_checked
from efficient_vla_benchmark.action_codecs.corpus import Corpus, write_corpus
from efficient_vla_benchmark.action_codecs.evaluation import evaluate
from efficient_vla_benchmark.action_codecs.normalization import (
    fit_quantiles,
    normalize,
    unnormalize,
)


def fixture_corpus(tmp_path):
    rng = np.random.default_rng(42)
    rows = []
    for i in range(4):
        actions = rng.uniform(-1, 1, (40, 7)).astype(np.float32)
        actions[:, 6] = np.where(actions[:, 6] > 0, 1.0, -1.0)
        rows.append(
            dict(
                suite="libero_goal",
                episode_index=i,
                offset=3,
                split="calibration" if i < 2 else "evaluation",
                actions=actions,
                initial_state=np.zeros(8, dtype=np.float32),
            )
        )
    target = tmp_path / "corpus"
    write_corpus(target, rows, horizon=40, provenance={"test": True})
    return Corpus(target), rows


def test_corpus_round_trip_and_tampering(tmp_path):
    corpus, rows = fixture_corpus(tmp_path)
    assert len(corpus.rows("evaluation")) == 2
    np.testing.assert_array_equal(
        corpus.window(corpus.manifest["windows"][0])[0], rows[0]["actions"]
    )
    with pytest.raises(FileExistsError):
        write_corpus(tmp_path / "corpus", rows, horizon=40, provenance={})
    path = tmp_path / "corpus/manifest.json"
    meta = json.loads(path.read_text())
    meta["windows"][2]["episode_index"] = 0
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="Duplicate"):
        Corpus(tmp_path / "corpus")


def test_duplicate_content_and_episode_rejected(tmp_path):
    _, rows = fixture_corpus(tmp_path)
    rows[2]["actions"] = rows[0]["actions"]
    with pytest.raises(ValueError, match="Duplicate action"):
        write_corpus(tmp_path / "bad", rows, horizon=40, provenance={})
    rows[2]["episode_index"] = 0
    with pytest.raises(ValueError, match="cannot overlap"):
        write_corpus(tmp_path / "bad", rows, horizon=40, provenance={})


def test_packet_cannot_carry_continuous_side_information():
    with pytest.raises(ValueError):
        Packet(np.ones(3, dtype=np.float32), 2048, 40)
    with pytest.raises(ValueError):
        Packet(np.array([2048]), 2048, 40)
    packet = Packet(np.arange(8), 256, 40)
    assert packet.nominal_bits == 64
    with pytest.raises(ValueError):
        packet.ids[0] = 1
    with pytest.raises(ValueError):
        Context("suite", np.zeros((40, 8)))
    with pytest.raises(ValueError):
        actions_checked(np.zeros((10, 7)), 40)


def test_quantile_fit_uses_calibration_only(tmp_path):
    corpus, _ = fixture_corpus(tmp_path)
    before = fit_quantiles(corpus)
    for r in corpus.rows("evaluation"):
        corpus.arrays[r["key"] + "_actions"][:] = 100
    assert fit_quantiles(corpus) == before
    bounds = {"low": [-1.0] * 6, "high": [1.0] * 6}
    action = np.zeros((2, 7), dtype=np.float32)
    action[:, 6] = [-1.0, 1.0]
    for signed in (False, True):
        np.testing.assert_allclose(
            unnormalize(
                normalize(action, bounds, signed_gripper=signed), bounds, signed_gripper=signed
            ),
            action,
            atol=1e-6,
        )


class DummyCodec:
    def encode(self, action, context):
        return Packet(np.rint((action + 1) * 127).astype(np.int64), 256, len(action))

    def decode(self, packet, context):
        return packet.ids / 127 - 1

    def metadata(self):
        return {"dummy": True}

    def synchronize(self):
        pass


def test_harness_counts_real_tokens_and_raw_error(tmp_path):
    corpus, _ = fixture_corpus(tmp_path)
    report = evaluate(DummyCodec(), corpus, tmp_path / "result", warmup=2)
    assert report["complete"] and report["errors"] == 0
    assert report["tokens_per_frame"] == 7 and report["nominal_bits_per_frame"] == 56
    assert 0 < report["mse"] < 0.0001
    assert report["encode_ms"] > 0 and report["decode_ms"] > 0
    with pytest.raises(FileExistsError):
        evaluate(DummyCodec(), corpus, tmp_path / "result")


def test_invalid_reconstruction_is_not_dropped_from_mean(tmp_path):
    class Bad(DummyCodec):
        def decode(self, packet, context):
            raise ValueError("Invalid codes, not a zero-action fallback")

    corpus, _ = fixture_corpus(tmp_path)
    report = evaluate(Bad(), corpus, tmp_path / "bad-result", warmup=0)
    assert not report["complete"] and report["errors"] == 2 and "mse" not in report
