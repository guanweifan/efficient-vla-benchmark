import numpy as np
import pytest

from efficient_vla_benchmark.action_codecs.core import Context, Packet
from efficient_vla_benchmark.action_codecs.corpus import Corpus, derive_prefix, write_corpus
from efficient_vla_benchmark.action_codecs.verification import independent_decode


def test_fixed_prefix_preserves_split_and_episode_identity(tmp_path):
    rows = [
        dict(
            suite="s",
            episode_index=i,
            offset=i,
            split=split,
            actions=np.full((40, 7), i, np.float32),
            initial_state=np.zeros(8),
        )
        for i, split in enumerate(("calibration", "evaluation"))
    ]
    write_corpus(tmp_path / "source", rows, horizon=40, provenance={})
    derive_prefix(tmp_path / "source", tmp_path / "derived", 16)
    derived = Corpus(tmp_path / "derived")
    assert derived.manifest["horizon"] == 16
    for expected, row in zip(rows, derived.manifest["windows"]):
        assert all(expected[k] == row[k] for k in ("suite", "episode_index", "offset", "split"))
        np.testing.assert_array_equal(derived.window(row)[0], expected["actions"][:16])
    with pytest.raises(ValueError):
        derive_prefix(tmp_path / "source", tmp_path / "bad", 80)


def test_independent_decode_detects_hidden_encoder_state():
    class Hidden:
        def metadata(self):
            return {}

        def encode(self, actions, context):
            self.saved = actions
            return Packet(np.array([0]), 2, len(actions))

        def decode(self, packet, context):
            return self.saved

    with pytest.raises(AttributeError):
        independent_decode(Hidden(), Hidden(), np.zeros((16, 7)), Context("s", np.zeros(8)))
