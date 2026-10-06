import numpy as np
import pytest

from efficient_vla_benchmark.action_codecs.core import Context, Packet
from efficient_vla_benchmark.action_codecs.implementations.uniform_binning.adapter import (
    UniformBinning,
)
from efficient_vla_benchmark.action_codecs.verification import independent_decode


def codec():
    return UniformBinning(
        calibration={
            "corpus_sha256": "fixture",
            "quantiles": {"s": {"low": [-1.0] * 6, "high": [1.0] * 6}},
        }
    )


def test_endpoints_match_native_digitize_and_midpoints():
    c = codec()
    actions = np.array([[-1.0] * 7, [1.0] * 7], dtype=np.float32)
    context = Context("s", np.zeros(8))
    packet = c.encode(actions, context)
    assert packet.ids[0, 0] == 0 and packet.ids[1, 0] == 255
    expected = np.array([-1 + 1 / 255, 1 - 1 / 255])
    np.testing.assert_allclose(c.decode(packet, context)[:, 0], expected, atol=1e-7)
    assert packet.nominal_bits == len(actions) * 56
    independent_decode(c, codec(), actions, context)
    with pytest.raises(ValueError):
        c.decode(Packet(np.zeros(4, np.int64), 256, 2), context)
