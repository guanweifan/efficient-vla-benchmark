"""Release checks are opt-in; no implicit network or local paths in tests."""

import os

import numpy as np
import pytest


def test_released_processor_parity_and_malformed_payload():
    assets = os.environ.get("EVB_FAST_PLUS_ASSETS")
    if not assets:
        pytest.skip("Explicit released processor directory required")
    from efficient_vla_benchmark.action_codecs.core import Context, Packet
    from efficient_vla_benchmark.action_codecs.implementations.fast_plus.adapter import FastPlus
    from efficient_vla_benchmark.action_codecs.normalization import normalize, unnormalize

    bounds = {"low": [-1.0] * 6, "high": [1.0] * 6}
    codec = FastPlus(
        assets=assets, calibration={"quantiles": {"test": bounds}, "corpus_sha256": "test"}
    )
    actions = np.random.default_rng(0).uniform(-0.8, 0.8, (40, 7)).astype(np.float32)
    actions[:, 6] = np.where(actions[:, 6] > 0, 1, -1)
    context = Context("test", np.zeros(8))
    expected_codes = codec.processor(normalize(actions, bounds, signed_gripper=True))[0]
    packet = codec.encode(actions, context)
    assert packet.ids.tolist() == expected_codes
    expected = unnormalize(
        codec.processor.decode([expected_codes], time_horizon=40, action_dim=7)[0],
        bounds,
        signed_gripper=True,
    )
    np.testing.assert_array_equal(codec.decode(packet, context), expected)
    with pytest.raises(ValueError, match="coefficient"):
        codec.decode(Packet(packet.ids[:1], 2048, 40), context)
    codec.close()
