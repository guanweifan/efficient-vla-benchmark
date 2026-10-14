import itertools

import numpy as np
import pytest
import torch

from efficient_vla_benchmark.methods.quantization.dyq_vla.config import DyQConfig
from efficient_vla_benchmark.methods.quantization.dyq_vla.dispatcher import DyQDispatcher
from efficient_vla_benchmark.methods.quantization.dyq_vla.thresholds import fit_thresholds


def make():
    return DyQDispatcher(DyQConfig(), dict(theta_2_4=0.1, theta_4_8=0.3, theta_fp=0.5))


def test_algorithm_one_matches_exhaustive_target_sequences():
    for sequence in itertools.product((2, 4, 8, 16), repeat=5):
        d = make()
        active, counter, maximum = 16, 0, 16
        for target in sequence:
            if target >= active:
                active, counter, maximum = target, 0, target
            else:
                previous = maximum
                maximum = max(target, maximum * int(counter > 0))
                counter = counter * int(maximum == previous) + 1
                active = maximum if counter == 3 else active
                counter %= 3
            assert d.dispatch_target(target) == active
            assert (d.counter, d.pending_max) == (counter, maximum)


def test_pending_upgrade_restarts_downgrade_delay():
    d = make()
    assert [d.dispatch_target(b) for b in (2, 2, 4, 4, 4)] == [16, 16, 16, 16, 4]
    assert d.dispatch_target(8) == 8
    d.reset()
    assert d.active_bits == 16 and not d.motion


def test_only_fused_sensitivity_is_clamped_and_no_future_commands():
    d = make()
    assert d.target() == 16
    for v in (1.0, 1.0, 1.0, 1.0, 10.0):
        d.observe(np.array([v, 0, 0, 0, 0, 0, -1]))
    assert d.motion[-1] < 0
    assert d.last_sensitivity == 0
    other = make()
    assert not other.motion
    with pytest.raises(ValueError):
        d.observe(np.full(7, np.nan))


def test_threshold_fit_stops_at_first_crossing():
    cfg = DyQConfig.parse(dict(sensitivity_bins=3, accuracy_budget=0.02))
    records = []
    for s, e in ((0.1, 0.01), (0.2, 1.0), (0.4, 0.001)):
        records += [dict(sensitivity=s, errors={"2": e, "4": 0.001, "8": 0.0001})] * 4
    thresholds, bins = fit_thresholds(records, cfg)
    assert thresholds["theta_2_4"] == 0.1
    assert thresholds["theta_4_8"] == 0.4 and len(bins) == 3
    records = [dict(sensitivity=0.1, errors={"2": 10.0, "4": 10.0, "8": 0.0001})] * 4
    thresholds, _ = fit_thresholds(records, cfg)
    assert thresholds["theta_2_4"] == thresholds["theta_4_8"] == -1


def test_configuration_bounds():
    for values in (
        {"motion_window": True},
        {"eta": 0},
        {"accuracy_budget": float("nan")},
        {"frames_per_trajectory": 1000},
        {"unexpected": 1},
    ):
        with pytest.raises(ValueError):
            DyQConfig.parse(values)


@pytest.mark.gpu
@torch.inference_mode()
def test_dynamic_packed_calls_match_native_dtype_dense_oracle_and_reset():
    from efficient_vla_benchmark.methods.quantization.dyq_vla.runtime import DynamicW4Linear

    torch.manual_seed(17)
    q = torch.randint(-8, 8, (64, 128), dtype=torch.int8)
    s = torch.rand(64) * 0.01 + 0.001
    a = torch.rand(128) * 2 + 1
    e = dict(codes=q, weight_scale=s, activation_amax=a, bias=None)
    op = DynamicW4Linear(e).cuda()
    x = torch.randn(33, 128, device="cuda", dtype=torch.bfloat16)
    w = q.to(x) * s.to(x)[:, None]
    for bits in (16, 2, 8, 4, 16):
        op.set_active_bits(bits)
        refx = x
        if bits < 16:
            maximum = (1 << (bits - 1)) - 1
            step = (a.to(x) / maximum).clamp_min(1e-8)
            refx = (x / step).round().clamp(-maximum - 1, maximum) * step
        torch.testing.assert_close(op(x), torch.nn.functional.linear(refx, w), atol=0.01, rtol=0.01)
    assert op.kernel_invocations == 5
    assert op.packed_weight.numel() == 64 * 128 // 2
