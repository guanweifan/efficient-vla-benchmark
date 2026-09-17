import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from efficient_vla_benchmark.methods.token_reduction.adp.config import ADPConfig
from efficient_vla_benchmark.methods.token_reduction.adp.motion import MotionGate


@pytest.mark.parametrize(
    "bad",
    [
        {"keep_ratio": 0},
        {"keep_ratio": True},
        {"keep_ratio": float("nan")},
        {"enabled": 1},
        {"window": 2},
        {"view_weights": [0, 0]},
        {"decision": "future"},
        {"other": 1},
    ],
)
def test_invalid_config(bad):
    with pytest.raises(ValueError):
        ADPConfig.parse(bad)


def reference_node(filename, name):
    root = Path(__file__).resolve().parents[6]
    source = root / "third_party/methods/adp" / filename
    if not source.exists():
        pytest.skip("Initialize the pinned ADP reference submodule")
    tree = ast.parse(source.read_text())
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name
    )
    node.decorator_list = []
    return compile(
        ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), "exec"
    )


@pytest.mark.parametrize("decision", ["avg", "adjacent"])
@pytest.mark.parametrize("delta", ["net", "sum"])
def test_observed_motion_matches_pinned_controller(decision, delta):
    cfg = ADPConfig.parse({"decision": decision, "delta": delta})
    namespace = {
        "np": np,
        "quat2axisangle": lambda q: np.zeros(3),  # Rotation is unused by both supported deltas.
        "cfg": SimpleNamespace(
            initial_state=0,
            decision_method=decision,
            delta_method=delta,
            adjacent_extrema_window=3,
            adjacent_last_state=False,
            limit_consecutive_pruned_enabled=True,
            limit_max_consecutive_pruned=3,
        ),
    }
    exec(
        reference_node(
            "experiments/robot/libero/run_libero_eval_prune_v2.py", "_OnlineStrategyController"
        ),
        namespace,
    )
    reference, gate = namespace["_OnlineStrategyController"](), MotionGate(cfg)
    rng = np.random.default_rng(12)
    pos = np.zeros(3)
    gate.observe(pos)
    for _ in range(40):
        reference.observe_step(pos, np.zeros(4))
        for _ in range(8):
            pos = pos + rng.normal(size=3) * rng.uniform(0, 0.1)
            gate.observe(pos)
            reference.observe_step(pos, np.zeros(4))
        gate.close_window()
        reference.close_window_and_decide_next()
        assert gate.state == bool(reference.curr_state)
        assert len(gate.history) <= cfg.window


def test_qk_scores_match_pinned_source():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from ..implementations.openvla_oft.method import embedding_scores, view_quotas

    torch.manual_seed(8)
    model = LlamaForCausalLM(
        LlamaConfig(
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_hidden_layers=1,
        )
    )
    text, vision = torch.randn(1, 7, 32), torch.randn(1, 16, 32)
    namespace = {"torch": torch, "PreTrainedModel": object}
    exec(
        reference_node(
            "prismatic/extern/hf/modeling_prismatic.py", "_compute_qk_importance_generic"
        ),
        namespace,
    )
    with torch.inference_mode():
        expected = namespace["_compute_qk_importance_generic"](model, text, vision, 0)
        actual = embedding_scores(model.model.layers[0], text, vision)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert view_quotas(384, (0.4, 0.6), 256) == [154, 230]
    assert view_quotas(1, (0.4, 0.6), 256) == [1, 1]


@pytest.mark.gpu
def test_real_oft_disabled_parity_active_pruning_and_reset(model_path, trace_path, request):
    from efficient_vla_backend_openvla_oft.backend import OpenVLAOFTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..implementations.openvla_oft.method import ADP

    backend = OpenVLAOFTBackend(model_path)
    request.addfinalizer(backend.close)
    context, samples, _ = load_trace(trace_path)
    backend.reset(context)
    expected = backend.predict_chunk(samples[0])
    for params in ({"enabled": False}, {"keep_ratio": 1.0}):
        backend.attach_method(ADP(backend, ADPConfig.parse(params)))
        backend.reset(context)
        assert not backend.method.requires_control_history
        np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)
        backend.detach_method()
    backend.attach_method(ADP(backend, ADPConfig.parse({"dynamic": False})))
    backend.reset(context)
    active = backend.predict_chunk(samples[0])
    assert active.shape == (8, 7) and np.isfinite(active).all()
    assert backend.query_diagnostics["visual_tokens_final"] == 384
    assert len(set(backend.query_diagnostics["layer_sequence_lengths"])) == 1
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(samples[0]), active)
    backend.detach_method()
    np.testing.assert_array_equal(backend.predict_chunk(samples[0]), expected)


@pytest.mark.parametrize("parameters", [{"enabled": False}, {"keep_ratio": 1}])
def test_noop_does_not_require_control_observations(parameters):
    from types import SimpleNamespace

    from ..implementations.openvla_oft.method import ADP

    backend = SimpleNamespace()
    method = ADP(backend, ADPConfig.parse(parameters))
    assert method.adapter is None and not method.requires_control_history
    method.before_query({})
    assert not backend.query_diagnostics["pruned"]
