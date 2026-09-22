"""Causal calibration isolation, artifact boundaries and native sampler invariants."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from ..artifact import load_artifact
from ..config import AWQConfig
from ..search import batch_inputs, layer_arguments


@torch.inference_mode()
def test_causal_calibration_matches_independent_native_samples():
    pytest.importorskip("efficient_vla_backend_cogact")
    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=1,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(7)
    layer = LlamaDecoderLayer(config, 0).eval()
    samples = [torch.randn(1, length, 64) for length in (12, 9)]
    inputs, valid = batch_inputs(samples, "cpu", torch.float32)
    actual = layer(inputs, **layer_arguments(valid, inputs.dtype, causal=True))[0]
    for index, sample in enumerate(samples):
        positions = torch.arange(sample.shape[1]).unsqueeze(0)
        expected = layer(sample, position_ids=positions, use_cache=False)[0]
        torch.testing.assert_close(actual[index, : sample.shape[1]], expected[0], atol=1e-6, rtol=0)
    inputs[0, -1] += 100
    changed = layer(inputs, **layer_arguments(valid, inputs.dtype, causal=True))[0]
    torch.testing.assert_close(changed[0, :-1], actual[0, :-1], atol=0, rtol=0)
    torch.testing.assert_close(changed[1], actual[1], atol=0, rtol=0)


def test_other_backend_artifact_rejected_before_mutation(monkeypatch, tmp_path):
    from .. import artifact

    monkeypatch.setattr(artifact, "read_manifest", lambda *args: {"backend": "openvla_oft"})
    with pytest.raises(ValueError, match="different backend"):
        load_artifact(SimpleNamespace(backend_id="cogact"), tmp_path, AWQConfig())


@pytest.mark.gpu
@pytest.mark.parametrize("kernel_precision", ["auto", "float16"])
def test_cogact_packed_scope_reset_and_reload(
    model_path, trace_path, artifact_path, kernel_precision
):
    from dataclasses import replace

    from efficient_vla_benchmark.core.config import ExperimentConfig
    from efficient_vla_benchmark.core.registry import build_backend
    from efficient_vla_benchmark.core.resources import project_source
    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..vendor import reference_module

    config = ExperimentConfig.load(project_source("configs/experiments/cogact") / "awq.yaml")
    config = replace(config, parameters={**config.parameters, "kernel_precision": kernel_precision})
    context, samples, _ = load_trace(trace_path)
    selected = [samples[0], samples[len(samples) // 2], samples[-1]]
    packed_type = reference_module("quantize.qmodule").WQLinear
    outputs = []
    for _ in range(2):
        backend = build_backend(config, model_path, "cuda:0", artifact_path)
        try:
            assert sum(isinstance(m, packed_type) for m in backend.model.modules()) == 224
            assert (
                sum(isinstance(m, packed_type) for m in backend.model_access.decoder.modules())
                == 224
            )
            assert backend.metadata()["action_head_precision"] == "float32"
            assert backend.metadata()["attention_mask"] == "causal"
            assert backend.metadata()["sampler"] == "ddim"
            backend.reset(context)
            actions = [backend.predict_chunk(obs) for obs in selected]
            assert all(a.shape == (16, 7) and np.isfinite(a).all() for a in actions)
            backend.reset(context)
            np.testing.assert_array_equal(backend.predict_chunk(selected[0]), actions[0])
            with pytest.raises(ValueError, match="fresh backend"):
                backend.detach_method()
            outputs.append(actions)
        finally:
            backend.close()
    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-6, rtol=0)
