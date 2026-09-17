from dataclasses import replace
from types import MethodType

import numpy as np
import pytest
import torch


@pytest.mark.gpu
@pytest.mark.parametrize("attention_backend", ["eager", "sdpa"])
def test_vla_cache_real_cold_parity_reuse_and_reset(model_path, trace_path, attention_backend):
    from efficient_vla_backend_openvla_oft.backend import OpenVLAOFTBackend

    from efficient_vla_benchmark.evaluation.traces import load_trace
    from efficient_vla_benchmark.methods.caching.vla_cache.config import VLACacheConfig

    from ..implementations.openvla_oft.method import (
        VLACache,
    )

    context, samples, _ = load_trace(trace_path, require_control_steps=True)
    backend = OpenVLAOFTBackend(model_path)
    backend.reset(context)
    originals = []
    if attention_backend == "eager":
        from transformers.models.llama.modeling_llama import LlamaAttention

        # Independent reference: the author's stock eager LlamaAttention, not
        # the adapter under test. These unpadded cold observations need no mask.
        for layer in backend.model.language_model.model.layers:
            attention = layer.self_attn
            originals.append((attention, attention.forward))
            attention.forward = MethodType(LlamaAttention.forward, attention)
    reference = backend.predict_chunk(samples[0])
    for attention, forward in originals:
        attention.forward = forward
    method = VLACache(backend, VLACacheConfig.parse({"attention_backend": attention_backend}))
    backend.attach_method(method)
    backend.reset(context)
    np.testing.assert_allclose(backend.predict_chunk(samples[0]), reference, atol=1e-6, rtol=0)
    assert backend.query_diagnostics["cache_cold"]
    reused = 0
    for sample in samples[8:32:8]:
        previous_keys = method.store.keys[31].clone()
        actions = backend.predict_chunk(sample)
        assert np.isfinite(actions).all()
        assert not backend.query_diagnostics["cache_cold"]
        dropped = method.full_positions[
            ~torch.isin(method.full_positions, method.adapter.state.positions)
        ]
        if dropped.numel():
            torch.testing.assert_close(
                method.store.keys[31][:, :, dropped], previous_keys[:, :, dropped], atol=0, rtol=0
            )
            reused += dropped.numel()
        assert all(
            value.shape[2] == method.last_layout.length for value in method.store.keys.values()
        )
    assert reused > 0
    backend.predict_chunk(replace(samples[0], instruction="pick up the object carefully"))
    assert backend.query_diagnostics["cache_cold"]
    backend.reset(context)
    assert not method.store.keys and method.previous_images is None
    np.testing.assert_allclose(backend.predict_chunk(samples[0]), reference, atol=1e-6, rtol=0)
    backend.close()
