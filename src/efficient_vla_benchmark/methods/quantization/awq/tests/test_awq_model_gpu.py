import gc

import numpy as np
import pytest
import torch

from efficient_vla_benchmark.core.resources import project_source


@pytest.mark.gpu
def test_awq_real_checkpoint_packed_scope_and_independent_reload(
    model_path, trace_path, artifact_path
):
    from efficient_vla_benchmark.core.config import ExperimentConfig
    from efficient_vla_benchmark.core.registry import build_backend
    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..implementations.openvla_oft.vendor import (
        reference_module,
    )

    config = ExperimentConfig.load(project_source("configs/experiments/openvla_oft") / "awq.yaml")
    context, samples, _ = load_trace(trace_path)
    WQLinear = reference_module("quantize.qmodule").WQLinear
    outputs = []
    for _ in range(2):
        backend = build_backend(config, model_path, "cuda:0", artifact_path)
        decoder = backend.model.language_model.model
        assert sum(isinstance(module, WQLinear) for module in decoder.modules()) == 224
        assert not any(
            isinstance(module, WQLinear) for module in backend.model.vision_backbone.modules()
        )
        assert not any(isinstance(module, WQLinear) for module in backend.action_head.modules())
        backend.reset(context)
        predictions = [
            backend.predict_chunk(sample)
            for sample in (samples[0], samples[len(samples) // 2], samples[-1])
        ]
        assert all(
            prediction.shape == (8, 7) and np.isfinite(prediction).all()
            for prediction in predictions
        )
        assert backend.metadata()["method_details"]["packed_linears"] == 224
        with pytest.raises(ValueError, match="fresh backend"):
            backend.detach_method()
        outputs.append(predictions)
        backend.close()
        del decoder, backend
        gc.collect()
        torch.cuda.empty_cache()
    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-6, rtol=0)
