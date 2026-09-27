import numpy as np


def check_native(backend, config, artifact_path, trace_path, length):
    import torch

    from efficient_vla_benchmark.control.chunk import ChunkPolicy
    from efficient_vla_benchmark.evaluation.traces import load_trace

    from ..method import PACE

    context, frames, _ = load_trace(trace_path, require_control_steps=True)
    backend.reset(context)
    native = backend.predict_chunk(frames[0])
    rng = torch.cuda.get_rng_state().clone()
    method = PACE(config, artifact_path, length)
    backend.attach_method(method)
    for _ in range(2):
        backend.reset(context)
        backend.observe(frames[0])
        actual = backend.predict_chunk(frames[0])
        np.testing.assert_array_equal(actual, native)
        assert torch.equal(rng, torch.cuda.get_rng_state())
        assert 1 <= backend.select_execution_horizon(actual, 8) <= length
    policy = ChunkPolicy(backend, 8)
    policy.reset(context)
    decisions = [policy.step(frame) for frame in frames[:32]]
    assert policy.query_count == sum(d.queried for d in decisions)
    assert all(1 <= d.diagnostics["selected_execution_horizon"] <= length for d in decisions)
    backend.detach_method()
    backend.reset(context)
    np.testing.assert_array_equal(backend.predict_chunk(frames[0]), native)
