"""Validate active causal hooks against a real native CogACT checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
from efficient_vla_backend_cogact.backend import CogACTBackend
from efficient_vla_backend_cogact.decoder import TokenStrategy

from efficient_vla_benchmark.evaluation.traces import load_trace


class FullRetention(TokenStrategy):
    def before_layer(self, index, state):
        if index == 0:
            return state.positions >= 0
        return None


class FixedSparse(TokenStrategy):
    def before_layer(self, index, state):
        if index == 0:
            return ~state.layout.visual_mask(state.positions) | (state.positions % 2 == 1)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    context, observations, _ = load_trace(args.trace)
    backend = CogACTBackend(args.model_path, args.device)
    try:
        backend.reset(context)
        expected = backend.predict_chunk(observations[0])
        attention_modules = [layer.self_attn for layer in backend.model_access.decoder.layers]
        adapter = backend.model_access.attach_token_strategy(FullRetention())
        try:
            backend.reset(context)
            full = backend.predict_chunk(observations[0])
            np.testing.assert_array_equal(full, expected)
            assert attention_modules == [
                layer.self_attn for layer in backend.model_access.decoder.layers
            ]
            lengths = adapter.layer_sequence_lengths
            assert len(lengths) == 32 and len(set(lengths)) == 1
        finally:
            adapter.close()
        adapter = backend.model_access.attach_token_strategy(FixedSparse())
        try:
            backend.reset(context)
            sparse = backend.predict_chunk(observations[0])
            assert sparse.shape == (16, 7) and np.isfinite(sparse).all()
            backend.reset(context)
            np.testing.assert_array_equal(backend.predict_chunk(observations[0]), sparse)
            assert adapter.layer_sequence_lengths == [lengths[0] - 128] * 32
        finally:
            adapter.close()
        backend.reset(context)
        np.testing.assert_array_equal(backend.predict_chunk(observations[0]), expected)
        print(
            json.dumps(
                {
                    "full_retention_active_hooks_exact_native_parity": True,
                    "native_attention_modules_unchanged": True,
                    "sparse_128_tokens_finite_and_reset_reproducible": True,
                    "detached_exact_native_parity": True,
                    "max_full_retention_error": float(np.max(np.abs(full - expected))),
                },
                indent=2,
            )
        )
    finally:
        backend.close()


if __name__ == "__main__":
    main()
