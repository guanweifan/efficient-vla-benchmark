"""Check wrapper/native parity on one real recorded LIBERO observation."""

import argparse
import json
from pathlib import Path

import numpy as np
from efficient_vla_backend_cogact.backend import CogACTBackend
from efficient_vla_backend_cogact.processor import process_actions

from efficient_vla_benchmark.evaluation.traces import load_trace


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
        wrapped = backend.predict_chunk(observations[0])
        backend.reset(context)
        native, _ = backend.model.predict_action(
            **backend.prepare(observations[0]),
            unnorm_key=backend.unnorm_key,
            cfg_scale=1.5,
            use_ddim=True,
            num_ddim_steps=10,
            do_sample=False,
        )
        expected = process_actions(native)
        np.testing.assert_array_equal(wrapped, expected)
        backend.reset(context)
        repeated = backend.predict_chunk(observations[0])
        np.testing.assert_array_equal(wrapped, repeated)
        print(
            json.dumps(
                {
                    "native_parity": True,
                    "reset_reproducible": True,
                    "maximum_absolute_error": float(np.max(np.abs(wrapped - expected))),
                    "shape": list(wrapped.shape),
                    "metadata": backend.metadata(),
                },
                indent=2,
            )
        )
    finally:
        backend.close()
        backend.close()


if __name__ == "__main__":
    main()
