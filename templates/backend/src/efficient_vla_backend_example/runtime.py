"""Small deterministic fixture for testing plugin contracts without model packages."""

import numpy as np

from efficient_vla_benchmark.core.lifecycle import MethodHost


class Monitor:
    def synchronize(self):
        pass

    def reset_peak_memory(self):
        pass

    def peak_memory(self):
        return {"allocated_bytes": None, "reserved_bytes": None}

    def hardware(self):
        return {"runtime": "cpu_contract_fixture", "performance_evidence": False}


class ExampleBackend(MethodHost):
    def __init__(self):
        self.initialize_method_state()
        self.runtime = Monitor()
        self.action_scale = 1.0

    def reset(self, context):
        self._ensure_open()
        self.query_diagnostics = {}
        if self.method is not None:
            self.method.reset(context)

    def predict_chunk(self, observation):
        self._ensure_open()
        observation.validate()
        return np.full((32, 3), self.action_scale, dtype=np.float32)

    def metadata(self):
        return {
            "backend": "example",
            "model_variant": "long_chunk",
            "method": self.method_name,
            "predicted_horizon": 32,
            "action_dim": 3,
            "contract_fixture": True,
        }


def build(config, resources):
    if resources.device != "cpu":
        raise ValueError("Example fixture is CPU-only")
    return ExampleBackend()
