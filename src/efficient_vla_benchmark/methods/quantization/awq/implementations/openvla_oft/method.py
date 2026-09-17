from efficient_vla_benchmark.core.method import InferenceMethod

from .artifact import load_artifact


class AWQ(InferenceMethod):
    name = "awq"
    reversible = False

    def __init__(self, backend, config, artifact_path):
        self.backend, self.config = backend, config
        self.artifact_sha256 = load_artifact(backend, artifact_path, config)

    def metadata(self):
        return {
            "artifact_sha256": self.artifact_sha256,
            "weight_bits": self.config.bits,
            "group_size": self.config.group_size,
            "scope": "language_decoder_block_linears",
            "packed_linears": 224,
            "calibration_samples": self.config.calibration_samples,
        }

    def before_query(self, prepared):
        self.backend.query_diagnostics = {"packed_linears": 224, "weight_bits": 4}
