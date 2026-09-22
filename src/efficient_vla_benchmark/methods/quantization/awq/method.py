from efficient_vla_benchmark.core.method import InferenceMethod

from .artifact import load_artifact


class AWQ(InferenceMethod):
    name = "awq"
    reversible = False

    def __init__(self, backend, config, artifact_path, target):
        self.backend, self.config = backend, config
        self.precision = target.precision
        self.kernel_precision = (
            target.precision if config.kernel_precision == "auto" else config.kernel_precision
        )
        self.artifact_sha256 = load_artifact(target, artifact_path, config)

    def metadata(self):
        return {
            "artifact_sha256": self.artifact_sha256,
            "weight_bits": self.config.bits,
            "group_size": self.config.group_size,
            "scope": "language_decoder_block_linears",
            "packed_linears": 224,
            "calibration_samples": self.config.calibration_samples,
            "kernel_precision": self.kernel_precision,
            "gemm_accumulation": "float16" if self.kernel_precision == "float16" else "float32",
            "kernel_boundary_casts": self.kernel_precision != self.precision,
        }

    def before_query(self, prepared):
        self.backend.query_diagnostics = {"packed_linears": 224, "weight_bits": 4}
