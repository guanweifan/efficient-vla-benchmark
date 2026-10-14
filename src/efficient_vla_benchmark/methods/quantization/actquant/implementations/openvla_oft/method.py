from efficient_vla_benchmark.core.method import InferenceMethod

from ...artifact import identity, read_manifest
from ...config import ActQuantConfig
from .runtime import AuthorRuntime


class ActQuant(InferenceMethod):
    reversible = False

    def __init__(self, backend, config, resources):
        manifest = read_manifest(resources.artifact_path, config, resources.runtime_path)
        if (
            backend.method is not None
            or backend.metadata()["precision"] != "bfloat16"
            or backend.metadata()["checkpoint_sha256"] != manifest["checkpoint_sha256"]
        ):
            raise ValueError("ActQuant requires a fresh checkpoint-matched BF16 OFT")
        self.name = "actquant" if isinstance(config, ActQuantConfig) else "actquant_floating"
        runtime = AuthorRuntime(backend, resources.artifact_path, resources.runtime_path, manifest)
        try:
            backend.install_execution_runtime(runtime)
        except Exception:
            runtime.close()
            raise
        self.manifest = manifest

    def metadata(self):
        return {
            "artifact_sha256": identity(self.manifest),
            "scope": self.manifest["scope"],
            "quantized_tensors": self.manifest.get("quantized_tensors", 0),
            "allocation_bpw": self.manifest.get("achieved_bpw"),
            "calibration_sha256": self.manifest.get("calibration_sha256"),
            "precision": self.manifest["precision"],
            "runtime": "pinned_author_gguf_cuda",
            "comparison": "native_BF16_primary_floating_GGUF_auxiliary",
        }
