"""Pinned author GGUF engine behind OFT's generic execution-runtime interface."""

import gc
import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import torch
from efficient_vla_backend_openvla_oft.processor import process_actions
from efficient_vla_backend_openvla_oft.runtime import TorchCUDAMonitor

from ...artifact import WEIGHTS
from ...vendor import REVISION


class ExternalMonitor(TorchCUDAMonitor):
    def peak_memory(self):
        # Torch cannot account for the GGML CUDA allocator. Missing is not zero.
        return {"allocated_bytes": None, "reserved_bytes": None}

    def hardware(self):
        result = super().hardware()
        result.update(
            runtime="author_ggml_cuda", memory_accounting="external_allocator_unavailable"
        )
        return result


class AuthorRuntime:
    def __init__(self, backend, root, runtime_path, manifest):
        if backend.device.type != "cuda" or backend.device.index not in (None, 0):
            raise ValueError("Select one visible GPU and use cuda:0 for the author runtime")
        binary = (runtime_path / "build/bin/openvla_oft.so").resolve()
        loaded = sys.modules.get("openvla_oft")
        if loaded is not None and getattr(loaded, "__file__", None) != str(binary):
            raise ValueError("Conflicting author extension import")
        if loaded is None:
            spec = importlib.util.spec_from_file_location("openvla_oft", binary)
            loaded = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(loaded)
            sys.modules["openvla_oft"] = loaded
        self.factory = loaded.OpenvlaOFTPipelineWithProprio
        self.root, self.model_path = root, backend.model_path
        self.manifest = manifest
        self.pipeline, self.suite = None, None
        self.closed = False
        self.monitor = ExternalMonitor(backend.device)

    def reset(self, context):
        if self.closed:
            raise RuntimeError("External runtime is closed")
        if self.pipeline is not None and self.suite == context.suite:
            # Pinned forward_oft_v2 clears all KV memory before EVERY query.
            return
        self.pipeline = None
        gc.collect()
        self.pipeline = self.factory(
            *[str(self.root / name) for name in WEIGHTS],
            str(self.model_path / "dataset_statistics.json"),
            str(self.model_path / "tokenizer.json"),
            "CUDA0",
            torch.get_num_threads(),
            8192,
            99,
            1024,
            context.suite,
        )
        self.suite = context.suite

    def prepare(self, observation):
        from experiments.robot.openvla_utils import prepare_images_for_vla

        if self.closed or self.pipeline is None:
            raise RuntimeError("Reset the external runtime before inference")
        observation.validate()
        images = prepare_images_for_vla(
            [observation.image, observation.wrist_image], SimpleNamespace(center_crop=True)
        )
        return {
            "images": [np.ascontiguousarray(x) for x in images],
            "instruction": observation.instruction,
            "proprio": observation.proprio.tolist(),
        }

    def forward(self, prepared):
        if self.closed or self.pipeline is None:
            raise RuntimeError("External runtime is not ready")
        actions = (
            self.pipeline.run2(*prepared["images"], prepared["instruction"], prepared["proprio"])
            .reshape(8, 7)
            .copy()
        )
        return process_actions(actions)

    def close(self):
        self.pipeline = None
        self.closed = True
        gc.collect()

    def metadata(self):
        return {
            "engine": "author_ggml_cuda",
            "reference_revision": REVISION,
            "runtime_binary_sha256": self.manifest["runtime_binary_sha256"],
            "precision": self.manifest["precision"],
            "native_preprocessing": "center_crop_raw_proprio_single_gripper_conversion",
            "query_state": "author_unconditional_KV_clear",
            "native_modules": "CPU_only_not_executed",
            "torch_peak_memory": "unavailable_for_external_allocator",
        }
