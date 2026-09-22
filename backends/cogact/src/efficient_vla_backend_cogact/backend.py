"""Read-only native CogACT baseline behind the common policy contract."""

import gc
from dataclasses import asdict

import torch

from efficient_vla_benchmark.core.lifecycle import MethodHost

from .checkpoint import checkpoint_files, model_manifest
from .config import CogACTParameters, validate
from .loader import load_native
from .model_access import CogACTModelAccess
from .processor import prepare_observation, process_actions
from .runtime import TorchCUDAMonitor


def build(config, resources):
    validate(config)
    return CogACTBackend(
        resources.model_path,
        resources.device,
        CogACTParameters.from_mapping(config.backend_parameters),
    )


class CogACTBackend(MethodHost):
    def __init__(self, model_path, device="cuda:0", parameters=None):
        self.initialize_method_state()
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("The native CogACT runtime requires CUDA")
        self.parameters = parameters or CogACTParameters()
        root, checkpoint, statistics, self.normalization_keys = checkpoint_files(model_path)
        self._model_manifest = model_manifest(root, checkpoint)
        self.model = load_native(root, checkpoint, statistics, self.device)
        self.model_access = CogACTModelAccess(self.model)
        self.runtime = TorchCUDAMonitor(self.device)
        self.context = None
        self.unnorm_key = None

    def reset(self, context):
        self._ensure_open()
        if context.suite not in self.normalization_keys:
            raise ValueError("Unsupported suite for this checkpoint")
        torch.manual_seed(context.seed)
        torch.cuda.manual_seed_all(context.seed)
        self.context = context
        self.unnorm_key = self.normalization_keys[context.suite]
        self.query_diagnostics = {}
        if self.method is not None:
            self.method.reset(context)

    def prepare(self, observation):
        self._ensure_open()
        if self.context is None:
            raise RuntimeError("Reset the backend with an episode before prediction")
        return prepare_observation(observation)

    @torch.inference_mode()
    def forward(self, prepared):
        self._ensure_open()
        if self.context is None:
            raise RuntimeError("Reset the backend with an episode before prediction")
        if self.method is not None:
            self.method.before_query(prepared)
        actions, _ = self.model.predict_action(
            **prepared,
            unnorm_key=self.unnorm_key,
            cfg_scale=self.parameters.cfg_scale,
            use_ddim=True,
            num_ddim_steps=self.parameters.num_ddim_steps,
            do_sample=False,
        )
        actions = process_actions(actions)
        if self.method is not None:
            self.method.after_query(actions)
        return actions

    def predict_chunk(self, observation):
        return self.forward(self.prepare(observation))

    def close(self):
        if self._closed:
            return
        try:
            super().close()
        finally:
            self._closed = True
            self.method = None
            self.model = None
            self.model_access = None
            gc.collect()
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

    def metadata(self):
        return {
            "backend": "cogact",
            "model_variant": "libero_joint",
            "method": self.method_name,
            "precision": "bfloat16",
            "action_head_precision": "float32",
            "attention_implementation": "sdpa",
            "native_attention_implementation": "sdpa",
            "attention_mask": "causal",
            "predicted_horizon": 16,
            "action_dim": 7,
            "camera_views": 1,
            "proprio_input": False,
            "image_size": 224,
            "image_preprocessing": "native_resize_no_extra_crop",
            "sampler": "ddim",
            "sampler_parameters": asdict(self.parameters),
            "normalization_keys": self.normalization_keys,
            "action_ensemble": False,
            "checkpoint_sha256": self._model_manifest["sha256"],
            "requires_control_history": bool(self.method and self.method.requires_control_history),
            "method_details": self.method.metadata() if self.method else {},
        }
