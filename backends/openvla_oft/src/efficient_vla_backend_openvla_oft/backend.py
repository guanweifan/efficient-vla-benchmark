"""Read-only native OFT assembly behind the optional backend interface."""

import gc
import json
from pathlib import Path

import torch

from efficient_vla_benchmark.core.lifecycle import MethodHost

from .config import validate
from .model_access import model_manifest
from .processor import prepare_observation, process_actions
from .runtime import TorchCUDAMonitor


def build(config, resources):
    validate(config)
    precision = "bfloat16" if config.precision == "auto" else config.precision
    return OpenVLAOFTBackend(resources.model_path, resources.device, precision)


class OpenVLAOFTBackend(MethodHost):
    def __init__(self, model_path: Path, device="cuda:0", precision="bfloat16"):
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import (
            PrismaticImageProcessor,
            PrismaticProcessor,
        )
        from prismatic.models.action_heads import L1RegressionActionHead
        from prismatic.models.projectors import ProprioProjector
        from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
        from transformers import AutoTokenizer

        if (NUM_ACTIONS_CHUNK, ACTION_DIM, PROPRIO_DIM) != (8, 7, 8):
            raise ValueError("The upstream constants must select the LIBERO platform")
        if not model_path.is_dir():
            raise ValueError("The supplied checkpoint directory does not exist")
        if precision not in {"bfloat16", "float16"}:
            raise ValueError("Unsupported OFT model precision")
        self.initialize_method_state()
        self.device = torch.device(device)
        self.dtype = getattr(torch, precision)
        self.precision = precision
        self.context = None
        self.model_path = model_path
        self._model_manifest = model_manifest(model_path)
        self.runtime = TorchCUDAMonitor(self.device)
        config = OpenVLAConfig.from_pretrained(model_path, local_files_only=True)
        self.model = (
            OpenVLAForActionPrediction.from_pretrained(
                model_path,
                config=config,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
                attn_implementation="sdpa",
            )
            .eval()
            .to(self.device)
        )
        self.model.vision_backbone.set_num_images_in_input(2)
        self.model.norm_stats = json.loads((model_path / "dataset_statistics.json").read_text())
        self.processor = PrismaticProcessor(
            image_processor=PrismaticImageProcessor.from_pretrained(
                model_path, local_files_only=True
            ),
            tokenizer=AutoTokenizer.from_pretrained(
                model_path, local_files_only=True, trust_remote_code=False
            ),
        )
        self.action_head = L1RegressionActionHead(
            input_dim=self.model.llm_dim, hidden_dim=self.model.llm_dim, action_dim=7
        )
        self.proprio_projector = ProprioProjector(llm_dim=self.model.llm_dim, proprio_dim=8)
        self._load_component(self.action_head, "action_head")
        self._load_component(self.proprio_projector, "proprio_projector")

    def _load_component(self, module, prefix):
        matches = sorted(self.model_path.glob(f"{prefix}--*_checkpoint.pt"))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one checkpoint component: {prefix}")
        state = torch.load(matches[0], map_location="cpu", weights_only=True)
        module.load_state_dict(
            {key.removeprefix("module."): value for key, value in state.items()}, strict=True
        )
        module.eval().to(device=self.device, dtype=self.dtype)

    def reset(self, context):
        self._ensure_open()
        torch.manual_seed(context.seed)
        torch.cuda.manual_seed_all(context.seed)
        key = context.suite
        if key not in self.model.norm_stats:
            key += "_no_noops"
        if key not in self.model.norm_stats:
            raise ValueError("The checkpoint lacks normalization statistics for this suite")
        self.context = context
        self.unnorm_key = key
        self.query_diagnostics = {}
        if self.method is not None:
            self.method.reset(context)

    def prepare(self, observation):
        self._ensure_open()
        return prepare_observation(self, observation)

    @torch.inference_mode()
    def forward(self, prepared):
        self._ensure_open()
        if self.method is not None:
            self.method.before_query(prepared)
        actions, _ = self.model.predict_action(
            **prepared["inputs"],
            unnorm_key=self.unnorm_key,
            do_sample=False,
            proprio=prepared["proprio"],
            proprio_projector=self.proprio_projector,
            action_head=self.action_head,
            use_film=False,
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
            self.action_head = None
            self.proprio_projector = None
            self.processor = None
            gc.collect()
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

    def metadata(self):
        return {
            "backend": "openvla_oft",
            "model_variant": "libero_joint",
            "method": self.method_name,
            "precision": self.precision,
            "attention_implementation": self.model.config._attn_implementation,
            "predicted_horizon": 8,
            "action_dim": 7,
            "camera_views": 2,
            "image_size": 224,
            "checkpoint_sha256": self._model_manifest["sha256"],
            "requires_control_history": bool(self.method and self.method.requires_control_history),
            "method_details": self.method.metadata() if self.method else {},
        }
