"""Equation 1 ranking and reversible physical decoder compaction."""

import json
import math
import time
from dataclasses import asdict

import torch
from torch import nn
from torch.nn import functional as F

from efficient_vla_benchmark.calibration.libero import training_observations
from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.core.provenance import source_metadata, stable_hash


def ranking(scores):
    if not scores or any(type(v) not in (int, float) or not math.isfinite(v) for v in scores):
        raise ValueError("Importance scores must be finite")
    return sorted(range(len(scores)), key=lambda index: (scores[index], index))


def read_artifact(root, config, backend_id):
    if root is None or not (root / "manifest.json").is_file():
        raise ValueError("Layer pruning needs a completed training-calibration artifact")
    result = json.loads((root / "manifest.json").read_text())
    if (
        result.get("schema_version") != 1
        or result.get("algorithm") != "efficient_vla_layer_pruning"
        or result.get("complete") is not True
        or result.get("backend") != backend_id
        or result.get("calibration_parameters") != config.calibration_parameters()
        or len(result.get("samples", [])) != config.calibration_samples
        or len(result.get("scores", [])) != result.get("num_layers")
    ):
        raise ValueError("Incompatible or incomplete layer-pruning artifact")
    ranking(result["scores"])
    if config.prune_layers >= result["num_layers"]:
        raise ValueError("Keep at least one decoder layer")
    return result


@torch.inference_mode()
def calibrate(backend, decoder, config, resources, backend_id):
    if backend.method is not None:
        raise ValueError("Calibrate an unmodified native policy")
    output = resources.artifact_dir
    if output.exists() and any(output.iterdir()):
        raise ValueError("Calibration output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    handles, inputs, current, observations, identities = [], {}, {}, [], []
    start = time.perf_counter()

    def before(index):
        def hook(module, args):
            if index in inputs:
                raise RuntimeError("Each calibration layer must execute once per sample")
            inputs[index] = args[0].detach()

        return hook

    def after(index):
        def hook(module, args, result):
            x = inputs.pop(index)
            current[index] = 1 - F.cosine_similarity(x.float(), result[0].float(), dim=-1).mean()

        return hook

    try:
        for index, layer in enumerate(decoder.layers):
            handles.append(layer.register_forward_pre_hook(before(index)))
            handles.append(layer.register_forward_hook(after(index)))
        for context, observation, identity in training_observations(
            resources.dataset_path, config.calibration_samples, config.calibration_seed
        ):
            current.clear()
            backend.reset(context)
            backend.predict_chunk(observation)
            if inputs or len(current) != len(decoder.layers):
                raise RuntimeError("Incomplete native layer calibration")
            observations.append(torch.stack([current[i] for i in range(len(decoder.layers))]).cpu())
            identities.append(identity)
            print(
                json.dumps({"stage": "layer_calibration", "samples": len(identities)}), flush=True
            )
    finally:
        for handle in handles:
            handle.remove()
        inputs.clear()
    scores = torch.stack(observations).mean(0).tolist()
    result = {
        "schema_version": 1,
        "algorithm": "efficient_vla_layer_pruning",
        "complete": True,
        "backend": backend_id,
        "checkpoint_sha256": backend._model_manifest["sha256"],
        "precision": str(next(decoder.parameters()).dtype),
        "num_layers": len(decoder.layers),
        "scores": scores,
        "ranking": ranking(scores),
        "calibration_parameters": config.calibration_parameters(),
        "samples": identities,
        "source": source_metadata(),
        "seconds": time.perf_counter() - start,
        "reduction": "mean within all native token positions, then equal sample mean",
    }
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True))
    temporary.replace(output / "manifest.json")
    return result


class LayerPruning(InferenceMethod):
    name = "efficient_vla_layer_pruning"

    def __init__(self, backend, decoder, config, resources, backend_id):
        self.backend, self.decoder, self.config = backend, decoder, config
        self.original = None
        self.removed = []
        self.artifact_digest = None
        self.device = next(decoder.parameters()).device
        if not config.enabled or config.prune_layers == 0:
            return
        manifest = read_artifact(resources.artifact_path, config, backend_id)
        if (
            manifest["checkpoint_sha256"] != backend._model_manifest["sha256"]
            or manifest["precision"] != str(next(decoder.parameters()).dtype)
            or manifest["num_layers"] != len(decoder.layers)
        ):
            raise ValueError("Layer scores belong to a different checkpoint or precision")
        if getattr(decoder, "_evb_compacted", False):
            raise ValueError("Decoder is already compacted")
        self.artifact_digest = stable_hash(manifest)
        self.removed = sorted(ranking(manifest["scores"])[: config.prune_layers])
        self.original = decoder.layers
        self.original_depth = decoder.config.num_hidden_layers
        self.original_indices = [layer.self_attn.layer_idx for layer in self.original]
        try:
            decoder.layers = nn.ModuleList(
                layer for index, layer in enumerate(self.original) if index not in self.removed
            )
            for index, layer in enumerate(decoder.layers):
                layer.self_attn.layer_idx = index
            decoder.config.num_hidden_layers = len(decoder.layers)
            decoder._evb_compacted = True
            for index in self.removed:
                self.original[index].cpu()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            self.close()
            raise

    def before_query(self, prepared):
        self.backend.query_diagnostics = {
            "decoder_layers_executed": len(self.decoder.layers),
            "removed_original_layer_indices": self.removed,
        }

    def close(self):
        if self.original is not None:
            for index, layer in enumerate(self.original):
                layer.to(self.device)
                layer.self_attn.layer_idx = self.original_indices[index]
            self.decoder.layers = self.original
            self.decoder.config.num_hidden_layers = self.original_depth
            delattr(self.decoder, "_evb_compacted")
            self.original = None

    def metadata(self):
        return {
            "scope": "EfficientVLA independent static non-contiguous layer-pruning core",
            "reference": "arXiv:2506.10100v1 Section 3.2, equation 1",
            "exclusions": "MLP compression, token pruning, DiT caching and dynamic routing",
            "taxonomy": "3.3 Model Compression; adjacent 5.3 depth-reduction reference",
            "pruning": "remove least-changing layers; retain original order of survivors",
            "storage": "removed layers offloaded to CPU for reversible detach, not GPU-resident",
            "removed_original_layer_indices": self.removed,
            "calibration_artifact_sha256": self.artifact_digest,
            "config": asdict(self.config),
        }
