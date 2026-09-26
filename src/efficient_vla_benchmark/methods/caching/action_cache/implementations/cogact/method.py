"""Retrieve genuine intermediate noisy actions and resume the native DDIM tail."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.forward_patch import ForwardPatches

from ...artifact import read_manifest
from ...projection import key_for, projection
from .capture import native_sampler


class ActionCache(InferenceMethod):
    name = "action_cache"

    def __init__(self, backend, config, artifact_path):
        self.backend, self.config = backend, config
        self.manifest = read_manifest(artifact_path, config)
        if self.manifest["checkpoint_sha256"] != backend.metadata()["checkpoint_sha256"]:
            raise ValueError("ActionCache bank belongs to another checkpoint")
        if self.manifest["sampler_parameters"] != asdict(backend.parameters):
            raise ValueError("ActionCache bank uses a different native sampler")
        self.sampler = native_sampler(backend)
        with np.load(Path(artifact_path) / "bank.npz", allow_pickle=False) as bank:
            keys, states, suites = bank["keys"], bank["states"], bank["suites"]
        steps = self.sampler.num_timesteps
        if (
            keys.ndim != 2
            or keys.shape[1] != config.key_dim
            or states.shape != (keys.shape[0], steps, 16, 7)
            or suites.shape != (keys.shape[0],)
            or not len(keys)
            or not np.isfinite(keys).all()
            or not np.isfinite(states).all()
            or keys.dtype != np.float32
            or states.dtype != np.float32
            or not np.issubdtype(suites.dtype, np.integer)
            or not np.isin(suites, np.arange(len(self.manifest["suites"]))).all()
            or not np.allclose(np.linalg.norm(keys, axis=1), 1, atol=1e-5)
            or config.refinement_steps >= steps
        ):
            raise ValueError("Invalid ActionCache bank geometry or values")
        self.banks = {}
        for index, suite in enumerate(self.manifest["suites"]):
            mask = suites == index
            if mask.any():
                self.banks[suite] = (
                    torch.tensor(keys[mask], device=backend.device),
                    torch.tensor(
                        states[mask, steps - config.refinement_steps], device=backend.device
                    ),
                )
        self.matrix = projection(
            4096, config.key_dim, config.projection_density, config.projection_seed, backend.device
        )
        self.patches = ForwardPatches()
        self.closed = False
        self.reset(None)
        if config.enabled:
            if hasattr(self.sampler, "_evb_action_cache"):
                raise ValueError("Only one action cache may wrap a sampler")
            self.sampler._evb_action_cache = self
            self.patches.wrap(self.sampler, self._loop, "ddim_sample_loop")

    def _loop(self, original):
        def sample(
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,
        ):
            if not self.query_open or torch.is_grad_enabled():
                raise RuntimeError("ActionCache requires an inference policy query")
            if (
                noise is None
                or eta != 0
                or progress
                or cond_fn is not None
                or denoised_fn is not None
            ):
                raise ValueError("Unsupported nonnative DDIM sampling options")
            key = key_for(model_kwargs["z"], self.matrix)
            bank = self.banks.get(self.suite)
            match, similarity = None, -1.0
            if bank is not None:
                scores = (bank[0] @ key).clamp(-1, 1)
                value, index = scores.max(0)
                similarity = float(value.item())
                if similarity > self.config.hit_threshold:
                    match = bank[1][index]
            self.backend.query_diagnostics.update(
                action_cache_hit=match is not None,
                action_cache_similarity=similarity,
                action_cache_partition_size=0 if bank is None else len(bank[0]),
                denoiser_evaluations=(
                    self.config.refinement_steps
                    if match is not None
                    else self.sampler.num_timesteps
                ),
            )
            if match is None:
                return original(
                    model,
                    shape,
                    noise=noise,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    device=device,
                    progress=progress,
                    eta=eta,
                )
            img = match.unsqueeze(0).expand(shape[0], -1, -1).clone()
            # Native eta=0 DDIM still draws unused per-step noise. Retain its RNG
            # consumption so future full-query noise is not shifted by cache hits.
            for _ in range(self.sampler.num_timesteps - self.config.refinement_steps):
                torch.randn_like(noise)
            for index in range(self.config.refinement_steps - 1, -1, -1):
                t = torch.full((shape[0],), index, device=img.device, dtype=torch.long)
                out = self.sampler.ddim_sample(
                    model, img, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs, eta=eta
                )
                img = out["sample"]
            return img

        return sample

    def reset(self, context):
        self.suite = None if context is None else context.suite
        self.query_open = False

    def before_query(self, prepared):
        if self.closed or self.query_open:
            raise RuntimeError("Closed or unfinished ActionCache query")
        self.query_open = True
        self.backend.query_diagnostics = {}

    def after_query(self, actions):
        self.query_open = False

    def close(self):
        self.patches.close()
        if getattr(self.sampler, "_evb_action_cache", None) is self:
            del self.sampler._evb_action_cache
        self.reset(None)
        self.closed = True

    def metadata(self):
        return dict(
            scope="ActionCache frozen-bank retrieval and native DDIM refinement",
            implementation_basis="paper section 3, DDIM-tail adaptation of flow warm starts",
            config=asdict(self.config),
            requires_training=False,
            key_source="native final cognition embedding; fixed balanced ternary projection",
            cache_policy="frozen successful-prefill bank; no evaluation-outcome writes",
            partition="normalization suite; cross-task retrieval within a suite",
            cached_object="actual normalized noisy actions before native tail transitions",
            artifact_sha256=self.manifest["bank_sha256"],
            population_digest=self.manifest["population_sha256"],
            excluded=["online population during evaluation", "zero-step action execution"],
        )
