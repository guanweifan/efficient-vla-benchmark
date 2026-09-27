"""Expand only the native action sampler; keep VLM and action conversion untouched."""

from dataclasses import asdict

import torch

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.forward_patch import ForwardPatches

from ...selection import select_index


class KeyStone(InferenceMethod):
    name = "keystone"

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.patches = ForwardPatches()
        self.closed = False
        self.reset(None)
        self.sampler = None
        if config.enabled:
            action_model = backend.model.action_model
            if action_model.net.training:
                raise ValueError("KeyStone requires an eval-mode action head")
            if action_model.ddim_diffusion is None:
                action_model.create_ddim(ddim_step=backend.parameters.num_ddim_steps)
            self.sampler = action_model.ddim_diffusion
            if self.sampler.num_timesteps != backend.parameters.num_ddim_steps:
                raise ValueError("Native DDIM schedule differs from backend configuration")
            if hasattr(self.sampler, "_evb_keystone"):
                raise ValueError("KeyStone already attached to this sampler")
            self.sampler._evb_keystone = self
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
                raise RuntimeError("KeyStone requires a nonnested inference policy query")
            if (
                noise is None
                or clip_denoised
                or eta != 0
                or progress
                or cond_fn is not None
                or denoised_fn is not None
                or not isinstance(model_kwargs, dict)
                or "z" not in model_kwargs
            ):
                raise ValueError("Unsupported nonnative DDIM options")
            cfg = model_kwargs.get("cfg_scale", 1.0) > 1.0
            rows = 2 if cfg else 1
            z = model_kwargs["z"]
            if tuple(shape) != tuple(noise.shape) or noise.ndim != 3:
                raise ValueError("Invalid native noise geometry")
            if noise.shape[0] != rows or z.shape[0] != rows:
                raise ValueError("KeyStone supports one native observation at a time")
            if cfg and not torch.equal(noise[0], noise[1]):
                raise ValueError("Native CFG noise branches must be paired")
            k = self.config.num_samples
            # Retain the native first draw, then generate K-1 independent samples.
            # DDIM eta=0 still consumes random draws internally; never reseed here.
            extra = torch.randn((k - 1, *noise.shape[1:]), device=noise.device).to(noise.dtype)
            expanded_noise = torch.cat((noise[:1], extra), dim=0)
            expanded_z = z[:1].expand(k, *z.shape[1:])
            if cfg:
                expanded_noise = torch.cat((expanded_noise, expanded_noise), dim=0)
                expanded_z = torch.cat((expanded_z, z[1:2].expand(k, *z.shape[1:])), dim=0)
            kwargs = dict(model_kwargs, z=expanded_z)
            samples = original(
                model,
                expanded_noise.shape,
                noise=expanded_noise,
                clip_denoised=False,
                denoised_fn=None,
                cond_fn=None,
                model_kwargs=kwargs,
                device=device,
                progress=False,
                eta=0.0,
            )
            if samples.shape != expanded_noise.shape:
                raise RuntimeError("Native sampler returned unexpected candidate geometry")
            index, clustered = select_index(samples[:k], self.config)
            self.backend.query_diagnostics.update(
                action_candidates=k,
                action_selection_index=int(index),
                action_selection_clustered=clustered,
                action_selection_horizon=self.config.selection_horizon,
                denoiser_calls=self.sampler.num_timesteps,
                denoiser_candidate_evaluations=k * self.sampler.num_timesteps,
                action_head_batch=k * rows,
                shared_context_evaluations=1,
            )
            chosen = samples[index : index + 1]
            if cfg:
                chosen = torch.cat((chosen, samples[k + index : k + index + 1]), dim=0)
            return chosen

        return sample

    def reset(self, context):
        self.query_open = False

    def before_query(self, prepared):
        if self.closed or self.query_open:
            raise RuntimeError("Closed or unfinished KeyStone query")
        self.query_open = True
        self.backend.query_diagnostics = {}

    def after_query(self, actions):
        self.query_open = False

    def close(self):
        self.patches.close()
        if self.sampler is not None and getattr(self.sampler, "_evb_keystone", None) is self:
            del self.sampler._evb_keystone
        self.reset(None)
        self.closed = True

    def metadata(self):
        return dict(
            scope="KeyStone shared-context parallel sampling and guarded cluster medoid",
            implementation_basis="paper equations 3-6 and pinned author selector conventions",
            reference_revision="29a15a606f7a32251ef48ac66c281e10c6785eb0",
            config=asdict(self.config),
            requires_training=False,
            selection_space="native normalized pre-clipping denoiser output; executed prefix",
            kmeans_initialization="first C candidates, as released author code",
            sampler="unchanged native DDIM schedule and CFG; K independent action noises",
            excluded=["learned scoring", "extra VLM encoding", "sampling-step reduction"],
        )
