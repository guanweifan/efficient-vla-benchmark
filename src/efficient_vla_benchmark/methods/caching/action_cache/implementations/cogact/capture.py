"""Offline native-trajectory capture; success decisions belong to the rollout caller."""

import torch

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.forward_patch import ForwardPatches

from ...projection import key_for, projection


def native_sampler(backend):
    action_model = backend.model.action_model
    if action_model.ddim_diffusion is None:
        action_model.create_ddim(ddim_step=backend.parameters.num_ddim_steps)
    sampler = action_model.ddim_diffusion
    if sampler.num_timesteps != backend.parameters.num_ddim_steps:
        raise ValueError("Native sampler schedule differs from backend parameters")
    return sampler


class NativeActionCapture(InferenceMethod):
    name = "none"

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.sampler = native_sampler(backend)
        self.matrix = projection(
            4096, config.key_dim, config.projection_density, config.projection_seed, backend.device
        )
        self.patches = ForwardPatches()
        self.patches.wrap(self.sampler, self._capture, "ddim_sample")
        self.reset(None)

    def _capture(self, original):
        def sample(model, x, t, *args, **kwargs):
            if self.current_key is None:
                self.current_key = key_for(kwargs["model_kwargs"]["z"], self.matrix).cpu()
            self.current_states.append(x[0].detach().cpu().clone())
            return original(model, x, t, *args, **kwargs)

        return sample

    def reset(self, context):
        self.pending = []
        self.current_key = None
        self.current_states = []

    def before_query(self, prepared):
        self.current_key = None
        self.current_states = []

    def after_query(self, actions):
        if len(self.current_states) != self.sampler.num_timesteps:
            raise RuntimeError("Incomplete native trajectory capture")
        self.pending.append((self.current_key, torch.stack(self.current_states)))
        self.current_states = []

    def close(self):
        self.patches.close()
        self.reset(None)

    def metadata(self):
        return {
            "scope": "offline native trajectory capture; no acceleration",
            "cache_population_only": True,
        }
