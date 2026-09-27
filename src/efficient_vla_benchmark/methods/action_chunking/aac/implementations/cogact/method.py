"""AAC uses one cognition encoding and real independently sampled native action chunks."""

from dataclasses import asdict

import numpy as np
import torch

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.candidate_sampling import expand_candidates
from efficient_vla_benchmark.model_ops.forward_patch import ForwardPatches

from ...selection import select_prefix


def native_commands(samples, statistics):
    """Mirror native postprocessing solely for AAC's entropy/movement inspection."""
    normalized = np.clip(np.asarray(samples), -1, 1)
    normalized[..., 6] = np.where(normalized[..., 6] < 0.5, 0, 1)
    low, high = np.asarray(statistics["q01"]), np.asarray(statistics["q99"])
    mask = np.asarray(statistics.get("mask", np.ones_like(low, dtype=bool)))
    if low.shape != (7,) or high.shape != (7,) or mask.shape != (7,) or mask[6]:
        raise ValueError("AAC requires native seven-dimensional deltas and unscaled binary gripper")
    commands = np.where(mask, 0.5 * (normalized + 1) * (high - low) + low, normalized)
    # Native gripper is open=1; AAC's reference uses close=1. Complementing it does
    # not alter Bernoulli entropy or the toggle lower bound, but keep labels explicit.
    commands[..., 6] = 1 - commands[..., 6]
    return commands


class AAC(InferenceMethod):
    name = "aac"
    requires_control_history = True

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.patches = ForwardPatches()
        self.closed, self.sampler = False, None
        self.reset(None)
        if config.enabled:
            action_model = backend.model.action_model
            if action_model.net.training:
                raise ValueError("AAC requires an eval-mode action head")
            if action_model.ddim_diffusion is None:
                action_model.create_ddim(ddim_step=backend.parameters.num_ddim_steps)
            self.sampler = action_model.ddim_diffusion
            if self.sampler.num_timesteps != backend.parameters.num_ddim_steps:
                raise ValueError("AAC native DDIM schedule differs from configured sampler")
            if hasattr(self.sampler, "_evb_aac"):
                raise ValueError("AAC is already attached")
            self.sampler._evb_aac = self
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
                raise RuntimeError("AAC requires a nonnested inference query")
            if (
                noise is None
                or tuple(shape) != tuple(noise.shape)
                or tuple(noise.shape[1:]) != (16, 7)
                or clip_denoised
                or eta != 0
                or progress
                or cond_fn is not None
                or denoised_fn is not None
                or not isinstance(model_kwargs, dict)
                or "z" not in model_kwargs
            ):
                raise ValueError("Unsupported nonnative AAC sampler options")
            using_cfg = model_kwargs.get("cfg_scale", 1.0) > 1.0
            k = self.config.num_samples
            batch, condition = expand_candidates(noise, model_kwargs["z"], k, using_cfg)
            samples = original(
                model,
                batch.shape,
                noise=batch,
                clip_denoised=False,
                denoised_fn=None,
                cond_fn=None,
                model_kwargs=dict(model_kwargs, z=condition),
                device=device,
                progress=False,
                eta=0.0,
            )
            if samples.shape != batch.shape:
                raise RuntimeError("AAC native sampler changed candidate geometry")
            normalized = samples[:k].cpu().numpy()
            commands = native_commands(
                normalized, self.backend.model.get_action_stats(self.backend.unnorm_key)
            )
            self.horizon, selection = select_prefix(normalized, commands, self.config)
            self.diagnostics = dict(
                horizon_policy="adaptive",
                aac_entropy_horizon=selection["entropy_horizon"],
                aac_movement_horizon=selection["movement_horizon"],
            )
            self.backend.query_diagnostics.update(
                action_candidates=k,
                action_selection_index=0,
                denoiser_calls=self.sampler.num_timesteps,
                denoiser_candidate_evaluations=k * self.sampler.num_timesteps,
                action_head_batch=k * (2 if using_cfg else 1),
                shared_context_evaluations=1,
            )
            # The author client serves candidate zero; no extra medoid or averaging.
            return torch.cat((samples[:1], samples[k : k + 1]), dim=0) if using_cfg else samples[:1]

        return sample

    def before_query(self, prepared):
        if self.closed or self.query_open:
            raise RuntimeError("Closed or unfinished AAC query")
        self.query_open, self.horizon = True, None
        self.diagnostics = {}
        self.backend.query_diagnostics = {}

    def after_query(self, actions):
        self.query_open = False
        if self.config.enabled and self.horizon is None:
            raise RuntimeError("AAC candidate sampler did not produce an execution horizon")

    def select_execution_horizon(self, actions, default_horizon):
        if not self.config.enabled:
            return default_horizon
        if self.closed or self.query_open or self.horizon is None:
            raise RuntimeError("AAC requires a completed current query")
        if actions.shape != (16, 7):
            raise ValueError("AAC cannot change native prediction geometry")
        return self.horizon

    def control_diagnostics(self):
        return dict(self.diagnostics)

    def reset(self, context):
        self.query_open, self.horizon = False, None
        self.diagnostics = {}

    def close(self):
        self.patches.close()
        if self.sampler is not None and getattr(self.sampler, "_evb_aac", None) is self:
            del self.sampler._evb_aac
        self.reset(None)
        self.closed = True

    def metadata(self):
        return dict(
            scope="AAC empirical action entropy elbow with cumulative movement lower bound",
            implementation_basis="paper equations 2-10 and pinned released LIBERO selector",
            reference_revision="ff8c8e421f7805b5845a72d07fbd8f1171df74bf",
            config=asdict(self.config),
            requires_training=False,
            execution_horizon_policy="adaptive" if self.config.enabled else "native",
            entropy_space="native normalized pre-clipping pose; binary native gripper",
            movement_space="native denormalized delta commands; ordered axis-angle composition",
            served_candidate=0,
            context_encodings_per_query=1,
            sampler="unchanged native DDIM, CFG, and sixteen predictions",
            excluded=["GR00T retraining", "backward candidate selection", "sampler-step reduction"],
            conventions="released gripper motion weight 0.2; paper appendix instead writes 1",
        )
