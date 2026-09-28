"""A3 official GR00T decision rules, with native CogACT conditional DDIM."""

from dataclasses import asdict

import torch

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.candidate_sampling import expand_candidates
from efficient_vla_benchmark.model_ops.forward_patch import ForwardPatches

from ...decoding import decode
from .sampling import conditional_ddim, make_sampler, native_commands


class A3Official(InferenceMethod):
    name = "a3_official"
    requires_control_history = True

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.patches = ForwardPatches()
        self.closed, self.sampler = False, None
        self.denoising_sampler = None
        self.reset(None)
        if config.enabled:
            action = backend.model.action_model
            if action.net.training:
                raise ValueError("A3 official-code requires an eval-mode action head")
            if action.ddim_diffusion is None:
                action.create_ddim(ddim_step=backend.parameters.num_ddim_steps)
            self.sampler = action.ddim_diffusion
            if self.sampler.num_timesteps != backend.parameters.num_ddim_steps:
                raise ValueError("A3 sampler differs from the configured native schedule")
            if hasattr(self.sampler, "_evb_a3_official"):
                raise ValueError("A3 official-code already attached")
            self.denoising_sampler = make_sampler(backend, config.num_denoising_steps)
            self.sampler._evb_a3_official = self
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
                raise RuntimeError("A3 requires a nonnested inference query")
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
                raise ValueError("Unsupported nonnative A3 sampler options")
            using_cfg = model_kwargs.get("cfg_scale", 1.0) > 1.0
            count = self.config.num_samples
            expanded, condition = expand_candidates(noise, model_kwargs["z"], count, using_cfg)
            draft_sample = (
                original
                if self.denoising_sampler is self.sampler
                else self.denoising_sampler.ddim_sample_loop
            )
            candidates = draft_sample(
                model,
                expanded.shape,
                noise=expanded,
                clip_denoised=False,
                model_kwargs=dict(model_kwargs, z=condition),
                device=device,
                progress=False,
                eta=0.0,
            )
            if candidates.shape != expanded.shape:
                raise RuntimeError("A3 candidate geometry changed")
            raw = candidates[:count].float().cpu().numpy()
            statistics = self.backend.model.get_action_stats(self.backend.unnorm_key)
            commands = native_commands(raw, statistics)

            def verify(fixed, mask):
                rows = fixed.shape[0]
                initial = torch.randn((rows, *noise.shape[1:]), device=noise.device).to(noise.dtype)
                context = model_kwargs["z"][:1].expand(rows, *condition.shape[1:])
                fixed_t = torch.as_tensor(fixed, device=noise.device, dtype=noise.dtype)
                mask_t = torch.as_tensor(mask, device=noise.device, dtype=torch.bool)
                if using_cfg:
                    initial = torch.cat((initial, initial), dim=0)
                    context = torch.cat(
                        (context, model_kwargs["z"][1:2].expand(rows, *condition.shape[1:])), dim=0
                    )
                    fixed_t = torch.cat((fixed_t, fixed_t), dim=0)
                    mask_t = torch.cat((mask_t, mask_t), dim=0)
                verified = (
                    conditional_ddim(
                        self.denoising_sampler,
                        model,
                        initial,
                        dict(model_kwargs, z=context),
                        fixed_t,
                        mask_t,
                    )[:rows]
                    .float()
                    .cpu()
                    .numpy()
                )
                return verified, native_commands(verified, statistics)

            selected, self.horizon, details = decode(raw, commands, self.config, verify)
            self.diagnostics = dict(
                horizon_policy="adaptive",
                **{"a3_" + key: value for key, value in details.items()},
            )
            steps = self.denoising_sampler.num_timesteps
            self.backend.query_diagnostics.update(
                action_candidates=count,
                action_selection_index=details["selected_path"],
                denoiser_calls=steps * (1 + details["verification_batches"]),
                denoiser_candidate_evaluations=steps * (count + details["verification_rows"]),
                verification_candidates=details["verification_rows"],
                shared_context_evaluations=1,
            )
            # Native conversion runs exactly once on this raw selected/repaired chunk.
            result = torch.as_tensor(selected, device=noise.device, dtype=noise.dtype).unsqueeze(0)
            return torch.cat((result, result), dim=0) if using_cfg else result

        return sample

    def before_query(self, prepared):
        if self.closed or self.query_open:
            raise RuntimeError("Closed or unfinished A3 query")
        self.query_open, self.horizon = True, None
        self.diagnostics = {}
        self.backend.query_diagnostics = {}

    def after_query(self, actions):
        self.query_open = False
        if self.config.enabled and self.horizon is None:
            raise RuntimeError("A3 sampler did not produce an execution prefix")

    def select_execution_horizon(self, actions, default_horizon):
        if not self.config.enabled:
            return default_horizon
        if self.closed or self.query_open or self.horizon is None:
            raise RuntimeError("A3 requires a completed current query")
        if actions.shape != (16, 7):
            raise ValueError("A3 cannot extend native predictions")
        return self.horizon

    def control_diagnostics(self):
        return dict(self.diagnostics)

    def reset(self, context):
        self.query_open, self.horizon = False, None
        self.diagnostics = {}

    def close(self):
        self.patches.close()
        if self.sampler is not None and getattr(self.sampler, "_evb_a3_official", None) is self:
            del self.sampler._evb_a3_official
        self.reset(None)
        self.closed = True

    def metadata(self):
        return dict(
            scope="A3 official-code version: global confidence and dual speculative verification",
            implementation_basis="official GR00T code, not the paper clustering/medoid version",
            reference_revision="96a85c698f3848f6e684e36a81caf6e275db2ecc",
            config=asdict(self.config),
            requires_training=False,
            execution_horizon_policy="adaptive" if self.config.enabled else "native",
            confidence_space="cumulative native execution deltas; first six axes",
            verification_space="native execution deltas; strict axis-wise thresholds",
            candidate_rule="maximum median-confidence count, then confidence sum; first tie",
            acceptance_rule="first action accepted; dual-prefix minimum plus rejection action",
            sampler="native conditional DDIM bridge; same budget for draft and both verifiers",
            sampling_grid=(
                list(self.denoising_sampler.timestep_map) if self.denoising_sampler else None
            ),
            context_encodings_per_query=1,
            excluded=[
                "paper clustering/medoid",
                "paper strict prefix",
                "paper W4A4 verifier",
                "GR00T weights or FM numerics",
                "padded actions beyond accepted_exec_len",
            ],
        )
