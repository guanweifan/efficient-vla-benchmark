"""Falcon on an unchanged native DDIM grid, with one batched historical probe."""

from dataclasses import asdict

import torch

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.forward_patch import ForwardPatches

from .buffer import LatentBuffer
from .selection import clean_estimates, reference_distances, selection_probabilities


class Falcon(InferenceMethod):
    name = "falcon"

    def __init__(self, backend, contract, config):
        self.backend, self.contract, self.config = backend, contract, config
        self.requires_control_history = config.enabled
        self.times = tuple(int(t) for t in contract.sampler.timestep_map)
        if (
            len(self.times) != contract.native_steps
            or tuple(sorted(set(self.times))) != self.times
            or self.times[-1] != contract.start_index
            or not any(config.min_noise_level < t < self.times[-1] for t in self.times)
        ):
            raise ValueError("Falcon needs a native intermediate starting noise level")
        self.buffer = LatentBuffer(config.buffer_size)
        self.patches = ForwardPatches()
        self.closed = False
        self.reset(None)
        if config.enabled:
            if "ddim_sample_loop" in contract.sampler.__dict__:
                raise ValueError("Another sampler replacement is attached")
            self.patches.wrap(contract.sampler, self._loop, "ddim_sample_loop")

    def reset(self, context):
        self.buffer.clear()
        self.previous = None
        self.query_index = self.emitted = 0
        self.query_open = self.sampled = False
        self.generator = None
        self.seed = (self.config.selection_seed + getattr(context, "seed", 0)) % 2**32

    def before_query(self, prepared):
        if self.closed or self.query_open:
            raise RuntimeError("Closed or unfinished Falcon query; reset before reuse")
        if (
            self.config.enabled
            and self.previous is not None
            and self.emitted != self.config.reference_horizon
        ):
            raise ValueError("Falcon reference requires the configured number of executed actions")
        self.query_open, self.sampled = True, False
        self.backend.query_diagnostics = {}

    def after_query(self, actions):
        self.query_open = False
        self.emitted = 0

    def after_action(self, action, remaining):
        self.emitted += 1
        return False

    def _probe(self, model, model_kwargs, noise):
        entries = self.buffer.entries
        candidates = torch.stack([item.latent for item in entries])
        count = len(entries)
        times = torch.tensor([item.noise_level for item in entries], device=noise.device)
        conditioning = model_kwargs["z"]
        rows = noise.shape[0]
        if rows == 2:
            values = torch.cat((candidates, candidates))
            probe_times = times.repeat(2)
            z = torch.cat(
                (
                    conditioning[:1].expand(count, *conditioning.shape[1:]),
                    conditioning[1:].expand(count, *conditioning.shape[1:]),
                )
            )
        else:
            values, probe_times = candidates, times
            z = conditioning.expand(count, *conditioning.shape[1:])
        predicted = model(values, probe_times, **dict(model_kwargs, z=z))
        if predicted.shape != values.shape or not bool(torch.isfinite(predicted).all()):
            raise ValueError("Invalid native epsilon output for Falcon candidates")
        alphas = torch.as_tensor(
            [self.contract.alphas_cumprod[item.noise_level] for item in entries],
            device=noise.device,
            dtype=torch.float32,
        )
        estimates = clean_estimates(candidates, predicted[:count], alphas)
        distances = reference_distances(estimates, self.previous, self.config.reference_horizon)
        indices, probabilities = selection_probabilities(
            distances,
            times,
            self.config.distance_threshold,
            self.config.min_noise_level,
            self.config.temperature,
        )
        explore = bool(
            torch.rand((), device=noise.device, generator=self.generator)
            < self.config.exploration_rate
        )
        selected = None
        if not explore and len(indices):
            offset = torch.multinomial(probabilities, 1, generator=self.generator)
            selected = entries[int(indices[offset].item())]
        return selected, count, len(indices), explore

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
            if self.closed or not self.query_open or self.sampled or torch.is_grad_enabled():
                raise RuntimeError("Falcon requires one inference sampler call per query")
            if (
                noise is None
                or tuple(shape) != tuple(noise.shape)
                or noise.ndim != 3
                or noise.dtype != torch.float32
                or clip_denoised
                or eta != 0
                or denoised_fn is not None
                or cond_fn is not None
                or progress
                or not isinstance(model_kwargs, dict)
                or "z" not in model_kwargs
                or set(model_kwargs) - {"z", "cfg_scale"}
            ):
                raise ValueError("Unsupported native Falcon sampling options")
            rows = 2 if model_kwargs.get("cfg_scale", 1.0) > 1 else 1
            if noise.shape[0] != rows or model_kwargs["z"].shape[0] != rows:
                raise ValueError("Falcon supports one native observation with paired CFG")
            if rows == 2 and not torch.equal(noise[0], noise[1]):
                raise ValueError("Native CFG initial noises must be paired")
            if noise.shape[1] <= self.config.reference_horizon:
                raise ValueError("Falcon requires overlapping predicted action chunks")
            if self.generator is None:
                self.generator = torch.Generator(device=noise.device).manual_seed(self.seed)
            self.sampled = True
            selected, candidates, eligible, explore = None, 0, 0, False
            if self.previous is not None:
                if not self.buffer.entries or self.previous.shape != noise.shape[1:]:
                    raise ValueError("Incomplete or changed Falcon history; reset required")
                selected, candidates, eligible, explore = self._probe(model, model_kwargs, noise)
            start = self.contract.native_steps - 1 if selected is None else selected.step
            value = (
                noise if selected is None else selected.latent[None].expand(rows, -1, -1).clone()
            )
            # Native eta-zero DDIM draws unused noise at every native step.
            # Keep those skipped draws so subsequent query initial noise stays aligned.
            for _ in range(self.contract.native_steps - 1 - start):
                torch.randn_like(noise)
            for step in range(start, -1, -1):
                self.buffer.add(value[0], self.query_index, step, self.times[step])
                t = torch.full((rows,), step, device=noise.device, dtype=torch.long)
                value = self.contract.sampler.ddim_sample(
                    model,
                    value,
                    t,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    eta=0.0,
                )["sample"]
            if not bool(torch.isfinite(value).all()):
                raise ValueError("Nonfinite Falcon sampled actions")
            self.previous = value[0].detach().clone()
            self.backend.query_diagnostics.update(
                denoiser_calls=start + 1 + bool(candidates),
                denoiser_candidate_evaluations=start + 1 + candidates,
                denoiser_evaluated_rows=rows * (start + 1 + candidates),
                falcon_probe_calls=int(bool(candidates)),
                falcon_probe_candidates=candidates,
                falcon_eligible_candidates=eligible,
                falcon_reused=selected is not None,
                falcon_exploration=explore,
                falcon_start_timestep=self.times[start],
                falcon_selected_age=0 if selected is None else self.query_index - selected.query,
                falcon_buffer_entries=len(self.buffer.entries),
                falcon_reference_tail_length=noise.shape[1] - self.config.reference_horizon,
                native_reference_steps=self.contract.native_steps,
            )
            self.query_index += 1
            return value

        return sample

    def close(self):
        self.patches.close()
        self.reset(None)
        self.closed = True

    def metadata(self):
        return dict(
            scope="Falcon DDIM historical-latent selection and partial-denoising core",
            paper="https://proceedings.mlr.press/v267/chen25ao.html",
            reference_commit="9a72d315dd072d46f3498bf6909784914c216e39",
            implementation_basis="paper Algorithm 2; disclosed executable selection conventions",
            config=asdict(self.config),
            requires_training=False,
            reference="immediately preceding normalized unexecuted action tail",
            posterior="current-condition epsilon probe with original cumulative alpha",
            selection="L2 divided by scalar count; threshold-qualified noise-level softmax",
            buffer="pre-update latents at actual noise indices; oldest/highest-noise eviction",
            rng="private selection generator; original native discarded draws retained",
            native_grid=list(self.times),
            cfg="all conditional candidates followed by paired unconditional candidates",
            differences_from_release=[
                "store pre-update latent with matching noise level",
                "update reference after every sampled chunk",
                "paper oldest/highest-noise priority rather than absolute floor distance",
                "all threshold-qualified candidates; no top20 truncation for buffers above20",
            ],
            exclusions=[
                "published task scores",
                "DDPM and DPM-Solver variants",
                "new policy or training",
                "OFT deterministic action head",
                "action-horizon changes",
            ],
        )
