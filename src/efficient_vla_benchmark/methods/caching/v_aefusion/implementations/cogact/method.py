"""Overlap a private action-head worker with unchanged native VLM execution."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import torch

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.forward_patch import ForwardPatches

from .sampling import private_ddim_step


class VAEFusion(InferenceMethod):
    name = "v_aefusion"
    requires_control_history = True

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.patches = ForwardPatches()
        self.closed = False
        self.executor = None
        self.future = None
        self.previous = None
        self.query_open = False
        self.events = {}
        self.sampler = None
        if not config.enabled:
            return
        self.action_model = backend.model.action_model
        if self.action_model.net.training or backend.model.vlm.training:
            raise ValueError("V-AEFusion requires eval-mode VLM and action head")
        if self.action_model.ddim_diffusion is None:
            self.action_model.create_ddim(ddim_step=backend.parameters.num_ddim_steps)
        self.sampler = self.action_model.ddim_diffusion
        if not 0 <= config.stale_steps < self.sampler.num_timesteps:
            raise ValueError("V-AEFusion requires a fresh denoising tail")
        if self.sampler.num_timesteps != backend.parameters.num_ddim_steps:
            raise ValueError("DDIM schedule differs from the backend configuration")
        if hasattr(self.sampler, "_evb_v_aefusion"):
            raise ValueError("V-AEFusion already attached")
        self.stream = torch.cuda.Stream(device=backend.device)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="evb-action-pipeline")
        self.sampler._evb_v_aefusion = self
        self.patches.wrap(self.sampler, self._loop, "ddim_sample_loop")
        self.patches.wrap(backend.model.vlm, self._vlm)

    def _vlm(self, original):
        def forward(*args, **kwargs):
            if self.events:
                if "vlm_end" in self.events:
                    raise RuntimeError("Only one native cognition forward is supported")
                self.events["vlm_start"] = torch.cuda.Event(enable_timing=True)
                self.events["vlm_start"].record()
            output = original(*args, **kwargs)
            if self.events:
                self.events["vlm_end"] = torch.cuda.Event(enable_timing=True)
                self.events["vlm_end"].record()
            return output

        return forward

    @torch.inference_mode()
    def _early(self, previous, rng_state, start_event, begin, end):
        # Never fork/restore global RNG in a concurrent thread. Native inference
        # retains its own global draws; this worker clones that stream privately.
        with torch.cuda.device(self.backend.device), torch.cuda.stream(self.stream):
            self.stream.wait_event(start_event)
            generator = torch.Generator(device=self.backend.device)
            generator.set_state(rng_state)
            net = self.action_model.net
            shape, dtype, kwargs = previous
            initial = torch.randn((1, *shape[1:]), device=self.backend.device, generator=generator)
            initial = initial.to(dtype)
            noise = torch.cat((initial, initial), dim=0) if shape[0] == 2 else initial
            x = noise
            model = net.forward_with_cfg if shape[0] == 2 else net.forward
            begin.record(self.stream)
            for index in range(self.sampler.num_timesteps - 1, -1, -1):
                if self.sampler.num_timesteps - 1 - index >= self.config.stale_steps:
                    break
                t = torch.tensor([index] * shape[0], device=self.backend.device)
                x = private_ddim_step(self.sampler, model, x, t, kwargs, generator)
            end.record(self.stream)
            return noise, x

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
                raise RuntimeError("V-AEFusion requires a nonnested inference query")
            if (
                noise is None
                or tuple(shape) != tuple(noise.shape)
                or noise.ndim != 3
                or clip_denoised
                or denoised_fn is not None
                or cond_fn is not None
                or progress
                or eta != 0
                or not isinstance(model_kwargs, dict)
                or "z" not in model_kwargs
            ):
                raise ValueError("Unsupported nonnative DDIM options")
            cfg = model_kwargs.get("cfg_scale", 1.0) > 1.0
            rows = 2 if cfg else 1
            if noise.shape[0] != rows or model_kwargs["z"].shape[0] != rows:
                raise ValueError("Only one native observation with paired CFG is supported")
            if cfg and not torch.equal(noise[0], noise[1]):
                raise ValueError("CFG noise branches must be paired")
            fresh = dict(model_kwargs, z=model_kwargs["z"].detach().clone())
            active_steps = 0
            if self.future is None:
                result = original(
                    model,
                    shape,
                    noise=noise,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    device=device,
                    progress=False,
                    eta=0.0,
                )
            else:
                initial, result = self.future.result()
                torch.cuda.current_stream(self.backend.device).wait_stream(self.stream)
                # This rejects any native VLM change that consumes random numbers
                # before its action noise, rather than silently changing the seed.
                if not torch.equal(initial, noise):
                    raise RuntimeError("Native noise differs from the private pipeline draw")
                active_steps = self.config.stale_steps
                for _ in range(active_steps):
                    torch.randn_like(noise)  # Preserve native eta-zero RNG advancement.
                for index in range(self.sampler.num_timesteps - active_steps - 1, -1, -1):
                    t = torch.tensor([index] * shape[0], device=noise.device)
                    result = self.sampler.ddim_sample(
                        model,
                        result,
                        t,
                        clip_denoised=False,
                        model_kwargs=model_kwargs,
                        eta=0.0,
                    )["sample"]
                self.future = None
            self.previous = (tuple(shape), noise.dtype, fresh)
            self.backend.query_diagnostics.update(
                stale_denoiser_steps=active_steps,
                fresh_denoiser_steps=self.sampler.num_timesteps - active_steps,
                denoiser_calls=self.sampler.num_timesteps,
                cognition_forwards=1,
            )
            return result

        return sample

    def before_query(self, prepared):
        if self.closed or self.query_open:
            raise RuntimeError("Closed or unfinished V-AEFusion query")
        self.query_open = True
        self.backend.query_diagnostics = {}
        self.events = {}
        if self.config.enabled and self.config.stale_steps and self.previous is not None:
            origin = torch.cuda.Event(enable_timing=True)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            origin.record()
            self.events = dict(origin=origin, early_start=begin, early_end=end)
            rng_state = torch.cuda.get_rng_state(self.backend.device).clone()
            self.future = self.executor.submit(
                self._early, self.previous, rng_state, origin, begin, end
            )

    def after_query(self, actions):
        if self.future is not None:
            raise RuntimeError("Native sampler did not join the early denoising worker")
        if self.events:
            if "vlm_end" not in self.events:
                raise RuntimeError("Missing current VLM execution")
            # Native action CPU conversion already joins the fresh tail and worker.
            origin = self.events["origin"]
            times = {key: origin.elapsed_time(event) for key, event in self.events.items()}
            overlap = max(
                0.0,
                min(times["early_end"], times["vlm_end"])
                - max(times["early_start"], times["vlm_start"]),
            )
            self.backend.query_diagnostics.update(
                pipeline_stream_interval_overlap_ms=overlap,
                pipeline_early_interval_ms=times["early_end"] - times["early_start"],
                pipeline_vlm_interval_ms=times["vlm_end"] - times["vlm_start"],
            )
        self.query_open = False
        self.events = {}

    def reset(self, context):
        try:
            if self.future is not None:
                self.future.result()
            if self.executor is not None:
                self.stream.synchronize()
        finally:
            self.future = None
            self.previous = None
            self.query_open = False
            self.events = {}

    def close(self):
        try:
            self.reset(None)
        finally:
            if self.executor is not None:
                self.executor.shutdown(wait=True)
                self.executor = None
            self.patches.close()
            if self.sampler is not None and getattr(self.sampler, "_evb_v_aefusion", None) is self:
                del self.sampler._evb_v_aefusion
            self.closed = True

    def metadata(self):
        return dict(
            scope="V-AEFusion concurrent VLM and stale-conditioned early action denoising",
            implementation_basis="paper section 5.3.1, independent CogACT port",
            paper="https://arxiv.org/html/2604.24447v1",
            config=asdict(self.config),
            requires_training=False,
            conditioning="previous cognition vector, then current cognition; not pi0 KV tensors",
            scheduling="fixed stale-step budget on a separate CUDA stream/host worker",
            sampler="native eta-zero DDIM equations, grid, CFG and RNG advancement",
            excluded=["DP-Cache", "compilation", "hardware-specific kernels", "OFT L1 head"],
        )
