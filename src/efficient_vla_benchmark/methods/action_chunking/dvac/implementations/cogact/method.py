"""Observe clean native DDIM predictions without modifying sampling."""

from collections import deque
from dataclasses import asdict

import torch
from efficient_vla_backend_cogact.sampling import EpsilonSamplerContract

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.model_ops.forward_patch import ForwardPatches

from ...selection import select_prefix


class DVAC(InferenceMethod):
    name = "dvac"
    requires_control_history = True

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.patches = ForwardPatches()
        self.closed = False
        self.contract = EpsilonSamplerContract.from_backend(backend)
        if config.tail_steps > self.contract.native_steps or config.max_horizon > 16:
            raise ValueError("DVAC exceeds native CogACT geometry")
        self.reset(None)
        if config.enabled:
            self.patches.wrap(self.contract.sampler, self._capture, "p_mean_variance")

    def _capture(self, original):
        def observe(*args, **kwargs):
            if not self.query_open or torch.is_grad_enabled():
                raise RuntimeError("DVAC requires a nonnested inference query")
            out = original(*args, **kwargs)
            self.calls += 1
            if self.calls > self.contract.native_steps - self.config.tail_steps:
                # Row zero is the guided conditional prediction. Averaging CFG
                # rows would mix the policy prediction with a different branch.
                clean = out["pred_xstart"]
                if clean.ndim != 3 or clean.shape[0] not in (1, 2):
                    raise ValueError("DVAC supports one native observation")
                self.clean.append(clean[0].detach().clone())
            return out

        return observe

    def reset(self, context):
        self.history = deque(maxlen=self.config.window_size)
        self.clean = []
        self.calls = 0
        self.query_open = False
        self.variance = None
        self.chosen = None
        self.threshold = None

    def before_query(self, prepared):
        if self.closed or self.query_open:
            raise RuntimeError("Closed or unfinished DVAC query")
        self.query_open = True
        self.clean = []
        self.calls = 0
        self.variance = None
        self.chosen = None
        self.backend.query_diagnostics = {}

    def after_query(self, actions):
        if self.config.enabled:
            if (
                self.calls != self.contract.native_steps
                or len(self.clean) != self.config.tail_steps
            ):
                raise RuntimeError("Native clean-prediction capture is incomplete")
            estimates = torch.stack(self.clean)
            self.variance = estimates.var(dim=0, correction=0).sum(dim=-1).cpu().numpy()
            if self.variance.shape != (len(actions),):
                raise RuntimeError("DVAC clean predictions and served chunk differ")
            self.backend.query_diagnostics.update(
                denoiser_calls=self.calls,
                clean_prediction_tail=len(self.clean),
                denoising_variance_total=float(self.variance.sum()),
            )
        self.clean = []
        self.query_open = False

    def select_execution_horizon(self, actions, default_horizon):
        if not self.config.enabled:
            return default_horizon
        if self.variance is None or self.chosen is not None:
            raise RuntimeError("DVAC requires one horizon decision per completed query")
        self.chosen, self.threshold = select_prefix(
            self.variance, self.history, self.config, default_horizon
        )
        self.history.append(self.variance.copy())
        return self.chosen

    def control_diagnostics(self):
        return dict(
            horizon_policy="adaptive" if self.config.enabled else "native",
            dvac_threshold=self.threshold,
            dvac_history_queries=len(self.history),
        )

    def close(self):
        self.patches.close()
        self.reset(None)
        self.closed = True

    def metadata(self):
        return dict(
            scope="DVAC tail clean-prediction variance and causal rolling threshold",
            implementation_basis="paper equations 4 and 6-9, Algorithm 1",
            config=asdict(self.config),
            requires_training=False,
            execution_horizon_policy="adaptive" if self.config.enabled else "native",
            sampler="unchanged native DDIM and CFG",
            clean_estimate="native guided pred_xstart, normalized before action clipping",
            bootstrap="native reference prefix on first query; then prior-query buffer",
            variance_convention="population variance and population window standard deviation",
            excluded=["flow-trained checkpoint claims", "sampling-step reduction"],
        )
