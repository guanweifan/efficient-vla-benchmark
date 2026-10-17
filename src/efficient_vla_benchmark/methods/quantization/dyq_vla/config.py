"""DyQ temporal mechanism settings; frozen calibration owns interval boundaries."""

import math
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class DyQConfig:
    runtime: str = "native_float"
    frames_per_trajectory: int = 8
    activation_percentile: float = 99.9
    activation_tokens: int = 256
    motion_window: int = 10
    jerk_window: int = 5
    sensitivity_lambda: float = 0.5
    theta_fp: float = 0.5
    hysteresis_window: int = 3
    accuracy_budget: float = 0.02
    sensitivity_bins: int = 10
    eta: float = 1e-6

    @classmethod
    def parse(cls, parameters):
        if set(parameters) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown DyQ parameter")
        c = cls(**parameters)
        if c.runtime not in ("native_float", "mixed_integer"):
            raise ValueError("Unsupported DyQ execution runtime")
        for name in (
            "frames_per_trajectory",
            "activation_tokens",
            "motion_window",
            "jerk_window",
            "hysteresis_window",
            "sensitivity_bins",
        ):
            value = getattr(c, name)
            if type(value) is not int or value < 1:
                raise ValueError("DyQ integer settings must be positive")
        if c.frames_per_trajectory > 32 or c.activation_tokens > 1024 or c.sensitivity_bins > 64:
            raise ValueError("Unbounded DyQ calibration")
        if c.jerk_window >= c.motion_window:
            raise ValueError("DyQ micro window must be shorter than macro window")
        for name in (
            "activation_percentile",
            "sensitivity_lambda",
            "theta_fp",
            "accuracy_budget",
            "eta",
        ):
            value = getattr(c, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("DyQ scalar must be finite")
        if not 0 < c.activation_percentile <= 100 or not 0 <= c.sensitivity_lambda <= 1:
            raise ValueError("Invalid DyQ percentile/fusion")
        if min(c.theta_fp, c.accuracy_budget, c.eta) <= 0:
            raise ValueError("DyQ bounds must be positive")
        return c

    def calibration_parameters(self):
        parameters = asdict(self)
        # Preserve the exact historical schema-2 identity; a new runtime is
        # explicitly schema-3 and may not reuse its activation thresholds.
        if self.runtime == "native_float":
            parameters.pop("runtime")
        return parameters
