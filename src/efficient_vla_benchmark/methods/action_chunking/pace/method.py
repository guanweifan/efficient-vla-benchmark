"""Shared PACE controller; backend bindings declare native action geometry."""

from dataclasses import asdict
from importlib.metadata import version

from efficient_vla_benchmark.core.method import InferenceMethod
from efficient_vla_benchmark.core.provenance import stable_hash

from .artifact import read_manifest
from .selection import instruction_key, select_prefix


class PACE(InferenceMethod):
    name = "pace"
    requires_control_history = True

    def __init__(self, config, artifact_path, prediction_length):
        self.config, self.prediction_length = config, prediction_length
        self.manifest, self.thresholds = read_manifest(artifact_path, config)
        if config.smoothing_window > prediction_length or (
            config.max_horizon is not None and config.max_horizon > prediction_length
        ):
            raise ValueError("PACE exceeds native prediction length")
        self.closed = False
        self.reset(None)

    def reset(self, context):
        self.instruction = None
        self.threshold = None
        self.accepted = 0

    def observe(self, observation):
        if self.closed:
            raise RuntimeError("PACE is closed")
        key = instruction_key(observation.instruction)
        if key not in self.thresholds:
            raise ValueError("PACE instruction has no training-only calibrated threshold")
        self.instruction, self.threshold = key, self.thresholds[key]

    def select_execution_horizon(self, actions, default_horizon):
        if self.closed or self.instruction is None:
            raise RuntimeError("PACE requires a current instruction")
        if len(actions) != self.prediction_length:
            raise ValueError("PACE native prediction length changed")
        if not self.config.enabled:
            return default_horizon
        horizon, self.accepted = select_prefix(actions, self.threshold, self.config)
        return horizon

    def control_diagnostics(self):
        return dict(
            horizon_policy="adaptive" if self.config.enabled else "native",
            pace_threshold=self.threshold,
            pace_accepted_valleys=self.accepted,
        )

    def close(self):
        self.reset(None)
        self.closed = True

    def metadata(self):
        return dict(
            scope="PACE smoothed Cartesian speed valleys and training-calibrated prominence",
            implementation_basis="paper Section 3 and Cartesian-smoothed ablation",
            config=asdict(self.config),
            requires_training=False,
            execution_horizon_policy="adaptive" if self.config.enabled else "native",
            artifact_sha256=self.manifest["statistics_sha256"],
            calibration_manifest_sha256=stable_hash(self.manifest),
            calibration_scope="task-level training demonstrations; never evaluation outcomes",
            scipy_version=version("scipy"),
            conventions=self.config.calibration_parameters(),
            action_mapping="XYZ displacement norms; boundary k executes through command k",
            excluded=["joint-space forward kinematics", "multi-arm experiments"],
        )
