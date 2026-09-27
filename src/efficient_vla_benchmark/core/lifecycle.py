"""One per-instance method, with explicit irreversible conversion semantics."""

import time

from efficient_vla_benchmark.core.method import InferenceMethod


class MethodHost:
    def initialize_method_state(self):
        self.method = None
        self.method_name = "none"
        self._closed = False
        self.query_diagnostics = {}
        self.auxiliary_runner = None

    def drain_auxiliary(self):
        """Join pending work and expose its full cost before reset/reuse/profiling."""
        if self.auxiliary_runner is not None:
            metrics = self.auxiliary_runner.drain()
            if metrics:
                self.query_diagnostics.update(metrics)
                if self.method is not None:
                    self.query_diagnostics.update(self.method.auxiliary_diagnostics(dict(metrics)))

    def run_native_query(self, prepared, native):
        """Backend-owned inference; methods only select modes through callbacks."""
        self.drain_auxiliary()
        method = self.method
        if method is not None:
            method.before_query(prepared)
        started = time.perf_counter_ns()
        actions = native(prepared)
        primary_ms = (time.perf_counter_ns() - started) / 1e6
        if method is not None:
            method.after_query(actions)
        if method is not None and method.wants_auxiliary_query():
            if self.auxiliary_runner is None:
                self.auxiliary_runner = self.make_auxiliary_runner()
            self.query_diagnostics.update(primary_forward_ms=primary_ms, auxiliary_queries=1)

            def work():
                try:
                    method.begin_auxiliary_query()
                    native(prepared)  # Output is deliberately NOT sent to the controller.
                finally:
                    method.end_auxiliary_query()

            self.auxiliary_runner.submit(work)
        return actions

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("The backend is closed; construct a new instance")

    def attach_method(self, method: InferenceMethod):
        self._ensure_open()
        if self.method is not None:
            raise ValueError("Exactly one acceleration method is permitted")
        self.method = method
        self.method_name = method.name

    def observe(self, observation):
        self._ensure_open()
        if self.auxiliary_runner is not None and self.auxiliary_runner.done():
            self.drain_auxiliary()  # Surface failures even while queued actions execute.
        if self.method is not None:
            self.method.observe(observation)

    def after_action(self, action, remaining):
        self._ensure_open()
        return self.method.after_action(action, remaining) if self.method else False

    def before_plan(self, observation, step):
        self._ensure_open()
        return self.method.before_plan(observation, step) if self.method else None

    def on_plan(self, actions, step):
        self._ensure_open()
        if self.method:
            self.method.on_plan(actions, step)

    def select_execution_horizon(self, actions, default_horizon):
        self._ensure_open()
        return (
            self.method.select_execution_horizon(actions, default_horizon)
            if self.method
            else default_horizon
        )

    def transform_action(self, action, step):
        self._ensure_open()
        return self.method.transform_action(action, step) if self.method else action

    def control_diagnostics(self):
        return self.method.control_diagnostics() if self.method else {}

    def detach_method(self):
        self._ensure_open()
        self.drain_auxiliary()
        if self.auxiliary_runner is not None:
            self.auxiliary_runner.close()
            self.auxiliary_runner = None
        if self.method is not None:
            if not self.method.reversible:
                raise ValueError(
                    "This method transforms weights; load a fresh backend for baseline"
                )
            self.method.close()
            self.method = None
            self.method_name = "none"

    def close(self):
        try:
            self.drain_auxiliary()
        finally:
            if self.auxiliary_runner is not None:
                self.auxiliary_runner.close()
                self.auxiliary_runner = None
            if not self._closed and self.method is not None:
                self.method.close()
            self._closed = True
