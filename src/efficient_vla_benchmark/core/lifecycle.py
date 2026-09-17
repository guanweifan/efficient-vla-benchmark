"""One per-instance method, with explicit irreversible conversion semantics."""

from efficient_vla_benchmark.core.method import InferenceMethod


class MethodHost:
    def initialize_method_state(self):
        self.method = None
        self.method_name = "none"
        self._closed = False
        self.query_diagnostics = {}

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

    def transform_action(self, action, step):
        self._ensure_open()
        return self.method.transform_action(action, step) if self.method else action

    def control_diagnostics(self):
        return self.method.control_diagnostics() if self.method else {}

    def detach_method(self):
        self._ensure_open()
        if self.method is not None:
            if not self.method.reversible:
                raise ValueError(
                    "This method transforms weights; load a fresh backend for baseline"
                )
            self.method.close()
            self.method = None
            self.method_name = "none"

    def close(self):
        if not self._closed and self.method is not None:
            self.method.close()
        self._closed = True
