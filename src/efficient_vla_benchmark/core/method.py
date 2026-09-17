"""Lifecycle of one complete paper mechanism, including its own controller."""

from efficient_vla_benchmark.core.types import EpisodeContext, Observation


class InferenceMethod:
    name = "none"
    requires_control_history = False
    reversible = True

    def reset(self, context: EpisodeContext) -> None:
        pass

    def observe(self, observation: Observation) -> None:
        """Receive every control frame, including frames that do not query the VLM."""
        pass

    def before_query(self, prepared: dict) -> None:
        pass

    def after_query(self, actions) -> None:
        pass

    def after_action(self, action, remaining: int) -> bool:
        """Return True to discard the remaining chunk and query on the next step."""
        return False

    def close(self) -> None:
        pass

    def metadata(self) -> dict:
        return {}
