"""Executable CPU contract fixture, not a real VLA implementation."""

from efficient_vla_benchmark.core.backend import BackendSpec


def register():
    return BackendSpec(
        name="example",
        factory="efficient_vla_backend_example.runtime:build",
        validator="efficient_vla_backend_example:validate",
        variants=("long_chunk",),
        capabilities=frozenset({"action_scale"}),
    )


def validate(config):
    if config.model_variant != "long_chunk" or config.execution_horizon > 32:
        raise ValueError("Example supports the long_chunk variant and horizons up to 32")
    if config.precision != "auto" or config.backend_parameters:
        raise ValueError("Example accepts default precision and no backend parameters")
