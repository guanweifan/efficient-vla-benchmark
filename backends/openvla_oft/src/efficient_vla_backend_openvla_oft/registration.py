"""Lightweight entry point. Factories are import strings, not eager imports."""

from efficient_vla_benchmark.core.backend import BackendSpec


def register():
    return BackendSpec(
        name="openvla_oft",
        factory="efficient_vla_backend_openvla_oft.backend:build",
        validator="efficient_vla_backend_openvla_oft.config:validate",
        variants=("libero_joint",),
        capabilities=frozenset(
            {
                "llama_decoder",
                "auxiliary_dense_inference",
                "visual_token_layout",
                "l1_action_head",
                "control_hooks",
                "libero_cartesian_delta_actions",
            }
        ),
    )
