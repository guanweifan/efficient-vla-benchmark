"""Lightweight API-v1 plugin registration."""

from efficient_vla_benchmark.core.backend import BackendSpec


def register():
    return BackendSpec(
        name="cogact",
        factory="efficient_vla_backend_cogact.backend:build",
        validator="efficient_vla_backend_cogact.config:validate",
        variants=("libero_joint",),
        capabilities=frozenset(
            {
                "control_hooks",
                "libero_cartesian_delta_actions",
                "causal_visual_token_selection",
                "causal_indexed_kv_reuse",
                "llama_decoder",
            }
        ),
    )
