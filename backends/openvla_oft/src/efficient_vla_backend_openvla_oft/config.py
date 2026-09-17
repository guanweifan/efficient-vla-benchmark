"""Restrictions of the pinned two-camera, joint-suite LIBERO checkpoint."""


def validate(config):
    if config.model_variant != "libero_joint":
        raise ValueError("OFT currently supports the joint-four-suite LIBERO checkpoint only")
    if config.precision not in {"auto", "bfloat16", "float16"}:
        raise ValueError("This OFT implementation requires bfloat16 or float16")
    if not 1 <= config.execution_horizon <= 8:
        raise ValueError("The native OFT checkpoint predicts eight actions per query")
    if config.backend_parameters:
        raise ValueError("Unknown OFT backend parameters; this preset retains native settings")
