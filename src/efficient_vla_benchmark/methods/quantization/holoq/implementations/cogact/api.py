"""Lazy binding; common harness remains algorithm-independent."""


def preflight(config, resources):
    from ...artifact import read_manifest

    read_manifest(resources.artifact_path, config)
    if config.dit_graphs:
        from efficient_vla_benchmark.model_ops.quantization.tinychat_runtime import read_runtime

        read_runtime(resources.runtime_path)


def build(backend, config, resources):
    from .method import HoloQ

    return HoloQ(backend, config, resources.artifact_path, resources.runtime_path)


def calibrate(backend, config, resources):
    from .calibrate import calibrate as run

    return run(backend, config, resources.dataset_path, resources.artifact_dir)
