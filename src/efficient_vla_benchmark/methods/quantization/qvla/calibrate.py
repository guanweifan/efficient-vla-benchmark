"""Read-only training collection and official-cost global allocation."""

import json
import time

import torch

from efficient_vla_benchmark.calibration.libero import training_observations
from efficient_vla_benchmark.core.provenance import digest_file, source_metadata

from .allocation import greedy_allocate
from .sensitivity import InputHessian, channel_scores, inverse_cholesky_diagonal


@torch.inference_mode()
def calibrate(backend, config, dataset, output):
    from .implementations.openvla_oft.target import targets

    if (
        backend.method is not None
        or backend.metadata()["backend"] != "openvla_oft"
        or backend.metadata()["precision"] != "bfloat16"
    ):
        raise ValueError("QVLA calibration requires fresh native BF16 OFT")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite calibration evidence")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    modules = targets(backend.model)
    statistics = {}

    def hook(name, module):
        def capture(_, inputs):
            if name not in statistics:
                statistics[name] = InputHessian(module)
            statistics[name].add_batch(inputs[0])

        return capture

    handles = [
        module.register_forward_pre_hook(hook(name, module)) for name, module in modules.items()
    ]
    identities = []
    try:
        for context, obs, identity in training_observations(
            dataset, config.calibration_samples, config.calibration_seed
        ):
            backend.reset(context)
            backend.predict_chunk(obs)
            identities.append(identity)
            print(
                f"QVLA: native covariance {len(identities)}/{config.calibration_samples}",
                flush=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    if len(identities) != config.calibration_samples:
        raise ValueError("Incomplete QVLA observations")
    proxies = {}
    for index, (name, module) in enumerate(modules.items()):
        if name not in statistics:
            continue
        diagonal = inverse_cholesky_diagonal(
            statistics.pop(name).matrix.to(module.weight.device), config.percdamp
        )
        proxies[name] = channel_scores(module.weight, diagonal, config.candidate_bits)
        if index % 16 == 0:
            print(f"QVLA: sensitivity {index + 1}/{len(modules)}", flush=True)
    if not proxies:
        raise ValueError("No active QVLA calibration targets")
    # Preserve model insertion order: it is the official allocator's tie order.
    torch.save(proxies, output / "sensitivity.pt")
    assignments, stats = greedy_allocate(proxies, config.candidate_bits, config.target_avg_bits)
    (output / "gates.json").write_text(json.dumps(assignments) + "\n")
    manifest = {
        "schema_version": 5,
        "algorithm": "qvla",
        "complete": True,
        "backend": "openvla_oft",
        "precision": "bfloat16",
        "config": config.calibration_parameters(),
        "checkpoint_sha256": backend.metadata()["checkpoint_sha256"],
        "reference_revision": "26cc4821a3be4c003d09d3c7997b38db2a347982",
        "sensitivity": "official_literal_recurrence_contiguous_diag_division",
        "allocation": "official_absolute_cost_per_saved_bit",
        "target_modules": list(proxies),
        "inactive_target_modules": [n for n in modules if n not in proxies],
        "files": {n: digest_file(output / n) for n in ("sensitivity.pt", "gates.json")},
        "stats": stats,
        "calibration_identities": identities,
        "source": source_metadata(),
        "calibration_seconds": time.monotonic() - started,
        "adaptation": "native_joint_checkpoint_inputs_and_training_only_balanced_observations",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
