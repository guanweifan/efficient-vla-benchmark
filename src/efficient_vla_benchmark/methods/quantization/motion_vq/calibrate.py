import json
import time

import torch

from efficient_vla_benchmark.calibration.libero import training_observations
from efficient_vla_benchmark.core.provenance import digest_file, source_metadata

from .codebook import quantize_weight
from .target import targets


@torch.inference_mode()
def calibrate(backend, config, dataset, output, prefix):
    if backend.method is not None or output.exists():
        raise ValueError("MotionVQ needs a native model and fresh artifact")
    output.mkdir(parents=True)
    start = time.monotonic()
    selected = targets(backend.model, prefix)
    sums = {}
    counts = {}
    handles = []
    identities = []

    def capture(name):
        def hook(module, args):
            x = args[0].detach().reshape(-1, args[0].shape[-1])
            if len(x) > config.calibration_tokens:
                x = x[
                    torch.linspace(0, len(x) - 1, config.calibration_tokens, device=x.device).long()
                ]
            value = x.float().square().sum(0)
            sums[name] = value if name not in sums else sums[name] + value
            counts[name] = counts.get(name, 0) + len(x)

        return hook

    try:
        for name, module in selected.items():
            handles.append(module.register_forward_pre_hook(capture(name)))
        for context, obs, identity in training_observations(
            dataset, config.calibration_samples, config.calibration_seed
        ):
            backend.reset(context)
            backend.predict_chunk(obs)
            identities.append(identity)
    finally:
        for handle in handles:
            handle.remove()
    if set(sums) != set(selected) or len(identities) != config.calibration_samples:
        raise ValueError("Incomplete MotionVQ calibration")
    entries = {}
    with torch.random.fork_rng(devices=[backend.device]):
        torch.manual_seed(config.kmeans_seed)
        torch.cuda.manual_seed_all(config.kmeans_seed)
        for i, (name, module) in enumerate(selected.items()):
            hdiag = sums.pop(name) / counts[name]
            e = dict(bias=None if module.bias is None else module.bias.detach().cpu())
            e["high"] = quantize_weight(module.weight, hdiag, 256, 256, config)
            e["low"] = quantize_weight(module.weight, hdiag, 128, 64, config)
            entries[name] = e
            print(json.dumps(dict(calibrated=i + 1, total=len(selected), module=name)), flush=True)
    torch.save(entries, output / "weights.pt")
    m = dict(
        schema_version=1,
        algorithm="motion_vq",
        complete=True,
        variant="official_code_LLM_only_MotionVQ",
        backend=backend.metadata()["backend"],
        checkpoint_sha256=backend.metadata()["checkpoint_sha256"],
        config=config.calibration_parameters(),
        targets=list(selected),
        calibration_identities=identities,
        official_commit="bc8b69c4fd4fd24f1a3b4694d58f78caf6c6f79a",
        files={"weights.pt": digest_file(output / "weights.pt")},
        source=source_metadata(),
        wall_seconds=time.monotonic() - start,
    )
    (output / "manifest.json").write_text(json.dumps(m, indent=2))
    return m
