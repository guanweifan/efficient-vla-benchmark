"""Independent-module native training calibration, without policy training."""

import json
import time

import torch

from efficient_vla_benchmark.calibration.libero import training_observations
from efficient_vla_benchmark.core.provenance import digest_file, source_metadata
from efficient_vla_benchmark.model_ops.quantization.gptq import quantize

from .target import flatten_input, targets


@torch.inference_mode()
def calibrate(backend, config, dataset_root, output, vision_prefix, language_prefix):
    if backend.method is not None or output.exists():
        raise ValueError("EaqVLA needs a fresh native model and new output")
    output.mkdir(parents=True)
    start = time.monotonic()
    selected = targets(backend.model, vision_prefix, language_prefix)
    captured = {name: [] for name in selected}
    handles = []
    identities = []

    def capture(name):
        def hook(module, args):
            x = flatten_input(module, args[0].detach())
            if len(x) > config.calibration_tokens:
                x = x[
                    torch.linspace(0, len(x) - 1, config.calibration_tokens, device=x.device).long()
                ]
            captured[name].append(x.cpu())

        return hook

    try:
        for name, (module, _) in selected.items():
            handles.append(module.register_forward_pre_hook(capture(name)))
        for context, observation, identity in training_observations(
            dataset_root, config.calibration_samples, config.calibration_seed
        ):
            backend.reset(context)
            backend.predict_chunk(observation)
            identities.append(identity)
    finally:
        for h in handles:
            h.remove()
    if len(identities) != config.calibration_samples or any(
        len(v) != len(identities) for v in captured.values()
    ):
        raise ValueError("Incomplete modular calibration capture")
    entries = {}
    for index, (name, (module, side)) in enumerate(selected.items()):
        width = module.weight.flatten(1).shape[1]
        h = torch.zeros((width, width), device=module.weight.device, dtype=torch.float32)
        count = 0
        for x in captured.pop(name):
            x = x.to(device=module.weight.device, dtype=torch.float32)
            h.addmm_(x.T, x)
            count += len(x)
        h *= 2 / count
        bits = config.vision_bits if side == "vision" else config.language_bits
        codes, scales = quantize(
            module.weight.flatten(1), h, bits, config.gptq_block_size, config.damp_percent
        )
        entries[name] = dict(
            codes=codes.cpu(),
            scales=scales.cpu(),
            bits=bits,
            side=side,
            native_dtype=str(module.weight.dtype),
            bias=None if module.bias is None else module.bias.detach().cpu(),
        )
        del h, x, codes, scales
        print(json.dumps(dict(calibrated=index + 1, total=len(selected), module=name)), flush=True)
    torch.save(entries, output / "weights.pt")
    m = dict(
        schema_version=1,
        algorithm="eaqvla",
        complete=True,
        config=config.calibration_parameters(),
        backend=backend.metadata()["backend"],
        checkpoint_sha256=backend.metadata()["checkpoint_sha256"],
        targets=list(selected),
        calibration_identities=identities,
        variant="modular_weight_GPTQ_native_activation",
        files={"weights.pt": digest_file(output / "weights.pt")},
        source=source_metadata(),
        wall_seconds=time.monotonic() - start,
    )
    (output / "manifest.json").write_text(json.dumps(m, indent=2))
    return m
