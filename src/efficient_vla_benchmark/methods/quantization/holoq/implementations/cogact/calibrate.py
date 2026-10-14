"""Training-only calibration; exact block-local GPTQ scales, no policy updates."""

import json
import time
from collections import defaultdict

import torch

from efficient_vla_benchmark.calibration.libero import training_observations
from efficient_vla_benchmark.core.provenance import digest_file, source_metadata

from ...gptq import quantize, rtn
from ...rotation import build, rotate, transformed_weight
from .step import DDIMStepTracker
from .target import targets


@torch.inference_mode()
def calibrate(backend, config, dataset_root, output):
    if backend.method is not None or backend.metadata()["precision"] != "bfloat16":
        raise ValueError("HoloQ needs a fresh native BF16 CogACT")
    if output.exists():
        raise ValueError("Use a new calibration output")
    output.mkdir(parents=True)
    start = time.monotonic()
    selected = targets(backend.model)
    tracker = DDIMStepTracker(backend.model.action_model, backend.parameters.num_ddim_steps)
    values = {name: defaultdict(list) for name in selected}
    handles, identities = [], []

    def capture(name, side):
        def hook(module, inputs):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            if len(x) > config.calibration_tokens:
                indices = torch.linspace(
                    0, len(x) - 1, config.calibration_tokens, device=x.device
                ).long()
                x = x[indices]
            row = 0 if side == "language" else tracker.current_step
            if row is None:
                raise ValueError("Missing DiT timestep during calibration")
            values[name][row].append(x.cpu())

        return hook

    try:
        for name, (module, side) in selected.items():
            handles.append(module.register_forward_pre_hook(capture(name, side)))
        for context, observation, identity in training_observations(
            dataset_root, config.calibration_samples, config.calibration_seed
        ):
            tracker.reset()
            backend.reset(context)
            backend.predict_chunk(observation)
            identities.append(identity)
        if len(identities) != config.calibration_samples:
            raise ValueError("Incomplete training calibration")
    finally:
        for handle in handles:
            handle.remove()
        tracker.close()
    weights = {}
    for index, (name, (module, side)) in enumerate(selected.items()):
        transform = build(module.weight, config.block_size, side != "language")
        expected_rows = 1 if side == "language" else backend.parameters.num_ddim_steps
        if set(values[name]) != set(range(expected_rows)):
            raise ValueError("Incomplete activation-scale table")
        scales, language_input = [], None
        for row in range(expected_rows):
            x = torch.cat(values[name].pop(row)).to(module.weight.device)
            x = rotate(x[:, transform["permutation"].to(x.device)], transform["input"].to(x))
            scale = (
                torch.quantile(x.float().abs(), config.activation_percentile / 100, dim=0) / 7
            ).clamp_min(1e-6)
            scales.append(scale.cpu())
            if side == "language":
                language_input = x.float()
        w = transformed_weight(module.weight, transform)
        if side == "language":
            codes, ws = quantize(
                w, language_input, config.gptq_block_size, config.gptq_damp_percent
            )
            group_size = config.gptq_block_size
        else:
            codes, ws = rtn(w)
            group_size = module.in_features
        weights[name] = dict(
            codes=codes,
            weight_scales=ws,
            weight_group_size=group_size,
            activation_steps=torch.stack(scales),
            transform=transform,
            side=side,
            native_dtype=str(module.weight.dtype),
            bias=None if module.bias is None else module.bias.detach().cpu(),
        )
        del w, language_input, x
        print(json.dumps(dict(calibrated=index + 1, total=len(selected), module=name)), flush=True)
    torch.save(weights, output / "weights.pt")
    manifest = dict(
        schema_version=1,
        algorithm="holoq",
        complete=True,
        variant="official_code_composite_gptq_perstep_cogact",
        checkpoint_sha256=backend.metadata()["checkpoint_sha256"],
        config=config.calibration_parameters(),
        schedule=tracker.schedule,
        targets=list(selected),
        calibration_identities=identities,
        calibration_scope="balanced_RLDS_training_observations_uniform_token_subsample",
        official_commit="3727e2203568db43fc5fba06ee8686c1b47c044f",
        files={"weights.pt": digest_file(output / "weights.pt")},
        source=source_metadata(),
        wall_seconds=time.monotonic() - start,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
