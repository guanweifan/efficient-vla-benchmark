"""Freeze all threshold evidence on independent successful native trajectories."""

import gc
import json
import time

import numpy as np
import torch

from efficient_vla_benchmark.core.provenance import digest_file, source_metadata, stable_hash

from .population import manifest, samples
from .runtime import DynamicW4Linear
from .target import replace, targets
from .thresholds import fit_thresholds


@torch.inference_mode()
def calibrate(backend, config, population_root, output, prefix):
    if backend.method is not None or output.exists():
        raise ValueError("DyQ needs a native model and new artifact output")
    backend_id = backend.metadata()["backend"]
    population = manifest(population_root, backend_id)
    output.mkdir(parents=True)
    start = time.monotonic()
    selected = targets(backend.model, prefix)
    maxima = {}
    handles = []

    def capture(name):
        def hook(module, inputs):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            if len(x) > config.activation_tokens:
                x = x[
                    torch.linspace(0, len(x) - 1, config.activation_tokens, device=x.device).long()
                ]
            p = torch.quantile(
                x.float().abs(), config.activation_percentile / 100, dim=0
            ).clamp_min(1e-6)
            maxima[name] = p if name not in maxima else torch.maximum(maxima[name], p)

        return hook

    records = []
    try:
        for name, module in selected.items():
            handles.append(module.register_forward_pre_hook(capture(name)))
        for context, obs, identity in samples(population_root, backend_id, config):
            backend.reset(context)
            seed = context.seed + int(identity["step"]) * 1009
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            action = backend.predict_chunk(obs)
            records.append(dict(identity=identity, native=np.asarray(action).copy(), seed=seed))
        if set(maxima) != set(selected) or not records:
            raise ValueError("Incomplete DyQ native activation coverage")
    finally:
        for h in handles:
            h.remove()
    weights = {}
    for name, module in selected.items():
        w = module.weight.detach().float()
        scale = (w.abs().amax(1) / 7).clamp_min(1e-8)
        q = (w / scale[:, None]).round().clamp(-8, 7).to(torch.int8)
        weights[name] = dict(
            codes=q.cpu(),
            weight_scale=scale.cpu(),
            activation_amax=maxima[name].cpu(),
            native_dtype=str(module.weight.dtype),
            bias=None if module.bias is None else module.bias.detach().cpu(),
        )
    torch.save(weights, output / "weights.pt")
    ops = []
    for name, module in selected.items():
        device = module.weight.device
        op = DynamicW4Linear(weights[name])
        module.cpu()
        op.to(device)
        replace(backend.model, name, op)
        ops.append(op)
    maxima.clear()
    del selected, weights
    gc.collect()
    torch.cuda.empty_cache()
    evidence = [dict(sensitivity=r["identity"]["sensitivity"], errors={}) for r in records]
    for bits in (2, 4, 8):
        for op in ops:
            op.set_active_bits(bits)
        count = 0
        for index, (context, obs, identity) in enumerate(
            samples(population_root, backend_id, config)
        ):
            r = records[index]
            if identity != r["identity"]:
                raise ValueError("Independent calibration replay order changed")
            backend.reset(context)
            torch.manual_seed(r["seed"])
            torch.cuda.manual_seed_all(r["seed"])
            action = np.asarray(backend.predict_chunk(obs))
            # Execute the same H8 prefix as the benchmark; do not score unused H16 tail.
            evidence[index]["errors"][str(bits)] = float(
                np.linalg.norm(action[:8, :6] - r["native"][:8, :6], axis=1).mean()
            )
            count += 1
        if count != len(records):
            raise ValueError("Incomplete DyQ precision replay")
        print(json.dumps(dict(precision=bits, frames=count)), flush=True)
    thresholds, bins = fit_thresholds(evidence, config)
    identities = [r["identity"] for r in records]
    m = dict(
        schema_version=2,
        algorithm="dyq_vla",
        complete=True,
        backend=backend_id,
        checkpoint_sha256=backend.metadata()["checkpoint_sha256"],
        config=config.calibration_parameters(),
        population_kind=population["kind"],
        population_sha256=population["sha256"],
        sample_identities=identities,
        sample_identities_sha256=stable_hash(identities),
        targets=[
            name
            for name, module in backend.model.named_modules()
            if isinstance(module, DynamicW4Linear)
        ],
        error_records=evidence,
        threshold_bins=bins,
        thresholds=thresholds,
        threshold_estimator="equal_count_bins_first_mean_error_bound_crossing",
        error_definition=(
            "mean L2 xyz+rotation over native H8 executed prefix; same stochastic seed"
        ),
        files={"weights.pt": digest_file(output / "weights.pt")},
        source=source_metadata(),
        wall_seconds=time.monotonic() - start,
    )
    (output / "manifest.json").write_text(json.dumps(m, indent=2))
    return m
