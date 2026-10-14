"""Training-only gradient probes, without optimizer or policy updates."""

import json
import time

import numpy as np
import torch

from efficient_vla_benchmark.calibration.libero import training_action_chunks
from efficient_vla_benchmark.core.provenance import digest_file, source_metadata

from ...drift import drift_weights, normalize_actions
from .target import targets


def epsilon_sensitivity(dit, cognition, chunks, layers, axis, indices, seed):
    state = [m.weight.requires_grad for _, m in layers]
    scores = {name: 0.0 for name, _ in layers}
    device = next(dit.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    timesteps = []
    try:
        for _, m in layers:
            m.weight.requires_grad_(True)
        with torch.enable_grad():
            for index in indices:
                z = (
                    cognition[index]
                    .to(device=device, dtype=torch.float32)
                    .clone()
                    .reshape(1, 1, -1)
                )
                x = torch.as_tensor(chunks[index], device=device, dtype=torch.float32).unsqueeze(0)
                noise = torch.randn(x.shape, device=device, generator=generator)
                t = torch.randint(
                    0, dit.diffusion.num_timesteps, (1,), device=device, generator=generator
                )
                pred = dit.net(dit.diffusion.q_sample(x, t, noise), t, z)
                loss = ((pred - noise).square() * axis.to(device).reshape(1, 1, 7)).mean()
                gradients = torch.autograd.grad(
                    loss, [m.weight for _, m in layers], allow_unused=False
                )
                for (name, _), gradient in zip(layers, gradients):
                    scores[name] += float(gradient.detach().abs().mean()) / len(indices)
                timesteps.append(int(t))
    finally:
        for (_, m), value in zip(layers, state):
            m.weight.requires_grad_(value)
    if not all(np.isfinite(v) and v > 0 for v in scores.values()):
        raise ValueError("Missing/nonfinite drift gradients")
    return scores, timesteps


def calibrate(backend, config, dataset_root, output):
    if backend.method is not None or output.exists():
        raise ValueError("DA-MPA needs a fresh native backend and new output")
    output.mkdir(parents=True)
    start = time.monotonic()
    selected = targets(backend.model)
    values = {name: [] for name in selected}
    cognition = []
    handles = []
    identities = []
    chunks = []
    raw_actions = []

    def capture(name):
        def hook(module, args):
            x = args[0].detach().reshape(-1, args[0].shape[-1])
            values[name].append(x.float().abs().amax(0).cpu())

        return hook

    feature = []

    def capture_cognition(module, args):
        if not feature:
            feature.append(args[0].detach().reshape(-1, args[0].shape[-1])[:1].cpu())

    try:
        for name, (module, _, _) in selected.items():
            handles.append(module.register_forward_pre_hook(capture(name)))
        handles.append(
            backend.model.action_model.net.z_embedder.linear.register_forward_pre_hook(
                capture_cognition
            )
        )
        horizon = backend.model.action_model.future_action_window_size + 1
        for context, obs, identity, raw in training_action_chunks(
            dataset_root, config.calibration_samples, config.calibration_seed, horizon
        ):
            feature.clear()
            backend.reset(context)
            backend.predict_chunk(obs)
            if len(feature) != 1:
                raise ValueError("Native conditioning capture failed")
            cognition.append(feature[0].clone())
            stats = backend.model.get_action_stats(backend.normalization_keys[context.suite])
            chunks.append(normalize_actions(raw, stats))
            raw_actions.append(raw[0].copy())
            identities.append(identity)
    finally:
        for h in handles:
            h.remove()
    if len(identities) != config.calibration_samples:
        raise ValueError("Incomplete DA-MPA training calibration")
    indices = np.linspace(0, len(identities) - 1, config.probe_steps, dtype=np.int64).tolist()
    axis = drift_weights(np.stack(raw_actions), config)
    layers = [(name, module) for name, (module, side, _) in selected.items() if side == "dit_mlp"]
    scores, timesteps = epsilon_sensitivity(
        backend.model.action_model,
        cognition,
        chunks,
        layers,
        axis,
        indices,
        config.calibration_seed,
    )
    entries = {}
    with torch.no_grad():
        for name, (module, side, block) in selected.items():
            w = module.weight.detach().float()
            scale = (w.abs().amax(1) / 7).clamp_min(1e-8)
            entries[name] = dict(
                codes=(w / scale[:, None]).round().clamp(-8, 7).to(torch.int8).cpu(),
                scales=scale.cpu(),
                activation_amax=torch.stack(values.pop(name))
                .quantile(config.activation_percentile / 100, dim=0)
                .clamp_min(1e-8),
                native_dtype=str(module.weight.dtype),
                side=side,
                block=block,
                bias=None if module.bias is None else module.bias.detach().cpu(),
            )
    torch.save(entries, output / "weights.pt")
    m = dict(
        schema_version=1,
        algorithm="da_mpa",
        complete=True,
        variant="DA_MPA_only_no_CSRC",
        checkpoint_sha256=backend.metadata()["checkpoint_sha256"],
        config=config.calibration_parameters(),
        targets=list(selected),
        calibration_identities=identities,
        axis_weights=axis.tolist(),
        sensitivity=scores,
        probe_indices=indices,
        probe_timesteps=timesteps,
        files={"weights.pt": digest_file(output / "weights.pt")},
        source=source_metadata(),
        wall_seconds=time.monotonic() - start,
    )
    (output / "manifest.json").write_text(json.dumps(m, indent=2))
    return m
