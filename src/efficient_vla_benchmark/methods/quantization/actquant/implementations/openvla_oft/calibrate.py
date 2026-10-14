"""HSIC + action/NLL Fisher on frozen, training-only native OFT observations.

Single-frame batches follow the author's supported default. Gradients are squared
only AFTER the full sample backward contribution reaches the parameter leaf.
There is no optimizer, model update, evaluation-data use or artificial target.
"""

import gc
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from efficient_vla_benchmark.calibration.libero import training_action_chunks
from efficient_vla_benchmark.core.provenance import digest_file, source_metadata

from ...vendor import REVISION, author_module, reference_source

PROJECTIONS = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}


def normalize_actions(actions, stats):
    value = np.asarray(actions, dtype=np.float32).copy()
    if value.shape != (8, 7) or not np.isfinite(value).all():
        raise ValueError("ActQuant needs complete finite eight-action chunks")
    # Same LIBERO OXE transform as training: clipped close-gripper -> open-gripper.
    value[:, -1] = 1 - np.clip(value[:, -1], 0, 1)
    lo, hi = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    mask = np.asarray(stats.get("mask", [True] * 7), bool)
    return (
        np.where(mask, 2 * (value - lo) / (hi - lo + 1e-8) - 1, value)
        .clip(-1, 1)
        .astype(np.float32)
    )


def calibrate(backend, config, dataset, output, runtime_path):
    reference_source(runtime_path)
    if backend.method is not None or backend.metadata()["precision"] != "bfloat16":
        raise ValueError("ActQuant calibration needs fresh BF16 OFT")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite calibration evidence")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    samples = list(
        training_action_chunks(dataset, config.calibration_samples, config.calibration_seed, 8)
    )
    model = backend.model.language_model
    tensor_in, tensor_out = {}, {}
    captures = {}
    handles = []
    for i, layer in enumerate(model.model.layers):
        for projection, suffix in PROJECTIONS.items():
            name = f"blk.{i}.{suffix}.weight"
            tensor_in[name] = []
            tensor_out[name] = []

            def hook(module, ins, out, name=name):
                tensor_in[name].append(ins[0][0].detach().float().mean(0).cpu().numpy())
                tensor_out[name].append(out[0].detach().float().mean(0).cpu().numpy())

            handles.append(layer.get_submodule(projection).register_forward_hook(hook))

    def inputs_hook(module, ins, kwargs):
        captures["embeddings"] = kwargs["inputs_embeds"].detach().cpu()
        captures["mask"] = kwargs["attention_mask"].detach().cpu()

    handles.append(model.model.register_forward_pre_hook(inputs_hook, with_kwargs=True))

    def head_hook(module, ins, out):
        captures["hidden"] = ins[0].detach().cpu().reshape(1, 56, -1)

    handles.append(backend.action_head.model.register_forward_hook(head_hook))
    frames = []
    ground_truth = []
    identities = []
    x_pool = []
    try:
        for index, (context, observation, identity, actions) in enumerate(samples):
            backend.reset(context)
            backend.predict_chunk(observation)
            frame = {k: v.clone() for k, v in captures.items()}
            gt = normalize_actions(actions, backend.model.get_action_stats(backend.unnorm_key))
            frame["gt"] = gt
            frame["tokens"] = np.concatenate(
                [
                    backend.processor.tokenizer.vocab_size
                    - np.digitize(gt.flatten(), np.linspace(-1, 1, 256)),
                    [2],
                ]
            )
            frames.append(frame)
            ground_truth.append(gt.flatten())
            identities.append(identity)
            x_pool.append(frame["embeddings"][0].float().mean(0).numpy())
            if index % 20 == 0:
                print(f"ActQuant: native collection {index + 1}/{len(samples)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    del samples
    x = np.stack(x_pool).astype(np.float32)
    y = np.stack(ground_truth).astype(np.float32)
    y = (y - y.mean(0)) / (y.std(0) + 1e-8)
    hsic = author_module(runtime_path, "tools/hsic/compute_hsic_tensor.py")
    scores = hsic.compute_tensor_hsic_scores(
        x,
        y,
        {k: np.stack(v).astype(np.float32) for k, v in tensor_in.items()},
        {k: np.stack(v).astype(np.float32) for k, v in tensor_out.items()},
        kernel_hidden="rbf",
        kernel_y="linear",
        hsic_mode="cca",
        sigma=None,
        lx=1.0,
        ly=1.0,
    )
    if len(scores["tensors"]) != 224 or any(
        not np.isfinite(v["F_out"]) for v in scores["tensors"].values()
    ):
        raise ValueError("Incomplete/nonfinite HSIC scores")
    scores["metadata"] = {
        "y_source": "normalized_demonstration_actions",
        "pool_all": True,
        "hsic_mode": "cca",
        "sigma": None,
        "lx": 1.0,
        "ly": 1.0,
        "reference_revision": REVISION,
    }
    (output / "hsic.json").write_text(json.dumps(scores, indent=2) + "\n")
    tensor_in.clear()
    tensor_out.clear()
    del x, y, x_pool, ground_truth
    for par in backend.model.parameters():
        par.requires_grad_(False)
    targets = []
    for i, layer in enumerate(model.model.layers):
        for projection, suffix in PROJECTIONS.items():
            par = layer.get_submodule(projection).weight
            par.requires_grad_(True)
            targets.append((f"blk.{i}.{suffix}.weight", par))
    model.lm_head.weight.requires_grad_(True)
    targets.append(("output.weight", model.lm_head.weight))
    backend.action_head.float()
    for par in backend.action_head.parameters():
        par.requires_grad_(False)
    fisher = {name: torch.zeros(par.shape, dtype=torch.float32) for name, par in targets}
    collected = set()

    def gradient_hook(name):
        def hook(par):
            # One leaf accumulation per complete frame, never square partial microgradients.
            grad = par.grad.detach().float().square().cpu()
            if not bool(torch.isfinite(grad).all()):
                raise ValueError("Nonfinite AMF gradient")
            if name in collected:
                raise ValueError("Repeated AMF leaf contribution within one frame")
            fisher[name].add_(grad)
            collected.add(name)
            par.grad = None

        return hook

    hooks = [par.register_post_accumulate_grad_hook(gradient_hook(name)) for name, par in targets]
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    losses = []
    try:
        for index, frame in enumerate(frames):
            collected.clear()
            hidden = model.model(
                inputs_embeds=frame["embeddings"].to(backend.device),
                attention_mask=frame["mask"].to(backend.device),
                use_cache=False,
            ).last_hidden_state
            # Native OFT's prompt excludes BOS; verified against the actual head input.
            readout = hidden[:, -58:-2]
            if index == 0 and not torch.equal(readout.detach().cpu(), frame["hidden"]):
                raise ValueError("Calibration readout differs from native OFT hidden states")
            truth = torch.tensor(frame["gt"], device=backend.device).unsqueeze(0)
            prediction = backend.action_head.predict_action(readout.float())
            action_loss = (prediction - truth).abs().sum()
            logits = model.lm_head(hidden[:, -59:-2].bfloat16()).float()
            tokens = torch.tensor(frame["tokens"], device=backend.device).long()
            nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens)
            loss = config.alpha * action_loss + (1 - config.alpha) * nll
            losses.append({"action_l1": float(action_loss.detach()), "nll": float(nll.detach())})
            loss.backward()
            if len(collected) != len(targets):
                raise ValueError("Missing AMF parameter gradients")
            del hidden, readout, truth, prediction, action_loss, logits, tokens, nll, loss
            print(f"ActQuant: AMF {index + 1}/{len(frames)}", flush=True)
    finally:
        for hook in hooks:
            hook.remove()
        for _, par in targets:
            par.grad = None
            par.requires_grad_(False)
    for value in fisher.values():
        value.div_(len(frames))
    author = author_module(runtime_path, "tools/fisher-diag/compute_amf_lq.py")
    author.save_af_gguf(
        {n: v.numpy() for n, v in fisher.items()},
        str(output / "amf.gguf"),
        len(frames),
        "frozen_training_rlds",
        "l1",
        1.0,
        config.alpha,
        None,
    )
    manifest = {
        "schema_version": 1,
        "algorithm": "actquant_calibration",
        "complete": True,
        "reference_revision": REVISION,
        "checkpoint_sha256": backend.metadata()["checkpoint_sha256"],
        "config": config.calibration_parameters(),
        "batch_size": 1,
        "optimizer_updates": 0,
        "calibration_identities": identities,
        "losses": losses,
        "calibration_seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "files": {n: digest_file(output / n) for n in ("hsic.json", "amf.gguf")},
        "source": source_metadata(),
        "scope": "224_llm_HSIC_tensors_225_AMF_tensors",
        "adaptations": [
            "balanced_training_demonstrations_not_author_episode_pool",
            "native_inference_equivalent_readout_and_corresponding_NLL_positions",
            "author_default_single_frame_batch_with_leaf_gradient_offload",
            "author_FP32_action_head_for_calibration_only",
        ],
    }
    (output / "calibration_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
