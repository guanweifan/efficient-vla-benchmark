"""Validate a local complete model export without following training resource paths."""

import json
from pathlib import Path

from efficient_vla_benchmark.core.provenance import file_manifest

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def checkpoint_files(model_path: Path):
    model_path = model_path.resolve()
    if model_path.is_file():
        checkpoint = model_path
        if checkpoint.suffix != ".pt" or checkpoint.parent.name != "checkpoints":
            raise ValueError("Expected a checkpoint .pt file inside checkpoints/")
        root = checkpoint.parents[1]
    elif model_path.is_dir():
        root = model_path
        matches = sorted((root / "checkpoints").glob("*.pt"))
        if len(matches) != 1:
            raise ValueError("Supply exactly one model checkpoint, or its explicit .pt path")
        checkpoint = matches[0]
    else:
        raise ValueError("The supplied local CogACT checkpoint does not exist")
    required = (
        "config.json",
        "dataset_statistics.json",
        "tokenizer/config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    )
    if not all((root / name).is_file() for name in required):
        raise ValueError("A complete export with bundled tokenizer and statistics is required")
    config = json.loads((root / "config.json").read_text())
    expected = {
        "action_model_type": "DiT-B",
        "action_dim": 7,
        "future_action_window_size": 15,
        "past_action_window_size": 0,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint architecture differs from the supported CogACT variant")
    if config.get("use_ema", False):
        raise ValueError("EMA checkpoints need an explicit supported loading recipe")
    if config.get("vla", {}).get("base_vlm") != "prism-dinosiglip-224px+7b":
        raise ValueError("Unsupported CogACT VLM architecture")
    statistics = json.loads((root / "dataset_statistics.json").read_text())
    keys = {}
    for suite in SUITES:
        key = suite if suite in statistics else suite + "_no_noops"
        if key not in statistics:
            raise ValueError("Checkpoint lacks normalization statistics for all four suites")
        action = statistics[key]["action"]
        if len(action["q01"]) != 7 or len(action["q99"]) != 7:
            raise ValueError("Expected seven-dimensional action statistics")
        mask = action.get("mask")
        if mask is None or len(mask) != 7 or bool(mask[-1]):
            raise ValueError("LIBERO gripper must be explicitly excluded from normalization")
        keys[suite] = key
    return root, checkpoint, statistics, keys


def model_manifest(root, checkpoint):
    names = ["config.json", "dataset_statistics.json", str(checkpoint.relative_to(root))]
    names += [
        str(path.relative_to(root))
        for path in sorted((root / "tokenizer").iterdir())
        if path.is_file() and not path.name.startswith("._")
    ]
    return file_manifest(root, names)
