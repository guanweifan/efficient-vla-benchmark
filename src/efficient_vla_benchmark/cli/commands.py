"""Portable configuration dispatch. External paths are supplied only at runtime."""

import argparse
import json
import os
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

from efficient_vla_benchmark.core.backend import RuntimeResources
from efficient_vla_benchmark.core.config import ExperimentConfig
from efficient_vla_benchmark.core.registry import (
    backend_entries,
    build_backend,
    calibrate_backend,
    method_catalog,
    validate_selection,
)


def parser():
    result = argparse.ArgumentParser(prog="evb")
    result.add_argument("--version", action="version", version=version("efficient-vla-benchmark"))
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List installed backend plugins and method compatibility")
    manifest = commands.add_parser("evaluation-manifest", help="Freeze stratified LIBERO Dev-200")
    manifest.add_argument("--libero-root", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--sample-seed", type=int, default=20260924)
    manifest.add_argument("--policy-seed", type=int, default=0)
    derive = commands.add_parser(
        "derive-manifest", help="Freeze timing40/final2000 from Dev-200; no rollout"
    )
    derive.add_argument("--development-manifest", type=Path, required=True)
    derive.add_argument("--purpose", choices=("timing", "final"), required=True)
    derive.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate-config", help="Validate without loading a model")
    validate.add_argument("--config", type=Path, required=True)
    check = commands.add_parser("check-model", help="Load the real model read-only")
    evaluate = commands.add_parser("evaluate", help="Shared LIBERO closed-loop evaluation")
    timing = commands.add_parser("benchmark", help="Shared sequential-observation timing")
    calibration = commands.add_parser("calibrate", help="Run method-provided offline calibration")
    for command in (check, evaluate, timing, calibration):
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--model-path", type=Path, required=True)
        command.add_argument("--device", default="cuda:0")
    for command in (check, evaluate, timing):
        command.add_argument("--artifact-path", type=Path)
    calibration.add_argument("--dataset-path", type=Path, required=True)
    calibration.add_argument("--artifact-dir", type=Path, required=True)
    evaluate.add_argument("--libero-root", type=Path, required=True)
    evaluate.add_argument("--dataset-path", type=Path)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--suites", nargs="+")
    evaluate.add_argument("--tasks", type=int, nargs="+")
    evaluate.add_argument("--trials", type=int)
    evaluate.add_argument("--seed", type=int)
    evaluate.add_argument("--record-traces", action=argparse.BooleanOptionalAction, default=None)
    evaluate.add_argument("--depth", action=argparse.BooleanOptionalAction, default=None)
    evaluate.add_argument("--formal", action="store_true")
    evaluate.add_argument(
        "--authorize-final",
        action="store_true",
        help="Explicitly schedule an owner-approved final-manifest rollout",
    )
    evaluate.add_argument("--episode-manifest", type=Path)
    evaluate.add_argument("--shard-index", type=int, default=0)
    evaluate.add_argument("--num-shards", type=int, default=1)
    timing.add_argument("--traces", type=Path, nargs="+", required=True)
    timing.add_argument("--output-dir", type=Path, required=True)
    timing.add_argument("--repeats", type=int)
    timing.add_argument("--warmup", type=int)
    timing.add_argument("--mode", choices=["control", "query", "prepared"])
    return result


def effective_protocol(config, args):
    """Save effective CLI overrides in the configuration accompanying every result."""
    if args.command == "evaluate":
        settings = {
            "suites": ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
            "tasks": [0],
            "trials": 1,
            "seed": 0,
            "record_traces": False,
            "depth": False,
            **config.evaluation,
        }
        for key in settings:
            if getattr(args, key) is not None:
                settings[key] = getattr(args, key)
        return replace(config, evaluation=settings)
    if args.command == "benchmark":
        settings = {"repeats": 3, "warmup": 10, "mode": "control", **config.profiling}
        for key in settings:
            if getattr(args, key) is not None:
                settings[key] = getattr(args, key)
        return replace(config, profiling=settings)
    return config


def protect_resources(args):
    destination = getattr(args, "output_dir", None) or getattr(args, "artifact_dir", None)
    if destination is None:
        return
    output = destination.resolve()
    protected = [
        getattr(args, key, None)
        for key in (
            "model_path",
            "libero_root",
            "dataset_path",
            "artifact_path",
        )
    ]
    if any(
        output == path.resolve() or output.is_relative_to(path.resolve())
        for path in protected
        if path is not None
    ):
        raise ValueError("Output must be outside model, dataset, artifact and upstream directories")


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "derive-manifest":
        from efficient_vla_benchmark.evaluation.libero.manifest import (
            derive_manifest,
            load_manifest,
        )

        if args.output.exists():
            raise ValueError("Refusing to overwrite an existing episode manifest")
        data = derive_manifest(load_manifest(args.development_manifest), args.purpose)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {"kind": data["kind"], "episodes": len(data["episodes"]), "sha256": data["sha256"]}
            )
        )
        return
    if args.command == "evaluation-manifest":
        from efficient_vla_benchmark.evaluation.libero import configure_libero
        from efficient_vla_benchmark.evaluation.libero.manifest import (
            collect_resources,
            make_manifest,
        )

        if args.output.resolve().is_relative_to(args.libero_root.resolve()):
            raise ValueError("Manifest output must be outside LIBERO source")
        if args.output.exists():
            raise ValueError("Refusing to overwrite an existing episode manifest")
        configure_libero(args.libero_root, args.output.parent / "_private" / "libero")
        data = make_manifest(
            collect_resources(args.libero_root), args.sample_seed, args.policy_seed
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"episodes": 200, "sha256": data["sha256"]}))
        return
    if args.command == "list":
        print(
            json.dumps(
                {
                    "installed_backends": sorted(backend_entries()),
                    "methods": [
                        {
                            "name": spec.name,
                            "category": spec.category,
                            "implementations": {
                                name: impl["variants"]
                                for name, impl in spec.implementations.items()
                            },
                        }
                        for spec in sorted(method_catalog().values(), key=lambda item: item.name)
                    ],
                },
                indent=2,
            )
        )
        return
    config = effective_protocol(ExperimentConfig.load(args.config), args)
    validate_selection(config)
    if args.command == "validate-config":
        print(json.dumps(asdict(config), indent=2))
        return
    protect_resources(args)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.command == "evaluate":
        from efficient_vla_benchmark.evaluation.libero import configure_libero, validate_protocol

        manifest = None
        if args.episode_manifest is not None:
            from efficient_vla_benchmark.evaluation.libero.manifest import (
                SUITES,
                load_manifest,
                select_shard,
                verify_resources,
            )

            if args.formal or any(
                getattr(args, k) is not None for k in ("suites", "tasks", "trials", "seed")
            ):
                raise ValueError(
                    "Manifest owns episode selection; do not override it or use --formal"
                )
            manifest = load_manifest(args.episode_manifest)
            if manifest["kind"] == "final_evaluation" and not args.authorize_final:
                raise ValueError("Final evaluation requires explicit --authorize-final scheduling")
            if manifest["kind"] != "final_evaluation" and args.authorize_final:
                raise ValueError("--authorize-final requires a final evaluation manifest")
            select_shard(manifest, args.shard_index, args.num_shards)
            verify_resources(manifest, args.libero_root)
            config = replace(
                config,
                evaluation={
                    **config.evaluation,
                    "suites": list(SUITES),
                    "tasks": list(range(10)),
                    "trials": manifest["samples_per_task"],
                    "seed": manifest["policy_seed"],
                    "record_traces": bool(args.record_traces),
                },
            )
        elif args.shard_index != 0 or args.num_shards != 1:
            raise ValueError("Sharding requires an explicit episode manifest")
        elif args.authorize_final:
            raise ValueError("--authorize-final requires a final evaluation manifest")
        protocol = config.evaluation
        validate_protocol(
            protocol["suites"],
            protocol["tasks"],
            protocol["trials"],
            args.formal or manifest is not None,
        )
        configure_libero(
            args.libero_root, args.output_dir / "_private" / "libero", args.dataset_path
        )
    elif args.command == "benchmark":
        if not all(path.is_file() for path in args.traces):
            raise ValueError("Every supplied observation trace must exist")
    elif args.command == "calibrate":
        calibrate_backend(
            config,
            RuntimeResources(
                args.model_path,
                args.device,
                dataset_path=args.dataset_path,
                artifact_dir=args.artifact_dir,
            ),
        )
        return
    backend = build_backend(config, args.model_path, args.device, args.artifact_path)
    try:
        if args.command == "check-model":
            print(json.dumps(backend.metadata(), indent=2))
        elif args.command == "evaluate":
            from efficient_vla_benchmark.control.chunk import ChunkPolicy
            from efficient_vla_benchmark.evaluation.libero import evaluate

            evaluate(
                ChunkPolicy(backend, config.execution_horizon),
                backend,
                config,
                args.output_dir,
                protocol["suites"],
                protocol["tasks"],
                protocol["trials"],
                protocol["seed"],
                protocol["record_traces"],
                args.formal,
                manifest=manifest,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
        elif args.command == "benchmark":
            from efficient_vla_benchmark.profiling.benchmark import benchmark

            benchmark(backend, config, args.traces, args.output_dir, **config.profiling)
    finally:
        backend.close()
