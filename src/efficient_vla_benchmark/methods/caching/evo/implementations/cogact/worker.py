"""One authorized offline job using the common independent episode runner."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from efficient_vla_benchmark.control.chunk import ChunkPolicy
from efficient_vla_benchmark.core.provenance import stable_hash
from efficient_vla_benchmark.evaluation.libero.independent import (
    evaluate_population,
    load_population,
)
from efficient_vla_benchmark.evaluation.libero.runner import configure_libero

from ...config import EVOConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--group", choices=("reference", "quick", "confirmation"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--collect-prior", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Output exists; inspect instead of rerunning search episodes")
    if args.candidate and args.collect_prior:
        raise ValueError("Prior collection must use uncached native rollouts")
    if (args.candidate is None) != (args.group == "reference"):
        raise ValueError("Native prior/reference and candidate states must be disjoint")
    for protected in (args.model_path, args.libero_root, args.population_root):
        if args.output_dir.resolve().is_relative_to(protected.resolve()):
            raise ValueError("Search outputs must not overwrite input resources")
    population = load_population(args.population_root)
    configure_libero(args.libero_root, args.output_dir.parent / "_private/libero")
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from .method import EVO
    from .prior import NativePrior

    host = CogACTBackend(args.model_path)
    collector = None
    try:
        binding = dict(
            num_blocks=2 * len(host.model_access.action_denoiser.blocks),
            num_steps=host.parameters.num_ddim_steps,
            precision=str(next(host.model_access.action_denoiser.parameters()).dtype),
            sampler_parameters=asdict(host.parameters),
            checkpoint_sha256=host._model_manifest["sha256"],
            population_sha256=population["sha256"],
        )
        if args.candidate:
            candidate = json.loads(args.candidate.read_text())
            if candidate["binding"] != binding:
                raise ValueError("Candidate belongs to another model/sampler/population")
            budget = candidate["budget"]
            manifest = {
                **binding,
                "complete": False,
                "layouts": {str(budget): {"layout": candidate["layout"]}},
            }
            host.attach_method(EVO(host, EVOConfig(refresh_budget=budget), manifest))
        elif args.collect_prior:
            collector = NativePrior(host)
        metadata = dict(
            backend=host.metadata(),
            binding=binding,
            candidate_sha256=stable_hash(candidate) if args.candidate else None,
            prior_observation_only=collector is not None,
            execution_horizon=8,
        )
        summary = evaluate_population(
            ChunkPolicy(host, 8),
            metadata,
            args.population_root,
            args.group,
            args.output_dir,
            before_episode=collector.before_episode if collector else None,
        )
        if collector:
            import numpy as np

            np.savez_compressed(args.output_dir / "prior.npz", dissimilarity=collector.result())
            (args.output_dir / "prior_samples.json").write_text(
                json.dumps(collector.identities, indent=2) + "\n"
            )
        (args.output_dir / "binding.json").write_text(json.dumps(binding, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    finally:
        if collector:
            collector.close()
        host.close()


if __name__ == "__main__":
    main()
