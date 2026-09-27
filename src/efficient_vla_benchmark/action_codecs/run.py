"""Run one explicitly configured codec; paths are command-line resources only."""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import yaml

from efficient_vla_benchmark.core.provenance import driver_version, source_metadata

from .corpus import Corpus, digest
from .evaluation import evaluate
from .loading import factory
from .normalization import fit_quantiles


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--sources", type=Path)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    config = yaml.safe_load(a.config.read_text())
    if set(config) != {"factory", "parameters"}:
        raise ValueError("Codec config must contain factory and parameters only")
    source_before = source_metadata()
    corpus = Corpus(a.corpus)
    calibration = {"corpus_sha256": corpus.sha256, "quantiles": fit_quantiles(corpus)}
    codec = factory(config["factory"])(
        assets=a.assets,
        sources=a.sources,
        device=a.device,
        calibration=calibration,
        corpus=corpus,
        **config["parameters"],
    )
    try:
        result = evaluate(codec, corpus, a.output)
        source_after = source_metadata()
        provenance = {
            "source": source_before,
            "source_unchanged": source_before == source_after,
            "driver_version": driver_version(),
            "config": config,
            "config_sha256": digest(a.config),
            "host_processor": platform.processor(),
            "codec_packages": {
                name: importlib.metadata.version(name)
                for name in ("scipy", "vector-quantize-pytorch", "einops", "diffusers")
            },
        }
        (a.output / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if source_before != source_after:
            raise RuntimeError("Source/runtime identity changed during codec evaluation")
        if not result["complete"]:
            raise SystemExit(1)
    finally:
        codec.close()


if __name__ == "__main__":
    main()
