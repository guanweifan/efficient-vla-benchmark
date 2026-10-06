"""Fetch pinned optional codec runtimes and licenses, without cloning a workspace."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

specs = {
    "spatialvla": (
        "SpatialVLA/SpatialVLA",
        "18fd74b2a633ec8d9ec7aadcd803969555cc9fbd",
        ["README.md", "model/action_tokenizer.py"],
    ),
    "mint_sdat": (
        "RenMing-Huang/MINT",
        "bed9f0c0e2445a123814e89e4ba8c35a9578ce52",
        [
            "LICENSE",
            "README.md",
            "SDAT/model/__init__.py",
            "SDAT/model/vqvae.py",
            "SDAT/model/basic_vae.py",
            "SDAT/model/quant.py",
            "SDAT/utils/dist.py",
            "SDAT/utils/normalizer.py",
            "SDAT/core.py",
            "SDAT/trainer.py",
        ],
    ),
    "minivla_rvq": (
        "jayLEE0301/vq_bet_official",
        "09d4851288ca5deaaa1ab367a208e520f8ee9a84",
        [
            "LICENSE",
            "vqvae/__init__.py",
            "vqvae/vqvae.py",
            "vqvae/vqvae_utils.py",
            "vector_quantize_pytorch/__init__.py",
            "vector_quantize_pytorch/residual_vq.py",
            "vector_quantize_pytorch/vector_quantize_pytorch.py",
        ],
    ),
    "vq_vla": (
        "xiaoxiao0406/VQ-VLA",
        "7fb78e63f5c0f7baf6d340166ffea527c6514610",
        [
            "LICENSE",
            "train_vae/scripts/action_vqvae_config/config.json",
            "train_vae/scripts/action_vqvae_config/config_action_type_pe_time_pe.json",
            *[
                "prismatic/action_vqvae/" + f
                for f in [
                    "__init__.py",
                    "action_vqvae_wrapper.py",
                    "modeling_block.py",
                    "modeling_causal_conv.py",
                    "modeling_causal_vae.py",
                    "modeling_enc_dec.py",
                    "modeling_resnet.py",
                    "residual_vq.py",
                    "vector_quantize_pytorch.py",
                    "vqvae_utils.py",
                ]
            ],
        ],
    ),
    "x_tokenizer": (
        "X-Square-Robot/X-Tokenizer",
        "9733990b16a07d51fb8b5d305ad8c16abb57aef2",
        [
            "LICENSE",
            "README.md",
            *[
                "xtokenizer/" + f
                for f in [
                    "__init__.py",
                    "api.py",
                    "configs/robot_types.yaml",
                    "data/__init__.py",
                    "data/action_layout.py",
                    "data/normalization.py",
                    "data/rotation.py",
                    "data/statistics.py",
                    "model/__init__.py",
                    "model/decoder.py",
                    "model/encoder.py",
                    "model/position_encoding.py",
                    "model/rvq.py",
                    "model/tokenizer.py",
                    "registry/__init__.py",
                    "registry/robot_type.py",
                    "tools/__init__.py",
                    "tools/statistics_builder.py",
                ]
            ],
        ],
    ),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--only", choices=sorted(specs), action="append")
    a = p.parse_args()
    for name, (repo, revision, files) in specs.items():
        if a.only and name not in a.only:
            continue
        directory = a.output / name
        directory.mkdir(parents=True, exist_ok=True)
        identities = []
        for file in files:
            data = urlopen(
                f"https://raw.githubusercontent.com/{repo}/{revision}/{file}", timeout=90
            ).read()
            original = hashlib.sha256(data).hexdigest()
            if name == "vq_vla" and file.endswith(".py"):
                data = data.replace(b"from prismatic.action_vqvae.", b"from .")
            path = directory / file
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_bytes() != data:
                raise ValueError("Existing source differs")
            path.write_bytes(data)
            identities.append(
                dict(name=file, upstream_sha256=original, sha256=hashlib.sha256(data).hexdigest())
            )
        manifest = dict(
            repository=repo,
            revision=revision,
            files=identities,
            adaptation="VQ-VLA absolute package imports made relative; no numerical edits"
            if name == "vq_vla"
            else "none",
        )
        path = directory / "source_manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            old = json.loads(path.read_text())
            if any(old.get(k) != manifest[k] for k in ("repository", "revision", "adaptation")):
                raise ValueError("Changed source identity")
            if any(row not in identities for row in old["files"]):
                raise ValueError(
                    "Existing source files changed; only identical-set extensions allowed"
                )
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(name, revision, len(files), flush=True)


if __name__ == "__main__":
    main()
