"""Download only pinned codec resources, never full policy checkpoints."""

import argparse
import json
from pathlib import Path

from .corpus import digest

MINI_PREFIX = "pretrain_vq+mx-libero_90+fach-7+ng-7+nemb-128+nlatent-512/"
RELEASES = {
    "minivla_rvq": (
        "Stanford-ILIAD/pretrain_vq",
        "30ef2227f97dde1abb6d522ea80af85384235008",
        [MINI_PREFIX + "config.json", MINI_PREFIX + "checkpoints/model.pt"],
    ),
    "mint_sdat": (
        "huangrm/MINT-tokenizer-libero-130",
        "11cbedb05fee2f73b14325fe127bf128ab2e7805",
        ["config.yaml", "ms_vqvae.pth"],
    ),
    "spatialvla": (
        "IPEC-COMMUNITY/spatialvla-4b-224-pt",
        "886c4557dfde53b44c0f74fde7d89bf83a26c3cd",
        ["processor_config.json"],
    ),
    "actioncodec_rvq": (
        "ZibinDong/ActionCodec-Base-RVQft",
        "8d6d8a546498d340d62689e5dcd53436adae3af9",
        [
            "README.md",
            "config.json",
            "configuration_actioncodec.py",
            "model.safetensors",
            "modeling_actioncodec.py",
            "modular_actioncodec.py",
            "rvq.py",
        ],
    ),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--codec", choices=sorted(RELEASES), action="append", required=True)
    a = p.parse_args()
    from huggingface_hub import hf_hub_download

    for name in dict.fromkeys(a.codec):
        repository, revision, files = RELEASES[name]
        root = a.output / name
        rows = []
        for filename in files:
            path = Path(hf_hub_download(repository, filename, revision=revision, local_dir=root))
            rows.append({"name": filename, "size": path.stat().st_size, "sha256": digest(path)})
        manifest = {"repository": repository, "revision": revision, "files": rows}
        target = root / "download_manifest.json"
        if target.exists() and json.loads(target.read_text()) != manifest:
            raise ValueError("Existing codec asset manifest differs")
        target.write_text(json.dumps(manifest, indent=2) + "\n")
        print(name, revision, flush=True)


if __name__ == "__main__":
    main()
