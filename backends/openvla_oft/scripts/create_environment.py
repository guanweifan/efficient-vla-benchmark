"""Create a clean, isolated Linux/Python 3.10 OFT environment from pinned inputs."""

import argparse
import os
import platform
import subprocess
import sys
import venv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--flash-attn-wheel", type=Path, required=True)
    parser.add_argument("--awq-wheel", type=Path, required=True)
    parser.add_argument(
        "--wheelhouse", type=Path, help="Install dependencies offline from this directory"
    )
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10) or platform.system() != "Linux":
        raise SystemExit("This tested environment recipe requires Linux and Python 3.10")
    destination = args.venv.resolve()
    if destination.exists():
        raise SystemExit("Refusing to overwrite an existing environment; choose a new directory")
    for path in (args.flash_attn_wheel, args.awq_wheel):
        if not path.is_file() or path.suffix != ".whl":
            raise SystemExit("Supply existing compatible CUDA extension wheels explicitly")
    if args.wheelhouse is not None and not args.wheelhouse.is_dir():
        raise SystemExit("The supplied offline wheelhouse must be an existing directory")
    package_source = (
        ["--no-index", "--find-links", str(args.wheelhouse.resolve())]
        if args.wheelhouse is not None
        else ["--index-url", "https://pypi.org/simple"]
    )
    root = Path(__file__).resolve().parents[3]
    sources = (
        ("third_party/runtimes/transformers_openvla_oft", False),
        ("third_party/runtimes/dlimp", False),
        ("third_party/benchmarks/libero", True),
        ("backends/openvla_oft/upstream", True),
        (".", False),
        ("backends/openvla_oft", False),
    )
    for relative, _ in sources:
        if not (
            (root / relative / "pyproject.toml").exists() or (root / relative / "setup.py").exists()
        ):
            raise SystemExit("Initialize the pinned runtime submodules before installation")
    venv.EnvBuilder(with_pip=True, symlinks=False).create(destination)
    python = destination / "bin" / "python"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PIP_REQUIRE_VIRTUALENV"] = "true"
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    pip = [str(python), "-m", "pip"]

    def run(arguments):
        subprocess.run(arguments, cwd=root, env=environment, check=True)

    run(
        pip
        + [
            "install",
            *package_source,
            "pip==26.0.1",
            "setuptools==79.0.1",
            "wheel==0.46.3",
            "packaging==26.0",
        ]
    )
    run(
        pip
        + [
            "install",
            *package_source,
            "--no-deps",
            "-r",
            str(root / "backends/openvla_oft/requirements.txt"),
        ]
    )
    for relative, compatibility_mode in sources:
        options = ["--config-settings", "editable_mode=compat"] if compatibility_mode else []
        run(
            pip
            + ["install", "--no-deps", "--no-build-isolation", *options, "-e", str(root / relative)]
        )
    run(
        pip
        + [
            "install",
            "--no-deps",
            str(args.flash_attn_wheel.resolve()),
            str(args.awq_wheel.resolve()),
        ]
    )
    run(pip + ["check"])
    print(
        "Environment created. Run CPU tests and explicit GPU/LIBERO checks before using results.",
        flush=True,
    )


if __name__ == "__main__":
    main()
