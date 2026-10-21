"""Install one native backend in an isolated Linux/Python 3.10 environment."""

import argparse
import importlib.util
import os
import platform
import subprocess
import sys
import venv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("openvla_oft", "cogact"), required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--flash-attn-wheel", type=Path, required=True)
    parser.add_argument("--awq-wheel", type=Path, help="Optional packed-quantization extension")
    parser.add_argument(
        "--wheelhouse", type=Path, help="Install dependencies offline from this directory"
    )
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10) or platform.system() != "Linux":
        raise SystemExit("This tested environment recipe requires Linux and Python 3.10")
    destination = args.venv.resolve()
    if destination.exists():
        raise SystemExit("Refusing to overwrite an existing environment; choose a new directory")
    extensions = [args.flash_attn_wheel]
    if args.awq_wheel is not None:
        extensions.append(args.awq_wheel)
    for path in extensions:
        if not path.is_file() or path.suffix != ".whl":
            raise SystemExit("Supply existing compatible CUDA extension wheels explicitly")
    if args.wheelhouse is not None and not args.wheelhouse.is_dir():
        raise SystemExit("The supplied offline wheelhouse must be an existing directory")
    bootstrap = None
    if args.wheelhouse is not None:
        bootstrap = args.wheelhouse.resolve() / "pip-26.0.1-py3-none-any.whl"
        if not bootstrap.is_file():
            raise SystemExit("Offline installation needs pip-26.0.1-py3-none-any.whl")
    elif importlib.util.find_spec("ensurepip") is None:
        raise SystemExit(
            "Python lacks ensurepip; use Python 3.10 with venv support or an explicit wheelhouse"
        )
    package_source = (
        ["--no-index", "--find-links", str(args.wheelhouse.resolve())]
        if args.wheelhouse is not None
        else ["--index-url", "https://pypi.org/simple"]
    )
    root = Path(__file__).resolve().parents[1]
    sources = source_packages(args.backend)
    for relative, _ in sources:
        if not (
            (root / relative / "pyproject.toml").exists() or (root / relative / "setup.py").exists()
        ):
            raise SystemExit("Initialize the pinned runtime submodules before installation")
    venv.EnvBuilder(with_pip=bootstrap is None, symlinks=False).create(destination)
    python = destination / "bin" / "python"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PIP_REQUIRE_VIRTUALENV"] = "true"
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    pip = [str(python), "-m", "pip"]

    def run(arguments):
        subprocess.run(arguments, cwd=root, env=environment, check=True)

    run(
        ([str(python), str(bootstrap / "pip")] if bootstrap is not None else pip)
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
            str(root / "backends" / args.backend / "requirements.txt"),
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
            *(str(path.resolve()) for path in extensions),
        ]
    )
    run(pip + ["check"])
    print(
        "Environment created. Run CPU tests and explicit GPU/LIBERO checks before using results.",
        flush=True,
    )


def source_packages(backend):
    """Keep conflicting Prismatic/Transformers sources out of the other backend."""
    if backend == "openvla_oft":
        model_sources = [("third_party/runtimes/transformers_openvla_oft", False)]
    elif backend == "cogact":
        model_sources = [("third_party/runtimes/prismatic_cogact", False)]
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    return [
        ("third_party/runtimes/dlimp", False),
        *model_sources,
        ("third_party/benchmarks/libero", True),
        (f"backends/{backend}/upstream", True),
        (".", False),
        (f"backends/{backend}", False),
    ]


if __name__ == "__main__":
    main()
