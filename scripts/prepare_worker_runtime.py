"""Relocate an explicitly transferred execution environment, never a global installation.

The caller transfers the OFT site-packages and CogACT overlay into dedicated venvs.
CogACT reuses read-only common dependencies from the OFT venv; its stock Transformers,
Prismatic, CogACT and AWQ packages take precedence. Validate both environments afterwards.
Any version differing from the common runtime must also be transferred into the overlay.
Compare runtime-source, active package AND CUDA-extension hashes with a reference run;
a matching package version alone does not establish binary identity.
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--library-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if sys.version_info[:2] != (3, 10):
        raise ValueError("Transferred runtime requires Python 3.10")
    sites = {}
    for name in (".venv", ".venv-cogact"):
        environment = root / name
        if not environment.is_dir():
            raise ValueError("Transfer runtime packages before preparing the worker")
        subprocess.run(
            [sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True
        )
        site = environment / "lib/python3.10/site-packages"
        sites[name] = site
        for pattern in ("*.pth", "*.egg-link"):
            for path in site.glob(pattern):
                text = path.read_text()
                if args.source_root in text:
                    path.write_text(text.replace(args.source_root, str(root)))
    # Only appended fallback dependencies; the CogACT overlay remains first in sys.path.
    (sites[".venv-cogact"] / "evb_common_dependencies.pth").write_text(str(sites[".venv"]) + "\n")
    for name, backend in ((".venv", "openvla_oft"), (".venv-cogact", "cogact")):
        python = root / name / "bin/python"
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--no-build-isolation",
                "-e",
                str(root),
                "-e",
                str(root / "backends" / backend),
                "-e",
                str(root / "third_party/runtimes/dlimp"),
            ],
            check=True,
        )
        subprocess.run([str(python), "-m", "pip", "check"], check=True)
        if args.library_dir:
            library = args.library_dir.resolve(strict=True)
            wrapper = python.with_name("python-evb")
            wrapper.write_text(
                "#!/bin/sh\n"
                f"export LD_LIBRARY_PATH={shlex.quote(str(library))}"
                "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n"
                f'exec {shlex.quote(str(python))} "$@"\n'
            )
            wrapper.chmod(0o755)
        launcher = python.with_name("python-gpu")
        interpreter = python.with_name("python-evb") if args.library_dir else python
        launcher.write_text(
            "#!/bin/sh\n"
            f"exec {shlex.quote(str(interpreter))} "
            f'{shlex.quote(str(root / "scripts/gpu_worker.py"))} "$@"\n'
        )
        launcher.chmod(0o755)


if __name__ == "__main__":
    main()
