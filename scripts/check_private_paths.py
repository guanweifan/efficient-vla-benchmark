"""Check first-party source, not upstream repositories or private run artifacts."""

import argparse
import re
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    names = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
        )
        .decode()
        .split("\0")
    )
    excluded = {"third_party", "openvla-oft", "cogact", "starvla"}
    private = re.compile(
        r"/(?:home|Users|data)/[A-Za-z0-9_.-]+/"
        r"|(?:192\.168|172\.(?:1[6-9]|2[0-9]|3[01]))\.[0-9]+\.[0-9]+"
        r"|node-[0-9]{2,}"
    )
    failures = []
    for name in sorted(set(names)):
        if not name or Path(name).parts[0] in excluded:
            continue
        path = root / name
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            failures.append(f"{name}: unexpected binary in first-party source")
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if private.search(line):
                failures.append(f"{name}:{number}: machine-local information")
    for index, run_dir in enumerate(args.run_dir):
        if not run_dir.is_dir():
            raise SystemExit("Each supplied public run/artifact directory must exist")
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
                continue
            relative = path.relative_to(run_dir)
            if "_private" in relative.parts:
                continue
            if private.search(path.read_text()):
                failures.append(f"run[{index}]/{relative}: machine-local information")
    if failures:
        raise SystemExit("\n".join(failures))
    print("First-party source and selected public-metadata private-path scan passed")


if __name__ == "__main__":
    main()
