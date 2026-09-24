"""Check tracked first-party hygiene and local Markdown targets without model runtimes."""

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

FORBIDDEN_DIRS = {
    "outputs",
    "artifacts",
    "_private",
    ".runtime",
    ".cache",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    "models",
    "datasets",
    "checkpoints",
}
FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".bin",
    ".npy",
    ".npz",
    ".h5",
    ".hdf5",
    ".parquet",
    ".log",
    ".mp4",
    ".pem",
    ".key",
    ".pyc",
    ".pyo",
    ".so",
    ".whl",
    ".bak",
    ".orig",
    ".rej",
    ".swp",
}
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{60,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
)

# Retired operational files may remain locally for an active campaign, but a
# force-add must not silently reintroduce them into the distributable project.
LOCAL_ONLY_FILES = {
    "scripts/run_evaluation_campaign.py",
    "scripts/prepare_worker_runtime.py",
    "scripts/gpu_worker.py",
    "tests/test_gpu_worker_launcher.py",
}


def file_issues(relative, content):
    path = Path(relative)
    issues = []
    if (
        relative in LOCAL_ONLY_FILES
        or any(part in FORBIDDEN_DIRS or part.startswith(".venv") for part in path.parts)
        or relative.startswith("docs/results/")
        or (path.name.startswith(".env") and path.name != ".env.example")
        or path.suffix in FORBIDDEN_SUFFIXES
        or ".tfrecord" in path.name
    ):
        issues.append("generated resource, local configuration or credential file")
    if len(content) > 1024 * 1024:
        issues.append("unexpected first-party file larger than 1 MiB")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return issues + ["unexpected non-text first-party file"]
    if "\x00" in text:
        issues.append("unexpected binary first-party file")
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        issues.append("possible credential; inspect locally without printing it")
    return issues


def link_issues(relative, text, tracked):
    """Check repository targets, including links into uninitialized gitlinks."""
    issues = []
    # Inline code and fenced examples are not Markdown link declarations.
    text = re.sub(r"\x60{3}.*?\x60{3}", "", text, flags=re.S)
    text = re.sub(r"\x60[^\x60\n]+\x60", "", text)
    for match in re.finditer(r"!?\[[^\]\n]*\]\(([^)\n]+)\)", text):
        target = match.group(1).strip()
        if target.startswith("<"):
            target = target.split(">", 1)[0][1:]
        else:
            target = target.split()[0]
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        if parsed.path.startswith("/"):
            issues.append("absolute local Markdown link")
            continue
        parts = list(Path(relative).parent.parts)
        escaped = False
        for part in Path(unquote(parsed.path)).parts:
            if part == "..":
                if not parts:
                    escaped = True
                    break
                parts.pop()
            elif part != ".":
                parts.append(part)
        candidate = "/".join(parts)
        exists = (
            candidate in tracked
            or any(name.startswith(candidate + "/") for name in tracked)
            or any(
                mode == "160000" and candidate.startswith(name + "/")
                for name, mode in tracked.items()
            )
        )
        if escaped or not exists:
            issues.append(f"untracked or missing Markdown target: {target}")
    return issues


def main():
    root = Path(__file__).resolve().parents[1]
    entries = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=root)
    tracked = {}
    for entry in entries.decode().split("\0"):
        if entry:
            header, name = entry.split("\t", 1)
            tracked[name] = header.split()[0]
    failures = []
    checked = 0
    for name, mode in tracked.items():
        if mode == "160000":
            continue
        path = root / name
        if not path.is_file():
            failures.append(f"{name}: tracked file missing from worktree")
            continue
        if path.is_symlink():
            failures.append(f"{name}: first-party symlink requires explicit review")
            continue
        content = path.read_bytes()
        problems = file_issues(name, content)
        if path.suffix == ".md":
            problems += link_issues(name, content.decode("utf-8", errors="replace"), tracked)
        failures.extend(f"{name}: {problem}" for problem in problems)
        checked += 1
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Tracked-file hygiene and Markdown targets passed ({checked} files)")


if __name__ == "__main__":
    main()
