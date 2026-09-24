import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def hygiene():
    script = Path(__file__).resolve().parents[1] / "scripts/check_repository.py"
    spec = importlib.util.spec_from_file_location("repository_check", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "name",
    [
        "outputs/run.json",
        "docs/results/run.json",
        ".env",
        ".env.local",
        ".venv/lib/module.py",
        "weights/model.safetensors",
        "logs/run.log",
        "_private/settings.yaml",
        "models/config.json",
        ".runtime/evaluation/launch.py",
        "scripts/run_evaluation_campaign.py",
        "scripts/prepare_worker_runtime.py",
        "scripts/gpu_worker.py",
        "tests/test_gpu_worker_launcher.py",
    ],
)
def test_generated_and_private_files_cannot_enter_tracked_tree(hygiene, name):
    assert hygiene.file_issues(name, b"{}")


def test_source_and_example_config_are_allowed(hygiene):
    assert not hygiene.file_issues("src/example.py", b"pass\n")
    assert not hygiene.file_issues(".env.example", b"MODEL_DIR=replace-me\n")


def test_binary_large_files_and_possible_tokens_are_rejected(hygiene):
    assert hygiene.file_issues("unexpected.dat", b"\x00data")
    assert hygiene.file_issues("oversized.txt", b"x" * (1024 * 1024 + 1))
    assert hygiene.file_issues("config.txt", ("ghp_" + "x" * 36).encode())


def test_markdown_links_resolve_from_document_and_need_no_submodule_checkout(hygiene):
    tracked = {
        "README.md": "100644",
        "docs/development.md": "100644",
        "src/example.py": "100644",
        "third_party/native": "160000",
    }
    text = (
        "[readme](../README.md) [source](../src/) "
        "[native](../third_party/native/README.md) [anchor](#section) "
        "[remote](https://example.com/file.md)"
    )
    assert not hygiene.link_issues("docs/development.md", text, tracked)
    assert hygiene.link_issues("README.md", "[missing](docs/removed.md)", tracked)
    assert hygiene.link_issues("README.md", "[escape](../outside.md)", tracked)
    assert hygiene.link_issues("README.md", "[absolute](/private/file.md)", tracked)


def test_ignored_local_evidence_is_not_a_valid_git_link(hygiene):
    tracked = {"docs/benchmark_results.md": "100644"}
    assert hygiene.link_issues(
        "docs/benchmark_results.md", "[record](../outputs/run.json)", tracked
    )
