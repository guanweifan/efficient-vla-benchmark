import subprocess
import sys

import pytest

from efficient_vla_benchmark.cli.commands import effective_protocol, parser, protect_resources
from efficient_vla_benchmark.core.config import ExperimentConfig


def test_cli_listing_without_framework_imports():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from efficient_vla_benchmark.cli import main; main(['list']); "
                "assert not any(name in sys.modules for name in "
                "('torch', 'transformers', 'tensorflow', 'prismatic', 'libero'))"
            ),
        ],
        check=True,
    )


def test_cli_overrides_are_saved_in_effective_protocol(tmp_path):
    args = parser().parse_args(
        [
            "benchmark",
            "--config",
            "preset.yaml",
            "--model-path",
            str(tmp_path),
            "--traces",
            "trace.npz",
            "--output-dir",
            "run",
            "--repeats",
            "2",
        ]
    )
    config = ExperimentConfig("example", profiling={"repeats": 5, "warmup": 7})
    result = effective_protocol(config, args)
    assert result.profiling == {"repeats": 2, "warmup": 7, "mode": "control"}
    assert config.profiling["repeats"] == 5


def test_output_cannot_write_inside_model(tmp_path):
    args = parser().parse_args(
        [
            "benchmark",
            "--config",
            "preset.yaml",
            "--model-path",
            str(tmp_path),
            "--traces",
            "trace.npz",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    with pytest.raises(ValueError, match="outside"):
        protect_resources(args)


@pytest.mark.parametrize("settings", [{"warmup": 0}, {"repeats": True}, {"mode": "fake"}])
def test_invalid_profiling_settings_fail_early(settings):
    with pytest.raises(ValueError):
        ExperimentConfig("example", profiling=settings)
