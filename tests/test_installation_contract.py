"""Installation plans must remain isolated and documented commands must stay live."""

import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "environment_installer", ROOT / "scripts/create_environment.py"
)
INSTALLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSTALLER)


@pytest.mark.parametrize("backend", ["openvla_oft", "cogact"])
@pytest.mark.parametrize("with_awq", [False, True])
@pytest.mark.parametrize("offline", [False, True])
def test_install_plan_is_backend_specific_and_isolated(
    tmp_path, monkeypatch, backend, with_awq, offline
):
    root = tmp_path / "project"
    for relative, _ in INSTALLER.source_packages(backend):
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "pyproject.toml").write_text("[project]\n")
    monkeypatch.setattr(INSTALLER, "__file__", str(root / "scripts/create_environment.py"))
    monkeypatch.setattr(INSTALLER.sys, "version_info", (3, 10))
    monkeypatch.setattr(INSTALLER.platform, "system", lambda: "Linux")
    monkeypatch.setattr(INSTALLER.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setenv("PYTHONPATH", "unexpected-parent-packages")
    destination = tmp_path / "isolated"
    flash = tmp_path / "flash.whl"
    flash.touch()
    arguments = [
        "installer",
        "--backend",
        backend,
        "--venv",
        str(destination),
        "--flash-attn-wheel",
        str(flash),
    ]
    if with_awq:
        awq = tmp_path / "awq.whl"
        awq.touch()
        arguments.extend(["--awq-wheel", str(awq)])
    if offline:
        wheelhouse = tmp_path / "wheels"
        wheelhouse.mkdir()
        (wheelhouse / "pip-26.0.1-py3-none-any.whl").touch()
        arguments.extend(["--wheelhouse", str(wheelhouse)])
    monkeypatch.setattr(INSTALLER.sys, "argv", arguments)
    creation, commands = [], []

    class Environment:
        def __init__(self, **kwargs):
            assert not kwargs.get("system_site_packages", False)
            assert kwargs["with_pip"] is not offline

        def create(self, path):
            creation.append(path)

    monkeypatch.setattr(INSTALLER.venv, "EnvBuilder", Environment)
    monkeypatch.setattr(
        INSTALLER.subprocess, "run", lambda args, **kwargs: commands.append((args, kwargs))
    )
    INSTALLER.main()
    if offline:
        assert commands[0][0][1] == str(wheelhouse / "pip-26.0.1-py3-none-any.whl" / "pip")
        assert "--no-index" in commands[0][0]
    else:
        assert commands[0][0][1:3] == ["-m", "pip"]
    assert creation == [destination]
    sources = [args[-1] for args, _ in commands if "-e" in args]
    expected = [str(root / name) for name, _ in INSTALLER.source_packages(backend)]
    assert sources == expected
    other = "cogact" if backend == "openvla_oft" else "openvla_oft"
    assert str(root / "backends" / other) not in sources
    if backend == "cogact":
        assert not any("transformers_openvla_oft" in name for name in sources)
    else:
        assert not any("prismatic_cogact" in name for name in sources)
    for _, kwargs in commands:
        assert "PYTHONPATH" not in kwargs["env"]
        assert "PYTHONHOME" not in kwargs["env"]
        assert kwargs["env"]["PYTHONNOUSERSITE"] == "1"
        assert kwargs["check"]
    extension_command = commands[-2][0]
    assert str(flash) in extension_command
    assert any("awq.whl" in arg for arg in extension_command) == with_awq
    assert commands[-1][0][-1] == "check"
    assert any(
        str(root / "backends" / backend / "requirements.txt") in args for args, _ in commands
    )


def test_installer_refuses_existing_destination_before_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(INSTALLER.sys, "version_info", (3, 10))
    monkeypatch.setattr(INSTALLER.platform, "system", lambda: "Linux")
    marker = tmp_path / "existing.txt"
    marker.write_text("preserve")
    monkeypatch.setattr(
        INSTALLER.sys,
        "argv",
        [
            "installer",
            "--backend",
            "cogact",
            "--venv",
            str(tmp_path),
            "--flash-attn-wheel",
            str(tmp_path / "missing.whl"),
        ],
    )
    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        INSTALLER.main()
    assert marker.read_text() == "preserve"


def test_installer_without_ensurepip_requires_offline_bootstrap(tmp_path, monkeypatch):
    monkeypatch.setattr(INSTALLER.sys, "version_info", (3, 10))
    monkeypatch.setattr(INSTALLER.platform, "system", lambda: "Linux")
    monkeypatch.setattr(INSTALLER.importlib.util, "find_spec", lambda name: None)
    destination = tmp_path / "fresh"
    flash = tmp_path / "flash.whl"
    flash.touch()
    monkeypatch.setattr(
        INSTALLER.sys,
        "argv",
        [
            "installer",
            "--backend",
            "cogact",
            "--venv",
            str(destination),
            "--flash-attn-wheel",
            str(flash),
        ],
    )
    with pytest.raises(SystemExit, match="lacks ensurepip"):
        INSTALLER.main()
    assert not destination.exists()


def test_documented_literal_experiment_commands_resolve():
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    checked = 0
    for name in paths:
        path = ROOT / name
        if path.suffix != ".md" or not path.is_file():
            continue
        for block in re.findall(r"```(?:bash|sh|shell)\n(.*?)```", path.read_text(), re.S):
            for target in re.findall(r"configs/experiments/[\w/-]+\.yaml", block):
                assert (ROOT / target).is_file(), (name, target)
                checked += 1
    assert checked > 10


def test_cogact_lock_is_complete_and_does_not_install_oft_fork():
    lines = (ROOT / "backends/cogact/requirements.txt").read_text().splitlines()
    pins = dict(line.split("==") for line in lines if line and not line.startswith("#"))
    assert pins["transformers"] == "4.40.1"
    assert pins["robosuite"] == "1.4.0"
    assert {"torch", "tensorflow", "bddl", "future", "easydict", "mujoco"} <= pins.keys()
    assert all("git+" not in line for line in lines)
