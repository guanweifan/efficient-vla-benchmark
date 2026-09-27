import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from efficient_vla_benchmark.core.provenance import digest_file, stable_hash

from ..artifact import read_manifest
from ..config import PACEConfig
from ..method import PACE
from ..selection import instruction_key, select_prefix, speed_profile, valleys


def commands(speed):
    actions = np.zeros((len(speed), 7))
    actions[:, 0] = speed
    return actions


@pytest.mark.parametrize(
    "bad",
    [
        {"enabled": 1},
        {"smoothing_window": 2},
        {"minimum_spacing": 0},
        {"calibration_percentile": float("nan")},
        {"calibration_percentile": -1},
        {"max_horizon": True},
        {"max_horizon": 0},
        {"unknown": 0},
    ],
)
def test_invalid_config(bad):
    with pytest.raises(ValueError):
        PACEConfig.parse(bad)


def artifact(root, config=PACEConfig()):
    root.mkdir()
    statistics = root / "prominences.json"
    values = {f"task {i}": [0.1, 0.2, 0.3, 0.4] for i in range(40)}
    statistics.write_text(json.dumps(values))
    samples = [dict(split="train", instruction=f"task {i}", episode_index=i) for i in range(40)]
    manifest = dict(
        schema_version=1,
        episodes_per_task=1,
        algorithm="pace",
        complete=True,
        split="train",
        calibration_parameters=config.calibration_parameters(),
        instructions=sorted(values),
        samples=samples,
        samples_sha256=stable_hash(samples),
        statistics_sha256=digest_file(statistics),
    )
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_speed_is_displacement_not_acceleration_and_excludes_gripper():
    config = PACEConfig(smoothing_window=1)
    actions = commands([4, 3, 0, 3, 4, 4, 0, 4])
    actions[:, 3:] = 99
    np.testing.assert_array_equal(speed_profile(actions, config), actions[:, 0])
    assert select_prefix(actions, 1, config) == (3, 2)
    assert select_prefix(actions, 4.1, config) == (8, 0)
    assert select_prefix(actions, 1, replace(config, max_horizon=4)) == (3, 1)
    assert select_prefix(actions, 4.1, replace(config, max_horizon=4)) == (4, 0)
    assert select_prefix(actions * 10, 10, config) == select_prefix(actions, 1, config)


def test_constant_motion_smoothing_and_valley_acceptance():
    config = PACEConfig()
    actions = commands(np.ones(8))
    np.testing.assert_array_equal(speed_profile(actions, config), np.ones(8))
    assert select_prefix(actions, 0, config) == (8, 0)
    profile = speed_profile(commands([0, 3, 6, 3, 0]), config)
    np.testing.assert_allclose(profile, [1, 3, 4, 3, 1])
    boundaries, prominence = valleys(
        commands([5, 2, 2, 5, 1, 5]), replace(config, smoothing_window=1)
    )
    np.testing.assert_array_equal(boundaries, [2, 5])
    np.testing.assert_array_equal(prominence, [3, 4])


def test_corruption_incomplete_tasks_and_incompatible_profile_rejected(tmp_path):
    path = artifact(tmp_path / "artifact")
    manifest, thresholds = read_manifest(path, PACEConfig())
    assert len(thresholds) == 40
    assert thresholds["task 0"] == pytest.approx(0.115)
    _, strict = read_manifest(path, PACEConfig(calibration_percentile=25))
    assert strict["task 0"] == pytest.approx(0.175)
    with pytest.raises(ValueError):
        read_manifest(path, PACEConfig(minimum_spacing=3))
    (path / "prominences.json").write_text("{}")
    with pytest.raises(ValueError):
        read_manifest(path, PACEConfig())
    (path / "prominences.json").write_text(json.dumps({"task 0": [0.1]}))
    manifest["statistics_sha256"] = digest_file(path / "prominences.json")
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        read_manifest(path, PACEConfig())


def test_shared_controller_reset_disabled_and_unknown_instruction(tmp_path):
    path = artifact(tmp_path / "artifact")
    method = PACE(PACEConfig(), path, 8)
    actions = commands([4, 4, 0, 0, 4, 4, 4, 4])
    with pytest.raises(RuntimeError):
        method.select_execution_horizon(actions, 8)
    method.observe(SimpleNamespace(instruction=" TASK_0 "))
    assert instruction_key(" TASK_0 ") == "task 0"
    assert 1 <= method.select_execution_horizon(actions, 8) < 8
    method.reset(None)
    assert method.instruction is None
    with pytest.raises(ValueError):
        method.observe(SimpleNamespace(instruction="new task"))
    method.close()
    method.close()
    with pytest.raises(RuntimeError):
        method.observe(SimpleNamespace(instruction="task 0"))
    disabled = PACE(PACEConfig(enabled=False), path, 8)
    disabled.observe(SimpleNamespace(instruction="task 0"))
    assert disabled.select_execution_horizon(actions, 8) == 8


def test_real_training_format_calibration_identity_and_no_overwrite(tmp_path):
    parquet = pytest.importorskip("pyarrow.parquet")
    arrow = pytest.importorskip("pyarrow")
    from ..calibrate import calibrate

    dataset = tmp_path / "data"
    (dataset / "meta").mkdir(parents=True)
    info = dict(
        codebase_version="v2.0",
        total_tasks=40,
        chunks_size=1000,
        splits={"train": "0:40"},
        data_path="episode_{episode_index:06d}.parquet",
    )
    (dataset / "meta/info.json").write_text(json.dumps(info))
    tasks = [dict(task_index=i, task=f"task {i}") for i in range(40)]
    episodes = [dict(episode_index=i, tasks=[f"task {i}"], length=8) for i in range(40)]
    (dataset / "meta/tasks.jsonl").write_text("\n".join(map(json.dumps, tasks)))
    (dataset / "meta/episodes.jsonl").write_text("\n".join(map(json.dumps, episodes)))
    actions = commands([4, 4, 0, 0, 4, 4, 0, 4]).astype(np.float32)
    for index in range(40):
        table = arrow.table(
            dict(
                actions=actions.tolist(),
                task_index=[index] * 8,
                episode_index=[index] * 8,
                frame_index=list(range(8)),
            )
        )
        parquet.write_table(table, dataset / f"episode_{index:06d}.parquet")
    output = tmp_path / "calibration"
    manifest = calibrate(dataset, output, PACEConfig(), episodes_per_task=1)
    assert len(manifest["samples"]) == 40
    read_manifest(output, PACEConfig())
    with pytest.raises(ValueError, match="overwrite"):
        calibrate(dataset, output, PACEConfig(), episodes_per_task=1)


@pytest.mark.parametrize("change", ["missing", "duplicate", "count", "evaluation"])
def test_sample_coverage_and_training_only_enforced(tmp_path, change):
    path = artifact(tmp_path / "artifact")
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if change == "missing":
        manifest["samples"] = manifest["samples"][:-1]
    elif change == "duplicate":
        manifest["samples"][1]["episode_index"] = 0
    elif change == "count":
        manifest["episodes_per_task"] = 2
    else:
        manifest["samples"][0]["split"] = "evaluation"
    manifest["samples_sha256"] = stable_hash(manifest["samples"])
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        read_manifest(path, PACEConfig())
