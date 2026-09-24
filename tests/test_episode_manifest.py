import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from efficient_vla_benchmark.core.provenance import stable_hash
from efficient_vla_benchmark.evaluation.libero.aggregate import merge_shards
from efficient_vla_benchmark.evaluation.libero.manifest import (
    SUITES,
    episode_key,
    load_manifest,
    make_manifest,
    select_shard,
    validate_manifest,
)


@pytest.fixture
def manifest():
    resources = {
        f"{s}/{t}": {
            "task_name": f"task_{t}",
            "bddl": {"path": f"bddl/{s}/{t}", "sha256": "abc"},
            "init_states": {"path": f"init/{s}/{t}", "sha256": "def"},
        }
        for s in SUITES
        for t in range(10)
    }
    return make_manifest(resources)


def test_deterministic_stratified_nonprefix_selection(manifest, tmp_path):
    assert make_manifest(manifest["resources"]) == manifest
    assert make_manifest(manifest["resources"], sample_seed=42) != manifest
    assert len({episode_key(e) for e in manifest["episodes"]}) == 200
    assert any(e["episode_id"] > 4 for e in manifest["episodes"])
    path = tmp_path / "episodes.json"
    path.write_text(json.dumps(manifest))
    assert load_manifest(path) == manifest


@pytest.mark.parametrize("count", [1, 2, 4, 8, 13, 40, 50, 200])
def test_shards_are_disjoint_exact_cover(manifest, count):
    rows = [e for i in range(count) for e in select_shard(manifest, i, count)]
    assert len(rows) == 200
    assert {episode_key(e) for e in rows} == {episode_key(e) for e in manifest["episodes"]}
    assert all(e["seed"] == 0 for e in rows)


@pytest.mark.parametrize("index,count", [(0, 0), (4, 4), (-1, 4), (0, 201), (True, 4)])
def test_invalid_shard(manifest, index, count):
    with pytest.raises(ValueError):
        select_shard(manifest, index, count)


def test_manifest_tamper_and_duplicate_rejected(manifest):
    modified = copy.deepcopy(manifest)
    modified["episodes"][0] = modified["episodes"][1]
    with pytest.raises(ValueError, match="digest"):
        validate_manifest(modified)
    modified["sha256"] = stable_hash({k: v for k, v in modified.items() if k != "sha256"})
    with pytest.raises(ValueError, match="distinct"):
        validate_manifest(modified)


def write_shard(root, manifest, index, *, error=False, identity="same"):
    root.mkdir()
    episodes = select_shard(manifest, index, 4)
    meta = {
        "kind": "development_validation",
        "experiment_id": identity,
        "episode_manifest": {"sha256": manifest["sha256"]},
        "config": {},
        "backend": {},
        "source": {"packages": {}, "cuda_extension_sha256": {}},
        "shard": {"index": index, "count": 4, "episodes": episodes},
    }
    (root / "metadata.json").write_text(json.dumps(meta))
    (root / "episodes.jsonl").write_text(
        "\n".join(
            json.dumps(
                {**e, "status": "error" if error and i == 0 else "completed", "success": not error}
            )
            for i, e in enumerate(episodes)
        )
    )
    return root


def test_merge_exact_200_and_incomplete_not_success_rate(manifest, tmp_path):
    paths = [write_shard(tmp_path / str(i), manifest, i) for i in range(4)]
    complete = merge_shards(manifest, paths)
    assert complete["complete"] and complete["success_rate"] == 1
    assert complete["formal_success_rate"] is None
    partial = merge_shards(manifest, paths[:1])
    assert not partial["complete"] and partial["success_rate"] is None
    assert len(partial["missing"]) == 150
    with pytest.raises(ValueError, match="Duplicate"):
        merge_shards(manifest, paths + paths[:1])


def test_merge_errors_and_configuration_mismatch(manifest, tmp_path):
    paths = [write_shard(tmp_path / str(i), manifest, i, error=i == 1) for i in range(4)]
    result = merge_shards(manifest, paths)
    assert result["errors"] == 1 and result["success_rate"] is None
    incompatible = write_shard(tmp_path / "wrong", manifest, 1, identity="different")
    with pytest.raises(ValueError, match="differ"):
        merge_shards(manifest, [paths[0], incompatible])


def test_current_campaign_matrix_is_executable(manifest):
    spec = importlib.util.spec_from_file_location(
        "evaluation_results", "scripts/evaluation_results.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    matrix = yaml.safe_load(Path("configs/evaluation/dev200_campaign.yaml").read_text())
    complete = module.plan(matrix, manifest, "example", 8)
    baseline = module.plan(matrix, manifest, "example", 8, True)
    assert len(complete["jobs"]) == 43 * 8
    assert len(baseline["jobs"]) == 4 * 8
    assert len({j["id"] for j in complete["jobs"]}) == len(complete["jobs"])
