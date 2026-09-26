import json

import numpy as np
import pytest

from efficient_vla_benchmark.core.provenance import stable_hash

from ..freeze import freeze


def test_freeze_requires_complete_disjoint_successful_population(tmp_path):
    source = tmp_path / "collection"
    source.mkdir()
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    rows, keys, values, partitions = [], [], [], []
    standards = {(s, t): np.zeros((50, 3)) for s in suites for t in range(10)}
    for replicate in range(2):
        for flat in range(40):
            index, task = divmod(flat, 10)
            state = np.array([0, flat + 100, replicate + 1], dtype=float)
            success = flat != 0 or replicate != 0
            rows.append(
                dict(
                    suite=suites[index],
                    task_id=task,
                    episode_id=1000 + replicate,
                    seed=910000 + replicate * 1000 + flat,
                    status="completed",
                    success=success,
                    initial_state_sha256=stable_hash(state.tolist()),
                    excluded_standard_state_count=50,
                    pending_pairs=1,
                    retained_pairs=int(success),
                )
            )
            np.savez(source / f"initial-{flat}-{replicate}.npz", state=state)
            if success:
                key = np.zeros(500, dtype=np.float32)
                key[0] = 1
                keys.append(key)
                values.append(np.zeros((10, 16, 7), dtype=np.float32))
                partitions.append(index)
    metadata = dict(
        kind="independent_native_action_cache_population",
        backend={"checkpoint_sha256": "fixture"},
        sampler_parameters={"num_ddim_steps": 10},
        source={},
    )
    (source / "metadata.json").write_text(json.dumps(metadata))
    (source / "summary.json").write_text(
        json.dumps(dict(episodes=80, successes=79, pairs=79, wall_seconds=12))
    )
    (source / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    np.savez(source / "bank.npz", keys=keys, states=values, suites=partitions)
    result = freeze([source], tmp_path / "bank", standards)
    assert result["entries"] == result["successful_episodes"] == 79
    assert result["population_episodes"] == 80
    assert result["episodes_disjoint_from_evaluation"]
    with pytest.raises(FileExistsError):
        freeze([source], tmp_path / "bank", standards)
    standards[(suites[0], 0)][0] = [0, 100, 1]
    with pytest.raises(ValueError, match="separation"):
        freeze([source], tmp_path / "overlap", standards)
    assert not (tmp_path / "overlap").exists()
