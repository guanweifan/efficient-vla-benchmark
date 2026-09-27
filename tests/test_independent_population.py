import json

import numpy as np
import pytest

from efficient_vla_benchmark.core.provenance import digest_file, stable_hash
from efficient_vla_benchmark.evaluation.libero.independent import (
    load_population,
    validate_independent_state,
)


def test_independence_ignores_simulator_time_and_checks_all_standard_states():
    standard = np.zeros((50, 4))
    state = np.ones(4)
    validate_independent_state(state, standard)
    for index in range(50):
        values = standard.copy()
        values[index] = state
        values[index, 0] = 123
        with pytest.raises(ValueError, match="overlaps"):
            validate_independent_state(state, values)
    with pytest.raises(ValueError, match="overlaps"):
        validate_independent_state(state, standard, [state])
    with pytest.raises(ValueError, match="geometry"):
        validate_independent_state(state, standard[:49])


def test_population_digest_state_tampering_and_cross_group_overlap(tmp_path):
    state = np.ones(4)
    np.savez_compressed(tmp_path / "state-0000.npz", state=state)
    entry = dict(
        group="quick",
        context=dict(suite="libero_spatial", task_id=0, episode_id=100, seed=1),
        filename="state-0000.npz",
        file_sha256=digest_file(tmp_path / "state-0000.npz"),
        state_sha256=stable_hash(state.tolist()),
        excluded_standard_state_count=50,
    )
    manifest = dict(schema_version=1, kind="independent_search_population", episodes=[entry])

    def save():
        manifest["sha256"] = stable_hash({k: v for k, v in manifest.items() if k != "sha256"})
        (tmp_path / "population.json").write_text(json.dumps(manifest))

    save()
    assert load_population(tmp_path)["episodes"] == [entry]
    duplicate = {
        **entry,
        "group": "confirmation",
        "context": {**entry["context"], "episode_id": 101},
    }
    manifest["episodes"].append(duplicate)
    save()
    with pytest.raises(ValueError, match="physical placement"):
        load_population(tmp_path)
    manifest["episodes"].pop()
    save()
    np.savez_compressed(tmp_path / "state-0000.npz", state=state * 2)
    with pytest.raises(ValueError, match="modified state"):
        load_population(tmp_path)
