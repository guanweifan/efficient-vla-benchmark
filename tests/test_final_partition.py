import copy

import pytest

from efficient_vla_benchmark.evaluation.libero.aggregate import final_partition
from efficient_vla_benchmark.evaluation.libero.manifest import (
    SUITES,
    derive_manifest,
    make_manifest,
)


def example():
    resources = {
        f"{s}/{t}": {
            "task_name": f"task_{t}",
            "bddl": {"path": f"bddl/{s}/{t}", "sha256": "a"},
            "init_states": {"path": f"init/{s}/{t}", "sha256": "b"},
        }
        for s in SUITES
        for t in range(10)
    }
    dev = make_manifest(resources)
    final = derive_manifest(dev, "final")
    chosen = {(r["suite"], r["task_id"], r["episode_id"]) for r in dev["episodes"]}
    rows = [
        dict(r, status="completed", success=(r["suite"], r["task_id"], r["episode_id"]) in chosen)
        for r in final["episodes"]
    ]
    result = {
        "kind": "final_evaluation",
        "manifest_sha256": final["sha256"],
        "complete": True,
        "episodes": 2000,
        "errors": 0,
        "experiment_id": "fixture",
        "episode_results": rows,
        "successes": 200,
        "success_rate": 0.1,
    }
    return dev, result


def test_final_partition_never_reuses_selection_outcomes():
    dev, result = example()
    value = final_partition(result, dev)
    assert value["full_2000"]["successes"] == 200
    assert value["untuned_1800"] == {"episodes": 1800, "successes": 0, "success_rate": 0.0}
    assert value["overlapping_200_in_final_run"]["success_rate"] == 1.0


@pytest.mark.parametrize("change", ["incomplete", "error", "duplicate", "seed", "success", "kind"])
def test_invalid_final_partition(change):
    dev, result = example()
    result = copy.deepcopy(result)
    if change == "incomplete":
        result["complete"] = False
    if change == "error":
        result["errors"] = 1
    if change == "duplicate":
        result["episode_results"][0] = result["episode_results"][1]
    if change == "seed":
        result["episode_results"][0]["seed"] += 1
    if change == "success":
        result["successes"] += 1
    if change == "kind":
        result["kind"] = "development_validation"
    with pytest.raises(ValueError):
        final_partition(result, dev)
