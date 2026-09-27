"""Synthetic evidence fixtures test audit rejection, not algorithm success."""

import json

import numpy as np
import pytest

from efficient_vla_benchmark.core.provenance import digest_file, stable_hash
from efficient_vla_benchmark.evaluation.libero.runner import SUITES

from ..artifact import load_artifact
from ..config import EVOConfig
from ..freeze import freeze_search


def make_fixture(root):
    population_root = root / "population"
    population_root.mkdir()
    entries = []
    for group, count in (("reference", 40), ("quick", 20), ("confirmation", 40)):
        for flat in range(count):
            suite_index, task = divmod(flat, count // 4)
            state = np.array([0.0, len(entries) + 1.0])
            filename = f"state-{len(entries):03d}.npz"
            np.savez_compressed(population_root / filename, state=state)
            entries.append(
                dict(
                    group=group,
                    context=dict(
                        suite=SUITES[suite_index],
                        task_id=task,
                        episode_id=100 + len(entries),
                        seed=1000 + len(entries),
                    ),
                    filename=filename,
                    file_sha256=digest_file(population_root / filename),
                    state_sha256=stable_hash(state.tolist()),
                    excluded_standard_state_count=50,
                )
            )
    population = dict(schema_version=1, kind="independent_search_population", episodes=entries)
    population["sha256"] = stable_hash(population)
    (population_root / "population.json").write_text(json.dumps(population))
    binding = dict(
        num_blocks=2,
        num_steps=4,
        precision="float32",
        sampler_parameters={},
        checkpoint_sha256="fixture",
        population_sha256=population["sha256"],
    )
    descriptor = dict(binding=binding, budget=6, layout=[[0, 3], [0, 1, 2, 3]])
    for name, group, candidate in (
        ("native", "reference", None),
        ("c0-quick", "quick", descriptor),
        ("c0-confirm", "confirmation", descriptor),
    ):
        run = root / "runs" / name
        run.mkdir(parents=True)
        selected = [e for e in entries if e["group"] == group]
        metadata = dict(
            kind="independent_search_rollout",
            population_sha256=population["sha256"],
            group=group,
            episodes=selected,
            source=dict(runtime_source_sha256="fixture"),
            policy=dict(
                binding=binding,
                execution_horizon=8,
                candidate_sha256=stable_hash(candidate) if candidate else None,
            ),
        )
        (run / "metadata.json").write_text(json.dumps(metadata))
        (run / "binding.json").write_text(json.dumps(binding))
        records = [
            dict(
                **e["context"],
                status="completed",
                success=True,
                initial_state_sha256=e["state_sha256"],
            )
            for e in selected
        ]
        (run / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        (run / "summary.json").write_text(
            json.dumps(
                dict(episodes=len(records), successes=len(records), errors=0, wall_seconds=1)
            )
        )
        if candidate is None:
            np.savez_compressed(run / "prior.npz", dissimilarity=np.ones((2, 4)))
    report = dict(
        limits=dict(max_episodes=640, max_candidates=24),
        reference_run="native",
        candidates=[
            dict(id="c0", **descriptor, quick_run="c0-quick", confirmation_run="c0-confirm")
        ],
        selected={"6": "c0"},
        reservations=[
            dict(id="native", kind="reference", episodes=40),
            dict(id="c0-quick", kind="candidate", episodes=20),
            dict(id="c0-confirm", kind="confirmation", episodes=40),
        ],
    )
    (root / "search.json").write_text(json.dumps(report))
    return population_root


def test_freeze_checks_outcomes_and_selectable_artifact(tmp_path):
    population = make_fixture(tmp_path)
    artifact = tmp_path / "artifact"
    manifest = freeze_search(tmp_path, population, artifact)
    assert manifest["search_episodes"] == 100
    assert manifest["layouts"]["6"]["target_confirmed"] is True
    assert load_artifact(artifact, EVOConfig(refresh_budget=6))["sha256"] == manifest["sha256"]
    with pytest.raises(ValueError, match="no frozen"):
        load_artifact(artifact, EVOConfig(refresh_budget=7))
    (artifact / "search.json").write_text("{}")
    with pytest.raises(ValueError, match="modified"):
        load_artifact(artifact, EVOConfig(refresh_budget=6))


@pytest.mark.parametrize("change", ["outcome", "source", "budget", "group"])
def test_freeze_rejects_tampering_and_mixed_source(tmp_path, change):
    population = make_fixture(tmp_path)
    run = tmp_path / "runs" / "c0-quick"
    if change == "outcome":
        (run / "episodes.jsonl").write_text("")
    elif change == "budget":
        path = tmp_path / "search.json"
        report = json.loads(path.read_text())
        report["reservations"].pop()
        path.write_text(json.dumps(report))
    else:
        path = run / "metadata.json"
        metadata = json.loads(path.read_text())
        if change == "source":
            metadata["source"]["runtime_source_sha256"] = "changed"
        else:
            metadata["group"] = "confirmation"
        path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        freeze_search(tmp_path, population, tmp_path / "artifact")
