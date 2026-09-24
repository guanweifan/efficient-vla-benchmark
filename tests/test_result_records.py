import copy
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def recorder():
    script = Path(__file__).resolve().parents[1] / "scripts/record_result.py"
    spec = importlib.util.spec_from_file_location("record_result", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_run(path):
    path.mkdir()
    meta = {
        "backend": {"backend": "fixture", "model_variant": "fixture", "checkpoint_sha256": "abc"},
        "config": {"execution_horizon": 2, "method": "example"},
        "traces": [{"sha256": "xyz"}],
        "protocol": {"mode": "control", "repeats": 1, "warmup_queries": 1},
        "hardware": {"runtime": "fixture"},
        "source": {"commit": "abc", "dirty": False},
    }
    summary = {
        "all_control_steps": {"count": 2, "mean_ms": 1.5},
        "all_queries": {"count": 1, "mean_ms": 2.0, "p95_ms": 2.0},
        "replay_policy_time_ms": 3.0,
        "peak_allocated_bytes": None,
    }
    (path / "metadata.json").write_text(json.dumps(meta))
    (path / "summary.json").write_text(json.dumps(summary))
    (path / "timings.jsonl").write_text(
        json.dumps({"queried": True, "latency_ms": 2.0})
        + "\n"
        + json.dumps({"queried": False, "latency_ms": 1.0})
        + "\n"
    )


def test_evidence_export_and_render_without_private_locations(tmp_path, recorder):
    run = tmp_path / "example-run"
    write_run(run)
    record = recorder.make_record(run, run, "5.3 Routing", "Example", "configs/example.yaml")
    assert record["replay_speedup"] == 1
    assert str(tmp_path) not in json.dumps(record)
    assert record["formal_success_rate"] is None
    assert "timings.jsonl" in record["run"]["files_sha256"]
    table = recorder.render([record])
    assert "5.3 Routing" in table and "1.000×" in table and "Not run" in table


def test_rejects_inconsistent_raw_data(tmp_path, recorder):
    run = tmp_path / "example-run"
    write_run(run)
    summary = json.loads((run / "summary.json").read_text())
    summary["replay_policy_time_ms"] = 1.0
    (run / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="Raw latency sum"):
        recorder.timing_evidence(run)


def test_rejects_incomparable_baseline(tmp_path, recorder):
    run, baseline = tmp_path / "example-run", tmp_path / "baseline"
    write_run(run)
    write_run(baseline)
    meta = json.loads((baseline / "metadata.json").read_text())
    meta["traces"] = []
    (baseline / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="baseline trace"):
        recorder.make_record(run, baseline, "5.3 Routing", "Example", "configs/example.yaml")


def test_token_diagnostics_only_count_queries(recorder):
    rows = [
        {"queried": True, "diagnostics": {"visual_tokens_final": 512, "keyframe": True}},
        {"queried": False, "diagnostics": {"visual_tokens_final": 512, "keyframe": True}},
        {"queried": True, "diagnostics": {"visual_tokens_final": 128, "keyframe": False}},
    ]
    summary = recorder.token_evidence(rows)
    assert summary["queries_with_diagnostics"] == 2
    assert summary["final_visual_tokens"] == {"mean": 320, "min": 128, "max": 512}
    assert summary["keyframe"] == {"true": 1, "observed": 2}
    assert recorder.token_evidence([{"queried": True}]) is None
    with pytest.raises(ValueError, match="visual-token"):
        recorder.token_evidence([{"queried": True, "diagnostics": {"visual_tokens_final": -1}}])


def test_cache_query_skipping_is_not_reported_as_context_pruning(recorder):
    rows = [
        {
            "queried": True,
            "diagnostics": {"visual_query_tokens_final": 300, "visual_kv_tokens": 512},
        }
    ]
    assert recorder.token_evidence(rows) is None


def development_fixture():
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    results, records, selection, ids = [], [], {}, {}
    for preset, successes in (("baseline", 180), ("example", 160)):
        config = {
            "backend": "fixture",
            "model_variant": "joint",
            "precision": "bf16",
            "method": "none" if preset == "baseline" else "example",
            "execution_horizon": 8,
            "parameters": {},
            "backend_parameters": {},
        }
        backend = {**config, "checkpoint_sha256": "checkpoint"}
        result = {
            "backend": backend,
            "config": config,
            "preset": preset,
            "complete": True,
            "episodes": 200,
            "errors": 0,
            "successes": successes,
            "success_rate": successes / 200,
            "manifest_sha256": "manifest",
            "experiment_id": preset,
            "source": {"runtime_source_sha256": "source"},
            "shards": [{}] * 8,
            "suites": {
                s: {"episodes": 50, "errors": 0, "successes": successes // 4} for s in suites
            },
        }
        records.append(
            {
                "run": {
                    "run_id": preset,
                    "metadata": {"backend": backend, "config": config},
                    "summary": {
                        "peak_allocated_bytes": 2**30,
                        "all_control_steps": {"mean_ms": 10},
                    },
                },
                "preset": f"configs/experiments/fixture/{preset}.yaml",
                "section": "5.1",
                "replay_speedup": 1.0,
            }
        )
        results.append(result)
        selection[f"fixture/{preset}"] = preset
        ids[preset] = preset
    return (
        {
            "kind": "development_validation",
            "all_complete": True,
            "manifest_sha256": "manifest",
            "configurations": results,
        },
        copy.deepcopy(records),
        selection,
        ids,
    )


def test_development_join_is_separate_from_formal_and_historical_timing(recorder):
    args = development_fixture()
    text = "\n".join(recorder.validation_tables(*args))
    assert "160/200 (80.0%)" in text and "-10.0" in text
    assert "not simultaneous" in text and "development/tuning subset" in text


@pytest.mark.parametrize(
    "mutation",
    [
        "incomplete",
        "errors",
        "different_config",
        "different_checkpoint",
        "superseded",
        "missing_timing",
    ],
)
def test_development_join_rejects_incomparable_evidence(recorder, mutation):
    validation, records, selection, ids = development_fixture()
    if mutation == "incomplete":
        validation["all_complete"] = False
    elif mutation == "errors":
        validation["configurations"][1]["errors"] = 1
    elif mutation == "different_config":
        records[1]["run"]["metadata"]["config"]["parameters"] = {"ratio": 0.5}
    elif mutation == "different_checkpoint":
        records[1]["run"]["metadata"]["backend"]["checkpoint_sha256"] = "another"
    elif mutation == "superseded":
        records[0]["supersedes"] = ["example"]
    else:
        selection.pop("fixture/example")
    with pytest.raises(ValueError):
        recorder.validation_tables(validation, records, selection, ids)


def test_superseded_evidence_is_preserved_and_labelled(tmp_path, recorder):
    import copy

    run = tmp_path / "example-run"
    write_run(run)
    old = recorder.make_record(run, run, "5.1 Tokens", "Old", "configs/example.yaml")
    old_path = tmp_path / "example-run.json"
    old_path.write_text(json.dumps(old))
    original_bytes = old_path.read_bytes()
    new = copy.deepcopy(old)
    new["run"]["run_id"], new["label"] = "corrected-run", "Corrected"
    recorder.annotate_supersedes(new, ["example-run"], tmp_path)
    assert old_path.read_bytes() == original_bytes
    assert new["supersedes"] == ["example-run"]
    table = recorder.render([old, new])
    assert "Superseded — Old" in table and "`example-run`" in table
    assert "Corrected" in table and "`corrected-run`" in table


@pytest.mark.parametrize("run_id", ["../example-run", "example-run", "not-found"])
def test_reject_bad_superseded_reference(tmp_path, recorder, run_id):
    run = tmp_path / "example-run"
    write_run(run)
    record = recorder.make_record(run, run, "5.1 Tokens", "Old", "configs/example.yaml")
    with pytest.raises(ValueError):
        recorder.annotate_supersedes(record, [run_id], tmp_path)


def test_cannot_supersede_another_method(tmp_path, recorder):
    import copy

    run = tmp_path / "example-run"
    write_run(run)
    old = recorder.make_record(run, run, "5.1 Tokens", "Old", "configs/example.yaml")
    (tmp_path / "example-run.json").write_text(json.dumps(old))
    new = copy.deepcopy(old)
    new["run"]["run_id"] = "new-run"
    new["run"]["metadata"]["config"]["method"] = "another_method"
    with pytest.raises(ValueError, match="another method"):
        recorder.annotate_supersedes(new, ["example-run"], tmp_path)


def test_cache_summary_keeps_all_context_and_skips_queued_rows(recorder):
    rows = [
        {
            "queried": True,
            "diagnostics": {"visual_query_tokens_final": 256, "visual_kv_tokens": 256},
        },
        {
            "queried": False,
            "diagnostics": {"visual_query_tokens_final": 256, "visual_kv_tokens": 256},
        },
        {
            "queried": True,
            "diagnostics": {"visual_query_tokens_final": 128, "visual_kv_tokens": 256},
        },
    ]
    evidence = recorder.cache_evidence(rows)
    assert evidence["queries_with_diagnostics"] == 2
    assert evidence["computed_visual_queries"] == {"mean": 192, "min": 128, "max": 256}
    assert evidence["retained_visual_kv"] == {"mean": 256, "min": 256, "max": 256}
    assert recorder.cache_evidence([{"queried": True}]) is None
    assert recorder.token_evidence(rows) is None
    with pytest.raises(ValueError, match="cached-query"):
        recorder.cache_evidence(
            [
                {
                    "queried": True,
                    "diagnostics": {
                        "visual_query_tokens_final": 512,
                        "visual_kv_tokens": 256,
                    },
                }
            ]
        )


def test_summary_uses_local_record_digest_without_private_file_links(tmp_path, recorder):
    import hashlib

    run = tmp_path / "example-run"
    write_run(run)
    record = recorder.make_record(run, run, "5.1 Tokens", "Example", "configs/example.yaml")
    table = recorder.render([record])
    payload = json.dumps(record, indent=2, sort_keys=True) + "\n"
    assert hashlib.sha256(payload.encode()).hexdigest() in table
    assert "(results/" not in table and f"{run.name}.json)" not in table
    assert "(../scripts/history_commit_map.json)" in table
    assert "[R01](#r01)" in table and '<a id="r01"></a>R01' in table
    assert "`example-run`" in table
    assert "[example](../configs/example.yaml)" in table


def test_render_is_deterministic_and_preserves_slower_variants(tmp_path, recorder):
    import copy

    run = tmp_path / "example-run"
    write_run(run)
    first = recorder.make_record(run, run, "6.1 Quantization", "Faster", "configs/a.yaml")
    second = copy.deepcopy(first)
    second["run"]["run_id"] = "slow-run"
    second["label"], second["preset"], second["replay_speedup"] = "Slower", "configs/b.yaml", 0.8
    assert recorder.render([first, second]) == recorder.render([second, first])
    table = recorder.render([first, second])
    assert "Slower" in table and "0.800×" in table
    assert table.count("\n| C01 |") == 1


def test_record_commands_default_to_ignored_outputs(tmp_path):
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "scripts/record_result.py"
    write_run(tmp_path / "run")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "export",
            "--run-dir",
            "run",
            "--baseline-dir",
            "run",
            "--section",
            "5.1 Tokens",
            "--label",
            "Example",
            "--preset",
            "configs/example.yaml",
        ],
        cwd=tmp_path,
        check=True,
    )
    assert (tmp_path / "outputs/result_records/run.json").is_file()
    subprocess.run(
        [sys.executable, str(script), "render", "--output", "summary.md"],
        cwd=tmp_path,
        check=True,
    )
    assert "Example" in (tmp_path / "summary.md").read_text()
    assert not (tmp_path / "docs/results").exists()


def test_history_mapping_preserves_original_evidence(tmp_path, recorder):
    import copy

    run = tmp_path / "example-run"
    write_run(run)
    record = recorder.make_record(run, run, "5.1 Tokens", "Example", "configs/example.yaml")
    old, new = "a" * 40, "b" * 40
    record["run"]["metadata"]["source"] = {"commit": old, "dirty": True}
    before = copy.deepcopy(record)
    digest = recorder.record_digest(record)
    table = recorder.render([record], commit_map={old: new})
    assert f"`{old}` (dirty) → `{new}`" in table
    assert digest in table
    assert record == before
    assert "not a new measurement" in table
    assert recorder.revision_text({"commit": new, "dirty": False}, {old: new}) == f"`{new}`"
    assert recorder.revision_text({"commit": None, "dirty": True}, {}) == "`unknown` (dirty)"


@pytest.mark.parametrize("mapping", [[], {"bad": "b" * 40}, {"a" * 40: "0" * 40}])
def test_history_map_rejects_invalid_revisions(tmp_path, recorder, mapping):
    path = tmp_path / "map.json"
    path.write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="history commit map"):
        recorder.load_history_map(path)


def test_tracked_history_map_contains_full_revisions(recorder):
    mapping = recorder.load_history_map()
    assert len(mapping) == 81


def test_diagnostic_rows_identify_backend_and_measurement_record(tmp_path, recorder):
    import copy

    run = tmp_path / "example-run"
    write_run(run)
    first = recorder.make_record(run, run, "5.1 Tokens", "Shared name", "configs/a.yaml")
    first["run"]["token_diagnostics"] = {
        "final_visual_tokens": {"mean": 128.0, "min": 128, "max": 128},
    }
    first["run"]["cache_diagnostics"] = {
        "computed_visual_queries": {"mean": 64.0, "min": 64, "max": 64},
        "retained_visual_kv": {"mean": 256.0, "min": 256, "max": 256},
    }
    second = copy.deepcopy(first)
    second["run"]["run_id"] = "second-run"
    second["run"]["metadata"]["backend"]["backend"] = "second_backend"
    text = recorder.render([first, second])
    for marker in (
        "## Observed visual-token retention",
        "## Computed visual queries and retained KV context",
    ):
        table = text.split(marker, 1)[1].split("\n## ", 1)[0]
        assert "[R01](#r01) Shared name | fixture" in table
        assert "[R02](#r02) Shared name | second_backend" in table


def test_auxiliary_evidence_never_omits_discarded_dense_work(tmp_path, recorder):
    run = tmp_path / "aux-run"
    write_run(run)
    diag = {
        "auxiliary_queries": 1,
        "primary_forward_ms": 0.8,
        "auxiliary_forward_ms": 0.9,
        "auxiliary_join_wait_ms": 0.85,
        "auxiliary_visual_tokens_final": 256,
    }
    rows = [
        {"queried": True, "latency_ms": 2.0, "diagnostics": diag},
        {"queried": False, "latency_ms": 1.0, "diagnostics": diag},
    ]
    (run / "timings.jsonl").write_text("\n".join(map(json.dumps, rows)))
    summary = json.loads((run / "summary.json").read_text())
    with pytest.raises(ValueError, match="auxiliary call count"):
        recorder.timing_evidence(run)
    summary.update(auxiliary_model_calls=1, total_native_model_calls=2)
    (run / "summary.json").write_text(json.dumps(summary))
    record = recorder.make_record(run, run, "5.1 Tokens", "Aux", "configs/aux.yaml")
    assert record["run"]["auxiliary_diagnostics"]["auxiliary_calls"] == 1
    assert record["replay_speedup"] == 1
    table = recorder.render([record])
    assert "Auxiliary dense-refresh work" in table
    assert "ALL auxiliary work" in table and "1 / 1" in table
    assert recorder.auxiliary_evidence([{"queried": True}]) is None


@pytest.mark.parametrize(
    "key,value",
    [
        ("auxiliary_queries", True),
        ("auxiliary_queries", 2),
        ("primary_forward_ms", float("nan")),
        ("auxiliary_forward_ms", -1),
        ("auxiliary_visual_tokens_final", 0),
    ],
)
def test_auxiliary_rejects_invalid_diagnostics(recorder, key, value):
    diag = {
        "auxiliary_queries": 1,
        "primary_forward_ms": 1.0,
        "auxiliary_forward_ms": 2.0,
        "auxiliary_join_wait_ms": 0.0,
        "auxiliary_visual_tokens_final": 256,
    }
    diag[key] = value
    with pytest.raises(ValueError, match="auxiliary"):
        recorder.auxiliary_evidence([{"queried": True, "diagnostics": diag}])
