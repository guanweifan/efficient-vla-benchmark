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
        "config": {"execution_horizon": 2},
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
