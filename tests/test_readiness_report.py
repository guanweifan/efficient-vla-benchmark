import copy
import importlib.util
from dataclasses import asdict
from pathlib import Path

import pytest

from efficient_vla_benchmark.core.config import ExperimentConfig


@pytest.fixture
def reporter():
    path = Path(__file__).resolve().parents[1] / "scripts/readiness_report.py"
    spec = importlib.util.spec_from_file_location("readiness_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    config = asdict(ExperimentConfig.load(Path("configs/experiments/openvla_oft/fastv.yaml")))
    native = asdict(ExperimentConfig.load(Path("configs/experiments/openvla_oft/baseline.yaml")))
    backend = {"checkpoint_sha256": "checkpoint", "method_details": {}}
    episodes = [
        {"suite": "suite", "task_id": i // 5, "episode_id": i % 5, "seed": 0} for i in range(200)
    ]

    def dev(cfg, exp, successes):
        return {
            "experiment_id": exp,
            "config": cfg,
            "backend": backend,
            "complete": True,
            "episodes": 200,
            "errors": 0,
            "success_rate": successes / 200,
            "successes": successes,
            "manifest_sha256": "manifest",
            "kind": "development_validation",
            "source": {"runtime_source_sha256": "old"},
            "episode_results": [
                dict(e, status="completed", success=i < successes) for i, e in enumerate(episodes)
            ],
        }

    summaries = {
        "summary": {"configurations": [dev(config, "candidate", 180), dev(native, "native", 198)]}
    }
    metadata = {
        "config": config,
        "backend": backend,
        "source": {"runtime_source_sha256": "new"},
        "protocol": {"workload_sha256": "workload"},
    }
    record = {
        "run": {
            "run_id": "method",
            "metadata": metadata,
            "summary": {
                "task_macro": {"mean_ms": 12.5, "repeat_std_ms": 0.1},
                "peak_allocated_bytes": 1024,
            },
        },
        "baseline": {"run_id": "native", "metadata": {"config": native}},
        "task_macro_speedup": 0.8,
    }
    chosen = {
        "config": "configs/experiments/openvla_oft/fastv.yaml",
        "status": "primary",
        "section": "5.1",
        "scope": "core",
        "reason": "Faithful active negative result, not excluded for its outcome.",
        "development_summary_sha256": "summary",
        "development_experiment_id": "candidate",
        "baseline_development_summary_sha256": "summary",
        "baseline_development_experiment_id": "native",
        "timing_record": "method",
        "search": {
            "complete": True,
            "stop": "mechanism_limit",
            "reason": "Fixture exercises a faithful active negative result at its mechanism limit.",
            "limitation": "Fixed instrument with no further admissible compute reduction.",
            "evidence": [
                {
                    "development_summary_sha256": "summary",
                    "development_experiment_id": "candidate",
                    "timing_record": "method",
                }
            ],
        },
    }
    selection = {
        "schema_version": 1,
        "development_manifest_sha256": "manifest",
        "source_compatibility": [
            {"development_runtime": "old", "timing_runtime": "new", "reason": "audited"}
        ],
        "timing_workloads": {"openvla_oft": "workload"},
        "configurations": [chosen],
    }
    quality = {
        "groups": [
            {
                "valid": True,
                "thermal_flags": [],
                "drift_fraction": 0.01,
                "jobs": ["native", "method", "post"],
                "baseline_pre": "native",
                "group": "g",
            }
        ]
    }
    return (
        selection,
        summaries,
        {"method": record},
        quality,
        {"sha256": "manifest", "episodes": episodes},
    )


def test_negative_outcome_is_ready_but_final_is_unmeasured(reporter):
    report = reporter.build(*fixture())
    row = report["primary"][0]
    assert not row["development"]["target_met"]
    assert row["timing"]["speedup"] == 0.8
    assert row["final"] is None
    assert report["final_launch_candidates"][0]["owner_authorization_required"]
    assert not report["final_launched_by_this_command"]
    assert "0.800×" in reporter.render(report)


def test_compact_diagnostic_does_not_become_final_candidate(reporter):
    selection, summaries, records, quality, manifest = fixture()
    selection["timing_protocol"] = "libero_replay4x128_v1"
    selection["configurations"][0]["status"] = "pending_core"
    record = records["method"]
    record["run"]["metadata"]["protocol"]["protocol_id"] = selection["timing_protocol"]
    record["run"]["summary"]["control_workload"] = {"mean_ms": 12.5}
    del record["run"]["summary"]["task_macro"]
    record["workload_speedup"] = record.pop("task_macro_speedup")
    report = reporter.build(selection, summaries, records, quality, manifest)
    assert not report["primary"] and not report["final_launch_candidates"]
    assert report["pending_core"][0]["timing"]["repeat_std_ms"] is None
    assert "0.800×" in reporter.render(report)


@pytest.mark.parametrize(
    "case",
    [
        "duplicate",
        "unmeasured_config",
        "checkpoint",
        "incomplete",
        "source",
        "thermal",
        "cadence",
        "outcomes",
        "dependencies",
        "coverage",
        "section",
    ],
)
def test_invalid_primary_join_is_rejected(reporter, case):
    selection, summaries, records, quality, manifest = copy.deepcopy(fixture())
    if case == "duplicate":
        selection["configurations"] *= 2
    if case == "unmeasured_config":
        records["method"]["run"]["metadata"]["config"]["parameters"] = {"wrong": 1}
    if case == "checkpoint":
        records["method"]["run"]["metadata"]["backend"] = dict(
            checkpoint_sha256="other", method_details={}
        )
    if case == "incomplete":
        summaries["summary"]["configurations"][0]["complete"] = False
    if case == "source":
        selection["source_compatibility"] = []
    if case == "thermal":
        quality["groups"][0]["thermal_flags"] = ["active"]
    if case == "cadence":
        records["method"]["baseline"]["metadata"]["config"] = dict(
            execution_horizon=1, method="none"
        )
    if case == "outcomes":
        summaries["summary"]["configurations"][0]["episode_results"][0]["success"] = False
    if case == "dependencies":
        records["method"]["run"]["metadata"]["source"]["packages"] = {"torch": "different"}
    if case == "coverage":
        selection["coverage_summary_sha256"] = "summary"
    if case == "section":
        del selection["configurations"][0]["section"]
    with pytest.raises(ValueError):
        reporter.build(selection, summaries, records, quality, manifest)


def test_unresolved_core_does_not_silently_enter_final_queue(reporter):
    values = list(fixture())
    values[0]["configurations"][0]["status"] = "pending_core"
    result = reporter.build(*values)
    assert len(result["pending_core"]) == 1
    assert result["primary"] == result["final_launch_candidates"] == []


def test_auxiliary_is_validated_without_entering_final_queue(reporter, monkeypatch):
    selection, summaries, records, quality, manifest = fixture()
    primary = selection["configurations"][0]
    alternative = copy.deepcopy(summaries["summary"]["configurations"][0])
    alternative["config"]["parameters"] = {"auxiliary_fixture": True}
    alternative["experiment_id"] = "auxiliary"
    summaries["summary"]["configurations"].append(alternative)
    records["auxiliary"] = copy.deepcopy(records["method"])
    records["auxiliary"]["run"]["run_id"] = "auxiliary"
    records["auxiliary"]["run"]["metadata"]["config"] = alternative["config"]
    quality["groups"][0]["jobs"].append("auxiliary")
    selection["auxiliary_configurations"] = [
        dict(
            primary,
            config="auxiliary.yaml",
            status="auxiliary",
            development_experiment_id="auxiliary",
            timing_record="auxiliary",
        )
    ]
    selection["auxiliary_configurations"][0].pop("search")
    original = reporter.ExperimentConfig.load
    monkeypatch.setattr(
        reporter.ExperimentConfig,
        "load",
        lambda path: ExperimentConfig(**alternative["config"])
        if path.name == "auxiliary.yaml"
        else original(path),
    )
    result = reporter.build(selection, summaries, records, quality, manifest)
    assert len(result["primary"]) == len(result["auxiliary"]) == 1
    assert len(result["final_launch_candidates"]) == 1
    assert result["final_launch_candidates"][0]["timing_record"] == "method"
    assert "Useful auxiliary configurations" in reporter.render(result)
    selection["auxiliary_configurations"] *= 2
    with pytest.raises(ValueError, match="distinct alternative"):
        reporter.build(selection, summaries, records, quality, manifest)


def test_primary_cannot_be_duplicated_as_auxiliary(reporter):
    selection, summaries, records, quality, manifest = fixture()
    selection["auxiliary_configurations"] = [
        dict(selection["configurations"][0], status="auxiliary")
    ]
    with pytest.raises(ValueError, match="distinct alternative"):
        reporter.build(selection, summaries, records, quality, manifest)


def test_accuracy_eligible_is_not_search_complete(reporter):
    selection, summaries, records, quality, manifest = fixture()
    selection["configurations"][0].pop("search")
    dev = summaries["summary"]["configurations"][0]
    dev["successes"], dev["success_rate"] = 198, 0.99
    for i, row in enumerate(dev["episode_results"]):
        row["success"] = i < 198
    result = reporter.build(selection, summaries, records, quality, manifest)
    assert result["primary"][0]["development"]["target_met"]
    assert not result["primary"][0]["search"]["complete"]
    assert not result["final_launch_candidates"]


@pytest.mark.parametrize("case", ["missing", "target_met", "boundary", "plateau", "limit"])
def test_search_closure_requires_resolvable_evidence(reporter, case):
    selection, summaries, records, quality, manifest = fixture()
    search = selection["configurations"][0]["search"]
    if case == "missing":
        search["evidence"] = []
    if case == "target_met":
        search["stop"] = "target_met"
    if case == "boundary":
        search["stop"] = "accuracy_boundary"
        dev = summaries["summary"]["configurations"][0]
        dev["successes"], dev["success_rate"] = 198, 0.99
        for i, row in enumerate(dev["episode_results"]):
            row["success"] = i < 198
    if case == "plateau":
        search["stop"] = "latency_plateau"
    if case == "limit":
        search.pop("limitation")
    with pytest.raises(ValueError):
        reporter.build(selection, summaries, records, quality, manifest)


def test_faster_eligible_candidate_cannot_be_ignored(reporter):
    selection, summaries, records, quality, manifest = fixture()
    better = copy.deepcopy(summaries["summary"]["configurations"][0])
    better["experiment_id"] = "faster"
    better["config"]["parameters"] = {"keep_ratio": 0.25}
    better["successes"], better["success_rate"] = 198, 0.99
    for i, row in enumerate(better["episode_results"]):
        row["success"] = i < 198
    summaries["summary"]["configurations"].append(better)
    records["faster"] = copy.deepcopy(records["method"])
    records["faster"]["run"]["run_id"] = "faster"
    records["faster"]["run"]["metadata"]["config"] = better["config"]
    records["faster"]["task_macro_speedup"] = 1.3
    quality["groups"][0]["jobs"].append("faster")
    selection["configurations"][0]["search"]["evidence"].append(
        {
            "development_summary_sha256": "summary",
            "development_experiment_id": "faster",
            "timing_record": "faster",
        }
    )
    with pytest.raises(ValueError, match="faster eligible"):
        reporter.build(selection, summaries, records, quality, manifest)


@pytest.mark.parametrize("case", ["workload", "reference", "source", "dependencies"])
def test_search_alternatives_need_matched_timing_provenance(reporter, case):
    selection, summaries, records, quality, manifest = fixture()
    alternative = copy.deepcopy(summaries["summary"]["configurations"][0])
    alternative["experiment_id"] = "alternative"
    alternative["config"]["parameters"] = {"keep_ratio": 0.25}
    summaries["summary"]["configurations"].append(alternative)
    timing = copy.deepcopy(records["method"])
    timing["run"]["run_id"] = "alternative"
    timing["run"]["metadata"]["config"] = alternative["config"]
    records["alternative"] = timing
    quality["groups"][0]["jobs"].append("alternative")
    selection["configurations"][0]["search"]["evidence"].append(
        {
            "development_summary_sha256": "summary",
            "development_experiment_id": "alternative",
            "timing_record": "alternative",
        }
    )
    if case == "workload":
        timing["run"]["metadata"]["protocol"]["workload_sha256"] = "other"
    if case == "reference":
        timing["baseline"]["run_id"] = "unmatched"
    if case == "source":
        timing["run"]["metadata"]["source"]["runtime_source_sha256"] = "unaudited"
    if case == "dependencies":
        timing["run"]["metadata"]["source"]["packages"] = {"torch": "other"}
    with pytest.raises(ValueError, match="Search"):
        reporter.build(selection, summaries, records, quality, manifest)
