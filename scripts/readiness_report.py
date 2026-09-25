"""Render only explicitly selected, evidence-complete primary configurations."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from efficient_vla_benchmark.core.config import ExperimentConfig
from efficient_vla_benchmark.core.provenance import digest_file, stable_hash
from efficient_vla_benchmark.evaluation.libero.aggregate import final_partition
from efficient_vla_benchmark.evaluation.libero.manifest import episode_key, load_manifest

POLICY_KEYS = (
    "backend",
    "model_variant",
    "precision",
    "execution_horizon",
    "parameters",
    "backend_parameters",
    "method",
)


def policy_identity(config):
    return {key: config.get(key) for key in POLICY_KEYS}


def check_development(result, manifest):
    if (
        result.get("kind") != "development_validation"
        or not result.get("complete")
        or result.get("episodes") != 200
        or result.get("errors") != 0
        or result.get("manifest_sha256") != manifest["sha256"]
        or result.get("success_rate") != result.get("successes", -1) / 200
    ):
        raise ValueError("Primary selection requires complete matching Dev-200 evidence")
    expected = {episode_key(e): e for e in manifest["episodes"]}
    rows = result["episode_results"]
    if (
        len(rows) != 200
        or {episode_key(r) for r in rows} != set(expected)
        or any(
            r["seed"] != expected[episode_key(r)]["seed"]
            or r["status"] != "completed"
            or type(r["success"]) is not bool
            for r in rows
        )
        or sum(r["success"] for r in rows) != result["successes"]
    ):
        raise ValueError("Development summary disagrees with its manifest or outcomes")


def scope_key(config):
    return (
        config["backend"],
        config["method"],
        config["execution_horizon"] if config["method"] == "none" else None,
    )


def environment_identity(source, backend):
    return {
        "packages": {
            name: value
            for name, value in source.get("packages", {}).items()
            if not name.startswith("efficient-vla-backend-")
            or name == "efficient-vla-backend-" + backend.replace("_", "-")
        },
        "extensions": source.get("cuda_extension_sha256", {}),
        "submodules": source.get("submodules", {}),
    }


def build(selection, summaries, records, quality, development_manifest, final_results=None):
    if selection.get("schema_version") != 1:
        raise ValueError("Unsupported selection schema")
    if selection["development_manifest_sha256"] != development_manifest["sha256"]:
        raise ValueError("Selection uses a different development manifest")
    manifest = development_manifest["sha256"]
    audited = {
        (r["development_runtime"], r["timing_runtime"])
        for r in selection["source_compatibility"]
        if r.get("reason")
    }
    quality_by_run = {}
    for group in quality["groups"]:
        if not group["valid"] or group["thermal_flags"] or group["drift_fraction"] > 0.03:
            continue
        for run_id in group["jobs"]:
            if run_id in quality_by_run:
                raise ValueError("Timing run belongs to multiple cohorts")
            quality_by_run[run_id] = group
    seen, rows, pending, launch = set(), [], [], []
    for selected in selection["configurations"]:
        config = asdict(ExperimentConfig.load(Path(selected["config"])))
        key = scope_key(config)
        if key in seen:
            raise ValueError("Multiple primary configurations for one backend/method scope")
        seen.add(key)
        if not all(selected.get(k) for k in ("reason", "scope", "section")):
            raise ValueError("Every selection needs a reason, core scope and section")
        if selected["status"] == "pending_core" and not selected.get("timing_record"):
            pending.append(
                {
                    "backend": config["backend"],
                    "method": config["method"],
                    "config": selected["config"],
                    "section": selected["section"],
                    "reason": selected["reason"],
                }
            )
            continue
        if selected["status"] not in {"primary", "pending_core"}:
            raise ValueError("Unknown readiness status")
        summary = summaries[selected["development_summary_sha256"]]
        devs = [
            r
            for r in summary["configurations"]
            if r["experiment_id"] == selected["development_experiment_id"]
            and policy_identity(r["config"]) == policy_identity(config)
        ]
        if len(devs) != 1:
            raise ValueError("Development selection is missing or ambiguous")
        dev = devs[0]
        check_development(dev, development_manifest)
        record = records[selected["timing_record"]]
        timing = record["run"]["metadata"]
        baseline = record["baseline"]["metadata"]
        if policy_identity(timing["config"]) != policy_identity(config):
            raise ValueError("Preset differs from measured timing configuration")
        if timing["backend"]["checkpoint_sha256"] != dev["backend"]["checkpoint_sha256"]:
            raise ValueError("Timing and development checkpoints differ")
        if timing["backend"] != dev["backend"]:
            raise ValueError("Timing and development backend/method metadata differ")
        if environment_identity(timing["source"], config["backend"]) != environment_identity(
            dev["source"], config["backend"]
        ):
            raise ValueError("Timing and development runtime dependencies differ")
        source_pair = (
            dev["source"]["runtime_source_sha256"],
            timing["source"]["runtime_source_sha256"],
        )
        if source_pair[0] != source_pair[1] and source_pair not in audited:
            raise ValueError("Unaudited development/timing runtime change")
        if (
            timing["protocol"].get("workload_sha256")
            != selection["timing_workloads"][config["backend"]]
        ):
            raise ValueError("Selected record does not use the frozen timing workload")
        if selection.get("timing_protocol") and (
            timing["protocol"].get("protocol_id") != selection["timing_protocol"]
        ):
            raise ValueError("Selected record uses a different timing protocol")
        cohort = quality_by_run.get(record["run"]["run_id"])
        if cohort is None or record["baseline"]["run_id"] != cohort["baseline_pre"]:
            raise ValueError("Missing matched pre/post timing quality evidence")
        if (
            baseline["config"]["method"] != "none"
            or baseline["config"]["execution_horizon"] != config["execution_horizon"]
        ):
            raise ValueError("Comparison needs a native baseline at matched cadence")
        refs = [
            r
            for r in summaries[selected["baseline_development_summary_sha256"]]["configurations"]
            if r["experiment_id"] == selected["baseline_development_experiment_id"]
            and r["config"]["method"] == "none"
            and r["config"]["backend"] == config["backend"]
        ]
        if not refs:
            raise ValueError("Missing development baseline")
        ref = refs[0]
        check_development(ref, development_manifest)
        if (
            ref["config"]["execution_horizon"] != config["execution_horizon"]
            or ref["backend"]["checkpoint_sha256"] != dev["backend"]["checkpoint_sha256"]
        ):
            raise ValueError("Development baseline cadence/checkpoint differs")
        if any(r["episode_results"] != ref["episode_results"] for r in refs):
            raise ValueError("Conflicting repeated native evidence")
        final = None
        if final_results:
            matches = [
                r for r in final_results if policy_identity(r["config"]) == policy_identity(config)
            ]
            if len(matches) > 1:
                raise ValueError("Cannot choose among repeated final campaigns")
            if matches:
                measured = matches[0]
                if measured["backend"]["checkpoint_sha256"] != dev["backend"]["checkpoint_sha256"]:
                    raise ValueError("Final checkpoint differs from selection")
                if (
                    measured["source"]["runtime_source_sha256"]
                    != selection["final_runtime_source_sha256"]
                ):
                    raise ValueError("Final runtime differs from the frozen final selection")
                final = final_partition(measured, development_manifest)
        compact = "control_workload" in record["run"]["summary"]
        latency = record["run"]["summary"]["control_workload" if compact else "task_macro"]
        delta = 100 * (dev["success_rate"] - ref["success_rate"])
        row = {
            "backend": config["backend"],
            "method": config["method"],
            "section": selected["section"],
            "preset": Path(selected["config"]).stem,
            "horizon": config["execution_horizon"],
            "scope": selected["scope"],
            "reason": selected["reason"],
            "development": {
                "successes": dev["successes"],
                "episodes": 200,
                "delta_pp": delta,
                "target_met": dev["successes"] >= ref["successes"] - 6,
                "experiment_id": dev["experiment_id"],
            },
            "timing": {
                "record": selected["timing_record"],
                "mean_ms": latency["mean_ms"],
                "speedup": record["workload_speedup" if compact else "task_macro_speedup"],
                "repeat_std_ms": latency.get("repeat_std_ms"),
                "peak_allocated_bytes": record["run"]["summary"]["peak_allocated_bytes"],
                "record_sha256": stable_hash(record),
                "quality_group": cohort["group"],
            },
            "final": final,
        }
        if selected["status"] == "pending_core":
            row["config"] = selected["config"]
            pending.append(row)
            continue
        rows.append(row)
        launch.append(
            {
                "config": selected["config"],
                "config_sha256": stable_hash(policy_identity(config)),
                "checkpoint_sha256": dev["backend"]["checkpoint_sha256"],
                "timing_record": selected["timing_record"],
                "owner_authorization_required": True,
            }
        )
    coverage = selection.get("coverage_summary_sha256")
    if coverage is not None:
        expected = {scope_key(r["config"]) for r in summaries[coverage]["configurations"]}
        if seen != expected:
            raise ValueError(
                "Selection omits or adds scopes relative to its frozen review coverage"
            )
    auxiliary, auxiliary_seen = [], set()
    primary_identities = {
        stable_hash(policy_identity(asdict(ExperimentConfig.load(Path(r["config"])))))
        for r in selection["configurations"]
    }
    primary_scopes = {(r["backend"], r["method"]) for r in rows}
    for selected in selection.get("auxiliary_configurations", []):
        config = asdict(ExperimentConfig.load(Path(selected["config"])))
        identity = stable_hash(policy_identity(config))
        if (
            selected.get("status") != "auxiliary"
            or (config["backend"], config["method"]) not in primary_scopes
            or identity in primary_identities | auxiliary_seen
        ):
            raise ValueError(
                "Auxiliary must be a distinct alternative to a certified primary scope"
            )
        auxiliary_seen.add(identity)
        scoped = dict(selection, configurations=[dict(selected, status="primary")])
        scoped.pop("auxiliary_configurations", None)
        scoped.pop("coverage_summary_sha256", None)
        checked = build(scoped, summaries, records, quality, development_manifest)
        auxiliary.extend(checked["primary"])
    return {
        "schema_version": 1,
        "kind": "primary_readiness",
        "timing_protocol": selection.get("timing_protocol", "legacy_task_macro"),
        "selection_sha256": stable_hash(selection),
        "development_manifest_sha256": manifest,
        "primary": rows,
        "pending_core": pending,
        "auxiliary": auxiliary,
        "final_launch_candidates": launch,
        "final_launched_by_this_command": False,
    }


def render(report):
    timing_description = (
        "Timing uses frozen four-suite prefixes (at most 128 controls each), ten warmup "
        "queries, one pass and the mean over all measured controls."
        if report.get("timing_protocol") == "libero_replay4x128_v1"
        else "Historical timing uses the protocol and aggregation recorded in each source."
    )
    lines = [
        "# Benchmark results",
        "",
        "One selected configuration per backend/method scope. Dev-200 is tuning evidence,",
        "not final accuracy. " + timing_description,
        "Native pre/post checks must pass. Final success",
        "is unmeasured until the owner schedules the frozen 2,000-episode campaign.",
        "Low success and slowdowns remain reportable. VOTE H1 is separate from H8.",
        "Each speedup uses its own same-GPU native pair; the displayed native row is",
        "not a universal denominator. Absolute latency across GPU pairs is descriptive.",
        "",
        f"Selection digest: `{report['selection_sha256']}`.",
        "",
    ]
    for backend in sorted({r["backend"] for r in report["primary"]}):
        lines += [
            f"## {backend}",
            "",
            "| Section | Preset | H | Dev SR (n/200) | Δ pp | ms/control | "
            "Speedup | Peak GiB | Final SR (n/2000) | Selection / scope |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for r in report["primary"]:
            if r["backend"] != backend:
                continue
            d, t = r["development"], r["timing"]
            final = "—"
            if r["final"]:
                f = r["final"]["full_2000"]
                final = f"{100 * f['success_rate']:.2f}% ({f['successes']})"
            memory = (
                f"{t['peak_allocated_bytes'] / 2**30:.2f}"
                if t["peak_allocated_bytes"] is not None
                else "—"
            )
            lines.append(
                f"| {r['section']} | `{r['preset']}` | {r['horizon']} | "
                f"{d['successes'] / 2:.1f}% ({d['successes']}) | {d['delta_pp']:+.1f} | "
                f"{t['mean_ms']:.3f} | {t['speedup']:.3f}× | "
                f"{memory} | {final} | "
                f"{r['reason']} {r['scope']} |"
            )
        lines += [""]
    if report.get("auxiliary"):
        lines += [
            "## Useful auxiliary configurations",
            "",
            "Secondary evidence only; these do not enter the default final-evaluation queue.",
            "",
            "| Backend / preset | Dev SR (n/200) | ms/control | Speedup | Reason / scope |",
            "|---|---:|---:|---:|---|",
        ]
        for r in report["auxiliary"]:
            d, t = r["development"], r["timing"]
            lines.append(
                f"| {r['backend']} / `{r['preset']}` | {d['successes'] / 2:.1f}% "
                f"({d['successes']}) | {t['mean_ms']:.3f} | {t['speedup']:.3f}× | "
                f"{r['reason']} {r['scope']} |"
            )
        lines.append("")
    lines += [
        "## Not certified for final evaluation",
        "",
        "Diagnostic measurements below do NOT certify the unresolved core or authorize final runs.",
        "",
        "| Backend / method | Dev SR (n/200) | ms/control | Speedup | Reason |",
        "|---|---:|---:|---:|---|",
    ]
    for r in report["pending_core"]:
        d, t = r.get("development"), r.get("timing")
        success = f"{d['successes'] / 2:.1f}% ({d['successes']})" if d else "—"
        latency = f"{t['mean_ms']:.3f}" if t else "—"
        speedup = f"{t['speedup']:.3f}×" if t else "—"
        lines.append(
            f"| {r['backend']} / `{r['method']}` | {success} | {latency} | "
            f"{speedup} | {r['reason']} |"
        )
    lines += [
        "",
        "## Evidence index",
        "",
        "Raw measurements and resource locations are ignored local evidence, not public files.",
        "Use the selection manifest and generated readiness JSON to resolve exact records.",
        "",
        "| Backend / preset | Dev experiment | Timing record | Record digest |",
        "|---|---|---|---|",
    ]
    for r in (
        report["primary"]
        + report.get("auxiliary", [])
        + [p for p in report["pending_core"] if p.get("timing")]
    ):
        lines.append(
            f"| {r['backend']} / `{r['preset']}` | `{r['development']['experiment_id']}` | "
            f"`{r['timing']['record']}` | `{r['timing']['record_sha256']}` |"
        )
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--development-summary", type=Path, action="append", required=True)
    p.add_argument("--development-manifest", type=Path, required=True)
    p.add_argument("--records-dir", type=Path, required=True)
    p.add_argument("--timing-quality", type=Path, required=True)
    p.add_argument("--final-summary", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--evidence-output", type=Path, required=True)
    a = p.parse_args()
    selection = yaml.safe_load(a.selection.read_text())
    summaries = {digest_file(f): json.loads(f.read_text()) for f in a.development_summary}
    records = {
        r["timing_record"]: json.loads((a.records_dir / (r["timing_record"] + ".json")).read_text())
        for r in selection["configurations"] + selection.get("auxiliary_configurations", [])
        if r.get("timing_record")
    }
    finals = [r for f in a.final_summary for r in json.loads(f.read_text())["configurations"]]
    report = build(
        selection,
        summaries,
        records,
        json.loads(a.timing_quality.read_text()),
        load_manifest(a.development_manifest),
        finals,
    )
    a.output.write_text(render(report))
    a.evidence_output.parent.mkdir(parents=True, exist_ok=True)
    a.evidence_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({"primary": len(report["primary"]), "pending_core": len(report["pending_core"])})
    )


if __name__ == "__main__":
    main()
