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


def cadence(metadata):
    """Read executed-prefix policy from measured metadata, never the preset name."""
    policy = (
        metadata.get("backend", {})
        .get("method_details", {})
        .get("execution_horizon_policy", "native")
    )
    if policy not in {"native", "adaptive"}:
        raise ValueError("Unknown measured execution-horizon policy")
    if metadata["config"]["method"] == "none" and policy != "native":
        raise ValueError("A native reference cannot have adaptive execution")
    predicted = metadata.get("backend", {}).get("predicted_horizon")
    if policy == "adaptive" and (type(predicted) is not int or predicted < 1):
        raise ValueError("Adaptive execution requires the measured predicted horizon")
    return policy, predicted


def cadence_label(row):
    if row["execution_horizon_policy"] == "adaptive":
        return f"adaptive ≤{row['predicted_horizon']}"
    return str(row["horizon"])


def check_search(
    selected, config, dev, reference, summaries, records, valid_runs, manifest, selection
):
    """Require evidence of search closure, independently of accuracy eligibility."""
    if config["method"] == "none":
        return {"complete": True, "stop": "native_reference", "reason": "Fixed native reference"}
    search = selected.get("search", {})
    if search.get("complete") is not True:
        return {
            "complete": False,
            "stop": "pending",
            "reason": search.get("reason", "Not reviewed"),
        }
    allowed = {
        "accuracy_boundary",
        "latency_plateau",
        "mechanism_limit",
        "no_feasible_configuration",
    }
    if search.get("stop") not in allowed or not search.get("reason"):
        raise ValueError(
            "Completed search needs an explicit boundary or limitation, not target_met"
        )
    candidates = []
    for evidence in search.get("evidence", []):
        candidate = None
        if evidence.get("development_experiment_id"):
            matches = [
                row
                for row in summaries[evidence["development_summary_sha256"]]["configurations"]
                if row["experiment_id"] == evidence["development_experiment_id"]
            ]
            if len(matches) != 1:
                raise ValueError("Search candidate development evidence is missing or ambiguous")
            candidate = matches[0]
            check_development(candidate, manifest)
            candidate_config = candidate["config"]
            if candidate["backend"]["checkpoint_sha256"] != dev["backend"]["checkpoint_sha256"]:
                raise ValueError("Search candidate checkpoint differs")
        timing = None
        if evidence.get("timing_record"):
            timing = records[evidence["timing_record"]]
            if timing["run"]["run_id"] not in valid_runs:
                raise ValueError("Search candidate timing has no valid quality cohort")
            measured = timing["run"]["metadata"]
            candidate_config = measured["config"]
            if candidate and policy_identity(candidate["config"]) != policy_identity(
                candidate_config
            ):
                raise ValueError("Search candidate Dev/timing configurations differ")
            if measured["backend"]["checkpoint_sha256"] != dev["backend"]["checkpoint_sha256"]:
                raise ValueError("Search candidate timing checkpoint differs")
            cohort = valid_runs[timing["run"]["run_id"]]
            cadence(measured)
            cadence(timing["baseline"]["metadata"])
            baseline = timing["baseline"]["metadata"]["config"]
            if (
                timing["baseline"]["run_id"] != cohort["baseline_pre"]
                or baseline["method"] != "none"
                or baseline["execution_horizon"] != config["execution_horizon"]
                or measured["protocol"].get("workload_sha256")
                != selection["timing_workloads"][config["backend"]]
                or (
                    selection.get("timing_protocol")
                    and measured["protocol"].get("protocol_id") != selection["timing_protocol"]
                )
            ):
                raise ValueError("Search timing changes the workload, protocol or native reference")
            if candidate:
                pair = (
                    candidate["source"]["runtime_source_sha256"],
                    measured["source"]["runtime_source_sha256"],
                )
                audited = {
                    (r["development_runtime"], r["timing_runtime"])
                    for r in selection["source_compatibility"]
                    if r.get("reason")
                }
                if (
                    measured["backend"] != candidate["backend"]
                    or environment_identity(measured["source"], config["backend"])
                    != environment_identity(candidate["source"], config["backend"])
                    or (pair[0] != pair[1] and pair not in audited)
                ):
                    raise ValueError("Search Dev/timing implementation or dependencies differ")
        if candidate is None and timing is None:
            raise ValueError("Search evidence must resolve a measurement, not a free-text claim")
        if scope_key(candidate_config) != scope_key(config) or any(
            candidate_config.get(k) != config.get(k)
            for k in ("model_variant", "precision", "execution_horizon", "backend_parameters")
        ):
            raise ValueError(
                "Search evidence changes the method scope or fixed baseline conditions"
            )
        candidates.append((candidate, timing))
    if not any(c and c["experiment_id"] == dev["experiment_id"] for c, _ in candidates):
        raise ValueError("Completed search must include the selected development candidate")
    complete_devs = [c for c, _ in candidates if c]
    threshold = reference["successes"] - 6
    if search["stop"] == "accuracy_boundary" and not any(
        c["successes"] < threshold for c in complete_devs
    ):
        raise ValueError("Accuracy boundary requires an actually measured below-target candidate")
    if (
        search["stop"] == "latency_plateau"
        and len({r["run"]["run_id"] for _, r in candidates if r}) < 2
    ):
        raise ValueError("Latency plateau requires at least two valid timed candidates")
    if search["stop"] == "no_feasible_configuration" and (
        len({c["experiment_id"] for c in complete_devs}) < 2
        or any(c["successes"] >= threshold for c in complete_devs)
    ):
        raise ValueError("No-feasible closure disagrees with the measured candidate set")
    if search["stop"] == "mechanism_limit" and not search.get("limitation"):
        raise ValueError("Mechanism-limit closure needs the concrete unalterable constraint")
    selected_record = records[selected["timing_record"]]
    selected_speed = selected_record.get(
        "workload_speedup", selected_record.get("task_macro_speedup")
    )
    faster = [
        c["experiment_id"]
        for c, r in candidates
        if c
        and r
        and c["successes"] >= threshold
        and r.get("workload_speedup", r.get("task_macro_speedup")) > selected_speed
        and c["experiment_id"] != dev["experiment_id"]
    ]
    if faster and not search.get("tie_reason"):
        raise ValueError(
            "A faster eligible measured candidate needs an explicit selection explanation"
        )
    return {
        **search,
        "eligible_candidates": sum(c["successes"] >= threshold for c in complete_devs),
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
    held = {"pending_core": pending, "pending_engineering": [], "excluded": []}
    scope_status = {}
    for selected in selection["configurations"]:
        config = asdict(ExperimentConfig.load(Path(selected["config"])))
        if config["execution_horizon"] != 8:
            raise ValueError("Primary comparison uses H8; historical H1 is not a special exception")
        key = scope_key(config)
        if key in seen:
            raise ValueError("Multiple primary configurations for one backend/method scope")
        seen.add(key)
        status = selected["status"]
        if status not in {"primary", *held}:
            raise ValueError("Unknown readiness status")
        scope_status[(config["backend"], config["method"])] = status
        if not all(selected.get(k) for k in ("reason", "scope", "section")):
            raise ValueError("Every selection needs a reason, core scope and section")
        if status in held and not selected.get("timing_record"):
            held[status].append(
                {
                    "backend": config["backend"],
                    "method": config["method"],
                    "config": selected["config"],
                    "section": selected["section"],
                    "reason": selected["reason"],
                    "status": status,
                    "scope": selected["scope"],
                }
            )
            continue
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
        horizon_policy, predicted_horizon = cadence(timing)
        cadence(baseline)
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
        final_runtime = selected.get(
            "final_runtime_source_sha256", selection.get("final_runtime_source_sha256")
        )
        if "final_runtime_source_sha256" in selected and (
            not final_runtime
            or (final_runtime != source_pair[1] and (source_pair[0], final_runtime) not in audited)
        ):
            raise ValueError("Unaudited scoped final runtime override")
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
        cadence(ref)
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
                if measured["source"]["runtime_source_sha256"] != final_runtime:
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
            "execution_horizon_policy": horizon_policy,
            "predicted_horizon": predicted_horizon,
            "scope": selected["scope"],
            "reason": selected["reason"],
            "hardware": timing.get("hardware", {}),
            "timing_commit": timing["source"].get("commit"),
            "development_commit": dev["source"].get("commit"),
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
        if status in held:
            row["config"] = selected["config"]
            row["status"] = status
            held[status].append(row)
            continue
        row["search"] = check_search(
            selected,
            config,
            dev,
            ref,
            summaries,
            records,
            quality_by_run,
            development_manifest,
            selection,
        )
        if final is not None and not row["search"]["complete"]:
            raise ValueError("Final results cannot certify an unfinished parameter search")
        rows.append(row)
        if not row["search"]["complete"] or final is not None:
            continue
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
    for selected in selection.get("auxiliary_configurations", []):
        config = asdict(ExperimentConfig.load(Path(selected["config"])))
        identity = stable_hash(policy_identity(config))
        parent_status = scope_status.get((config["backend"], config["method"]))
        if (
            selected.get("status") != "auxiliary"
            or parent_status not in {"primary", "pending_engineering"}
            or identity in primary_identities | auxiliary_seen
        ):
            raise ValueError(
                "Auxiliary must be a distinct alternative to a certified primary scope "
                "or an explicitly engineering-held scope"
            )
        auxiliary_seen.add(identity)
        scoped = dict(selection, configurations=[dict(selected, status=parent_status)])
        scoped.pop("auxiliary_configurations", None)
        scoped.pop("coverage_summary_sha256", None)
        checked = build(scoped, summaries, records, quality, development_manifest)
        auxiliary.extend(checked["primary"])
        for row in checked["pending_engineering"]:
            held["pending_engineering"].append(dict(row, auxiliary=True))
    return {
        "schema_version": 1,
        "kind": "primary_readiness",
        "timing_protocol": selection.get("timing_protocol", "legacy_task_macro"),
        "selection_sha256": stable_hash(selection),
        "development_manifest_sha256": manifest,
        "primary": rows,
        "pending_core": pending,
        "pending_engineering": held["pending_engineering"],
        "excluded": held["excluded"],
        "auxiliary": auxiliary,
        "final_launch_candidates": launch,
        "final_launched_by_this_command": False,
    }


def render(report):
    uncertified = (
        report["pending_core"] + report.get("pending_engineering", []) + report.get("excluded", [])
    )
    measured = report["primary"] + report.get("auxiliary", []) + uncertified
    devices = sorted({r["hardware"]["gpu"] for r in measured if r.get("hardware", {}).get("gpu")})
    timing_description = (
        "Timing uses frozen four-suite prefixes (at most 128 controls each), ten warmup "
        "queries, one pass and the mean over all measured controls."
        if report.get("timing_protocol") == "libero_replay4x128_v1"
        else "Historical timing uses the protocol and aggregation recorded in each source."
    )
    final_count = sum(row["final"] is not None for row in report["primary"])
    final_description = (
        f"Final success is measured for {final_count}/{len(report['primary'])} primary "
        "configurations, using one frozen 2,000-episode run per measured configuration."
        if final_count
        else "Final success is unmeasured until the owner schedules the frozen "
        "2,000-episode campaign."
    )
    lines = [
        "# Benchmark results",
        "",
        "One selected configuration per backend/method scope. Dev-200 is tuning evidence,",
        "not final accuracy. " + timing_description,
        "Native pre/post checks must pass. " + final_description,
        "Low success and slowdowns remain reportable. All native references use H8.",
        "Adaptive rows execute real method-selected prefixes; H8 is their reference/default,",
        "not their fixed execution length. The shown upper bound is native prediction length.",
        "Complete measurements do not certify parameter-search completion; unfinished",
        "searches are excluded from the final launch list even when Dev meets the target.",
        "Selection targets at most 3 percentage points of Dev loss versus the matched",
        "native; this is not an outcome filter. Δ pp uses that same native-H8 reference.",
        "Speedup >1 means faster and <1 means slower. Near-1 differences from one pass",
        "do not establish a reliable gain; no repeat standard deviation is claimed.",
        "Each speedup uses its own same-GPU native pair; the displayed native row is",
        "not a universal denominator. Absolute latency across GPU pairs is descriptive.",
        "Measured GPU models: " + (", ".join(devices) or "see retained source metadata") + ".",
        "",
        f"Selection digest: `{report['selection_sha256']}`.",
        "",
    ]
    for backend in sorted({r["backend"] for r in report["primary"]}):
        lines += [
            f"## {backend}",
            "",
            "| Section | Preset | Execution H | Dev SR (n/200) | Δ pp | ms/control | "
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
                f"| {r['section']} | `{r['preset']}` | {cadence_label(r)} | "
                f"{d['successes'] / 2:.1f}% ({d['successes']}) | {d['delta_pp']:+.1f} | "
                f"{t['mean_ms']:.3f} | {t['speedup']:.3f}× | "
                f"{memory} | {final} | "
                f"{r['reason']} {r['scope']} "
                f"Search: {r['search']['stop']}. |"
            )
        lines += [""]
    if report.get("auxiliary"):
        lines += [
            "## Useful auxiliary configurations",
            "",
            "Secondary evidence only; these do not enter the default final-evaluation queue.",
            "",
            "| Backend / preset | Execution H | Dev SR (n/200) | ms/control | "
            "Speedup | Reason / scope |",
            "|---|---|---:|---:|---:|---|",
        ]
        for r in report["auxiliary"]:
            d, t = r["development"], r["timing"]
            lines.append(
                f"| {r['backend']} / `{r['preset']}` | {cadence_label(r)} | "
                f"{d['successes'] / 2:.1f}% "
                f"({d['successes']}) | {t['mean_ms']:.3f} | {t['speedup']:.3f}× | "
                f"{r['reason']} {r['scope']} |"
            )
        lines.append("")
    lines += [
        "## Not certified for final evaluation",
        "",
        "Diagnostic measurements below do NOT certify core fidelity or engineering readiness, "
        "and never authorize final runs. Exclusion reasons are independent of measured outcomes.",
        "",
        "| Backend / method | Status | Dev SR (n/200) | ms/control | Speedup | Reason |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in uncertified:
        d, t = r.get("development"), r.get("timing")
        success = f"{d['successes'] / 2:.1f}% ({d['successes']})" if d else "—"
        latency = f"{t['mean_ms']:.3f}" if t else "—"
        speedup = f"{t['speedup']:.3f}×" if t else "—"
        lines.append(
            f"| {r['backend']} / `{r.get('preset', r['method'])}` | "
            f"{r.get('status', 'pending_core')} | {success} | {latency} | "
            f"{speedup} | {r['reason']} |"
        )
    lines += [
        "",
        "## Evidence index",
        "",
        "Raw measurements and resource locations are ignored local evidence, not public files.",
        "Use the selection manifest and generated readiness JSON to resolve exact records.",
        "Recorded revisions predating history cleanup can be resolved through the",
        "[history map](../scripts/history_commit_map.json); recorded source IDs are not rewritten.",
        "",
        "| Backend / preset | Dev experiment | Final experiment | Timing record | "
        "Dev / timing commit | Record digest |",
        "|---|---|---|---|---|---|",
    ]
    for r in (
        report["primary"]
        + report.get("auxiliary", [])
        + [p for p in uncertified if p.get("timing")]
    ):
        final_id = f"`{r['final']['experiment_id']}`" if r["final"] else "—"
        lines.append(
            f"| {r['backend']} / `{r['preset']}` | `{r['development']['experiment_id']}` | "
            f"{final_id} | `{r['timing']['record']}` | "
            f"`{r.get('development_commit') or 'unknown'}` / "
            f"`{r.get('timing_commit') or 'unknown'}` | `{r['timing']['record_sha256']}` |"
        )
    return "\n".join(lines) + "\n"


def protect_supplementary_output(path):
    """Primary regeneration must not silently erase separately audited controls."""
    if path.exists() and "<!-- supplementary-native-controls -->" in path.read_text():
        raise ValueError(
            "Output contains independently audited native controls; generate the primary "
            "report to a different ignored output and reconcile the supplementary evidence"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--development-summary", type=Path, action="append", required=True)
    p.add_argument("--development-manifest", type=Path, required=True)
    p.add_argument("--records-dir", type=Path, required=True)
    p.add_argument("--timing-quality", type=Path, required=True)
    p.add_argument("--final-summary", type=Path, action="append", default=[])
    p.add_argument("--codec-run", type=Path, action="append", default=[])
    p.add_argument("--codec-auxiliary-run", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--evidence-output", type=Path, required=True)
    a = p.parse_args()
    protect_supplementary_output(a.output)
    selection = yaml.safe_load(a.selection.read_text())
    summaries = {digest_file(f): json.loads(f.read_text()) for f in a.development_summary}
    selected = selection["configurations"] + selection.get("auxiliary_configurations", [])
    record_ids = {r["timing_record"] for r in selected if r.get("timing_record")}
    record_ids.update(
        e["timing_record"]
        for r in selected
        for e in r.get("search", {}).get("evidence", [])
        if e.get("timing_record")
    )
    records = {
        name: json.loads((a.records_dir / (name + ".json")).read_text()) for name in record_ids
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
    from efficient_vla_benchmark.action_codecs.report import load_result
    from efficient_vla_benchmark.action_codecs.report import render_groups as render_codecs

    if (
        not a.codec_run
        and a.output.exists()
        and "## Independent action codec reconstruction" in a.output.read_text()
    ):
        raise ValueError("Existing codec evidence must be supplied; refuse to silently drop it")
    report["action_codecs"] = [load_result(directory) for directory in a.codec_run]
    if (
        not a.codec_auxiliary_run
        and a.output.exists()
        and "Codec device checks" in a.output.read_text()
    ):
        raise ValueError("Existing auxiliary codec checks must be supplied")
    report["auxiliary_action_codecs"] = [
        load_result(directory) for directory in a.codec_auxiliary_run
    ]
    a.output.write_text(
        render(report) + render_codecs(report["action_codecs"], report["auxiliary_action_codecs"])
    )
    a.evidence_output.parent.mkdir(parents=True, exist_ok=True)
    a.evidence_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({"primary": len(report["primary"]), "pending_core": len(report["pending_core"])})
    )


if __name__ == "__main__":
    main()
