"""Keep audited experiment records local and render the public measurement summary."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def load_history_map(path=None):
    """Resolve historical IDs without changing immutable measurement evidence."""
    path = path or Path(__file__).with_name("history_commit_map.json")
    mapping = read_json(path)
    if not isinstance(mapping, dict) or any(
        not isinstance(old, str)
        or not isinstance(new, str)
        or not re.fullmatch(r"[0-9a-f]{40}", old)
        or not re.fullmatch(r"[0-9a-f]{40}", new)
        or new == "0" * 40
        for old, new in mapping.items()
    ):
        raise ValueError("Invalid history commit map")
    return mapping


def revision_text(source, commit_map):
    commit = source["commit"] or "unknown"
    dirty = " (dirty)" if source["dirty"] else ""
    original = f"`{commit}`{dirty}"
    cleaned = commit_map.get(commit)
    return f"{original} → `{cleaned}`" if cleaned and cleaned != commit else original


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def token_evidence(rows):
    """Summarize query diagnostics, never counting queued-action rows twice."""
    observations = []
    for row in rows:
        if not row["queried"]:
            continue
        diag = row.get("diagnostics", {})
        # Accept the original diagnostic spelling for historical integrations.
        count = diag.get("visual_tokens_final", diag.get("final_visual_tokens"))
        if count is None:
            continue
        if type(count) is not int or count < 0:
            raise ValueError("Invalid visual-token diagnostic")
        observations.append((count, diag))
    if not observations:
        return None
    counts = [count for count, _ in observations]
    result = {
        "queries_with_diagnostics": len(counts),
        "final_visual_tokens": {
            "mean": sum(counts) / len(counts),
            "min": min(counts),
            "max": max(counts),
        },
    }
    for key in ("pruned", "keyframe", "warmup"):
        values = [d[key] for _, d in observations if key in d]
        if values:
            if any(type(value) is not bool for value in values):
                raise ValueError("Invalid token-mode diagnostic")
            result[key] = {"true": sum(values), "observed": len(values)}
    early = [d["visual_tokens_early"] for _, d in observations if "visual_tokens_early" in d]
    if early:
        result["early_visual_tokens"] = {
            "mean": sum(early) / len(early),
            "min": min(early),
            "max": max(early),
        }
    return result


def cache_evidence(rows):
    """Keep skipped queries distinct from retained KV context; ignore queued actions."""
    counts = []
    for row in rows:
        if not row["queried"]:
            continue
        diag = row.get("diagnostics", {})
        query, context = diag.get("visual_query_tokens_final"), diag.get("visual_kv_tokens")
        if query is None and context is None:
            continue
        if type(query) is not int or type(context) is not int or not 0 <= query <= context:
            raise ValueError("Invalid cached-query/context diagnostic")
        counts.append((query, context))
    if not counts:
        return None
    result = {"queries_with_diagnostics": len(counts)}
    for index, name in enumerate(("computed_visual_queries", "retained_visual_kv")):
        values = [pair[index] for pair in counts]
        result[name] = {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}
    return result


def auxiliary_evidence(rows):
    """Count discarded native passes separately, without hiding them in serving counts."""
    observations = []
    for row in rows:
        if not row["queried"]:
            continue
        diag = row.get("diagnostics", {})
        calls = diag.get("auxiliary_queries", 0)
        if type(calls) is not int or calls not in (0, 1):
            raise ValueError("Invalid auxiliary call count")
        if not calls:
            continue
        for key in ("primary_forward_ms", "auxiliary_forward_ms", "auxiliary_join_wait_ms"):
            value = diag.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid auxiliary timing diagnostic")
        count = diag.get("auxiliary_visual_tokens_final")
        if type(count) is not int or count <= 0:
            raise ValueError("Invalid auxiliary visual-token diagnostic")
        observations.append(diag)
    if not observations:
        return None
    result = {"auxiliary_calls": len(observations)}
    for key in (
        "primary_forward_ms",
        "auxiliary_forward_ms",
        "auxiliary_join_wait_ms",
        "auxiliary_visual_tokens_final",
    ):
        values = [diag[key] for diag in observations]
        result[key] = {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}
    return result


def timing_evidence(directory):
    metadata = read_json(directory / "metadata.json")
    summary = read_json(directory / "summary.json")
    rows = [json.loads(line) for line in (directory / "timings.jsonl").read_text().splitlines()]
    if metadata["protocol"]["mode"] != "control":
        raise ValueError("The comparison ledger requires full control replay")
    if len(rows) != summary["all_control_steps"]["count"]:
        raise ValueError("Raw row count disagrees with summary")
    queries = [row for row in rows if row["queried"]]
    if len(queries) != summary["all_queries"]["count"]:
        raise ValueError("Raw query count disagrees with summary")
    latencies = [row["latency_ms"] for row in rows]
    if not all(math.isfinite(value) and value > 0 for value in latencies):
        raise ValueError("Invalid latency in raw records")
    if not math.isclose(sum(latencies), summary["replay_policy_time_ms"], rel_tol=1e-9):
        raise ValueError("Raw latency sum disagrees with summary")
    if not math.isclose(
        sum(latencies) / len(rows), summary["all_control_steps"]["mean_ms"], rel_tol=1e-9
    ):
        raise ValueError("Raw control mean disagrees with summary")
    if not math.isclose(
        sum(r["latency_ms"] for r in queries) / len(queries),
        summary["all_queries"]["mean_ms"],
        rel_tol=1e-9,
    ):
        raise ValueError("Raw query mean disagrees with summary")
    auxiliary = auxiliary_evidence(rows)
    auxiliary_calls = auxiliary["auxiliary_calls"] if auxiliary else 0
    if summary.get("auxiliary_model_calls", 0) != auxiliary_calls:
        raise ValueError("Raw auxiliary call count disagrees with summary")
    if summary.get("total_native_model_calls", len(queries)) != len(queries) + auxiliary_calls:
        raise ValueError("Raw total native call count disagrees with summary")
    return {
        "run_id": directory.name,
        "metadata": metadata,
        "summary": summary,
        "token_diagnostics": token_evidence(rows),
        "cache_diagnostics": cache_evidence(rows),
        "auxiliary_diagnostics": auxiliary,
        "files_sha256": {
            name: digest(directory / name)
            for name in ("metadata.json", "summary.json", "timings.jsonl")
        },
    }


def make_record(run_dir, baseline_dir, section, label, preset, smoke_dir=None):
    run, baseline = timing_evidence(run_dir), timing_evidence(baseline_dir)
    left, right = run["metadata"], baseline["metadata"]
    for key in ("backend", "model_variant", "checkpoint_sha256"):
        if left["backend"][key] != right["backend"][key]:
            raise ValueError(f"Incomparable baseline: {key}")
    if left["traces"] != right["traces"] or left["protocol"] != right["protocol"]:
        raise ValueError("Incomparable baseline trace or timing protocol")
    if left["hardware"] != right["hardware"]:
        raise ValueError("Incomparable baseline hardware")
    for key in ("source_sha256", "submodules", "packages", "cuda_extension_sha256"):
        if left["source"].get(key) != right["source"].get(key):
            raise ValueError(f"Incomparable baseline software: {key}")
    if (
        run["summary"]["all_control_steps"]["count"]
        != baseline["summary"]["all_control_steps"]["count"]
    ):
        raise ValueError("Incomparable baseline control-step count")
    differences = {}
    for key in ("precision", "attention_implementation"):
        a, b = left["backend"].get(key), right["backend"].get(key)
        if a != b:
            differences[key] = {"method": a, "baseline": b}
    a, b = left["config"]["execution_horizon"], right["config"]["execution_horizon"]
    if a != b:
        differences["execution_horizon"] = {"method": a, "baseline": b}
    record = {
        "schema_version": 1,
        "section": section,
        "label": label,
        "preset": preset,
        "run": run,
        "baseline": baseline,
        "replay_speedup": baseline["summary"]["replay_policy_time_ms"]
        / run["summary"]["replay_policy_time_ms"],
        "comparison_differences": differences,
        "formal_success_rate": None,
    }
    if smoke_dir is not None:
        meta = read_json(smoke_dir / "metadata.json")
        if meta["kind"] != "development_smoke":
            raise ValueError("--smoke-dir must not contain a formal evaluation")
        for key in ("backend", "model_variant", "method", "checkpoint_sha256", "precision"):
            if meta["backend"][key] != left["backend"][key]:
                raise ValueError(f"Smoke identity differs: {key}")
        for key in ("execution_horizon", "parameters"):
            if meta["config"][key] != left["config"][key]:
                raise ValueError(f"Smoke configuration differs: {key}")
        record["smoke"] = {
            "run_id": smoke_dir.name,
            "metadata": meta,
            "summary": read_json(smoke_dir / "summary.json"),
            "episodes": [
                json.loads(line) for line in (smoke_dir / "episodes.jsonl").read_text().splitlines()
            ],
            "files_sha256": {
                name: digest(smoke_dir / name)
                for name in ("metadata.json", "summary.json", "episodes.jsonl")
            },
            "same_source_fingerprint": meta["source"]["source_sha256"]
            == left["source"]["source_sha256"],
        }
    return record


def annotate_supersedes(record, run_ids, records_dir):
    """Annotate corrected implementations without rewriting historical evidence."""
    if not run_ids:
        return
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Duplicate superseded run ID")
    current = record["run"]["metadata"]
    for run_id in run_ids:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id) or run_id == record["run"]["run_id"]:
            raise ValueError("Invalid superseded run ID")
        path = records_dir / f"{run_id}.json"
        if not path.is_file():
            raise ValueError("Superseded evidence must already exist")
        previous = read_json(path)["run"]["metadata"]
        for key in ("backend", "model_variant"):
            if previous["backend"][key] != current["backend"][key]:
                raise ValueError("Cannot supersede another backend or model variant")
        if (
            current["config"].get("method") is None
            or previous["config"].get("method") != current["config"]["method"]
        ):
            raise ValueError("Cannot supersede another method")
    record["supersedes"] = sorted(run_ids)


def record_digest(record):
    """Identify the complete local record without committing or linking its contents."""
    payload = json.dumps(record, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def render(records, commit_map=None):
    commit_map = load_history_map() if commit_map is None else commit_map
    records = sorted(records, key=lambda record: record["run"]["run_id"])
    ids = {record["run"]["run_id"]: f"R{index:02d}" for index, record in enumerate(records, 1)}
    cohorts, cohort_ids = {}, {}
    for record in records:
        run = record["run"]
        meta = run["metadata"]
        identity = {
            "backend": meta["backend"]["backend"],
            "variant": meta["backend"]["model_variant"],
            "checkpoint": meta["backend"]["checkpoint_sha256"],
            "source": meta["source"],
            "hardware": meta["hardware"],
            "protocol": meta["protocol"],
            "traces": meta["traces"],
        }
        key = json.dumps(identity, sort_keys=True)
        if key not in cohorts:
            cohorts[key] = (f"C{len(cohorts) + 1:02d}", run)
        cohort_ids[run["run_id"]] = cohorts[key][0]
    superseded = {run_id for record in records for run_id in record.get("supersedes", [])}

    def display_label(record):
        prefix = "Superseded — " if record["run"]["run_id"] in superseded else ""
        return prefix + record["label"]

    lines = [
        "# Algorithm measurement ledger",
        "",
        "Generated from local, ignored JSON records in `outputs/result_records/`.",
        "Only this summary is tracked now; raw runs and the live record archive stay local.",
        "This is a development ledger, not a final paper comparison or formal success rate.",
        "Superseded rows preserve earlier, corrected implementations for traceability;",
        "do not use them as results of the current implementation. Their raw records",
        "are unchanged. Valid earlier cohorts remain separate, not selected by best speed.",
        "Mechanism scope and limits are detailed in the method READMEs.",
        "",
        "Backend, hardware and protocol are recorded per row below. Policy timing",
        "includes preprocessing, transfer, all method/control overhead and postprocessing,",
        "but not model loading, trace I/O or simulation. Fixed replay is not closed-loop",
        "task completion time. Different protocol cohorts must not be pooled.",
        "",
        "Speedup = paired native baseline total replay time / method total replay time.",
        "H is the executed prefix, distinct from the backend's full prediction length.",
        "VOTE presets use a matched H=1 baseline on each backend; the other current presets",
        "use native H=8. Do not compare their speedup",
        "ratios as if query cadence were identical. Peak GiB is runtime allocated tensor",
        "memory, not loading peak or total process memory. Smoke is successes/episodes,",
        "not a success-rate estimate. Formal success rate is **not measured for any row**.",
        "",
    ]
    sections = sorted({record["section"] for record in records})
    for section in sections:
        lines += [
            f"## {section}",
            "",
            "| Record / configuration | Backend | H | Queries / replacements | "
            "Query mean / p95 ms | ms/control step | Replay speedup | Peak GiB | Smoke |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for record in records:
            if record["section"] != section:
                continue
            run = record["run"]
            summary, meta = run["summary"], run["metadata"]
            query = summary["all_queries"]
            smoke = record.get("smoke", {}).get("summary")
            smoke_text = (
                f"{smoke['successes']}/{smoke['episodes']}; errors {smoke['errors']}"
                if smoke
                else "Not run"
            )
            memory = summary["peak_allocated_bytes"]
            memory_text = f"{memory / 2**30:.2f}" if memory is not None else "—"
            lines.append(
                f"| [{ids[run['run_id']]}](#{ids[run['run_id']].lower()}) "
                f"{display_label(record)} | "
                f"{meta['backend']['backend']} | "
                f"{meta['config']['execution_horizon']} | "
                f"{query['count']} / {summary.get('replacement_steps', 0)} | "
                f"{query['mean_ms']:.2f} / {query['p95_ms']:.2f} | "
                f"{summary['all_control_steps']['mean_ms']:.2f} | "
                f"{record['replay_speedup']:.3f}× | "
                f"{memory_text} | {smoke_text} |"
            )
        lines.append("")
    token_records = [r for r in records if r["run"].get("token_diagnostics")]
    if token_records:
        lines += [
            "## Observed visual-token retention",
            "",
            "Served-query-weighted counts, including dense served warmup/refresh; these",
            "are not configured keep ratios or FLOPs. Final counts do not describe",
            "earlier-layer cost. Per-layer sequence lengths remain in raw diagnostics.",
            "Discarded auxiliary passes are counted separately below, not as served queries.",
            "",
            "| Record / method | Backend | Final mean / min / max | Early mean | "
            "Mode counts (true/observed) |",
            "|---|---|---:|---:|---|",
        ]
        for record in token_records:
            diag = record["run"]["token_diagnostics"]
            final = diag["final_visual_tokens"]
            early = diag.get("early_visual_tokens")
            early_text = f"{early['mean']:.1f}" if early else "—"
            modes = (
                "; ".join(
                    f"{key} {diag[key]['true']}/{diag[key]['observed']}"
                    for key in ("pruned", "keyframe", "warmup")
                    if key in diag
                )
                or "—"
            )
            lines.append(
                f"| [{ids[record['run']['run_id']]}](#{ids[record['run']['run_id']].lower()}) "
                f"{display_label(record)} | {record['run']['metadata']['backend']['backend']} | "
                f"{final['mean']:.1f} / {final['min']} / "
                f"{final['max']} | {early_text} | {modes} |"
            )
        lines.append("")
    auxiliary_records = [r for r in records if r["run"].get("auxiliary_diagnostics")]
    if auxiliary_records:
        lines += [
            "## Auxiliary dense-refresh work",
            "",
            "Main replay rows include ALL auxiliary work. Primary-forward diagnostics exclude",
            "shared preparation and are not end-to-end serving latency. Refresh timing does",
            "not establish real-robot deadline feasibility or reduced total computation.",
            "",
            "| Record / backend | Served / auxiliary calls | Primary forward mean ms | "
            "Refresh mean / max ms | Auxiliary visual tokens |",
            "|---|---:|---:|---:|---:|",
        ]
        for record in auxiliary_records:
            run = record["run"]
            diag = run["auxiliary_diagnostics"]
            refresh = diag["auxiliary_forward_ms"]
            tokens = diag["auxiliary_visual_tokens_final"]
            lines.append(
                f"| {ids[run['run_id']]} / {run['metadata']['backend']['backend']} | "
                f"{run['summary']['all_queries']['count']} / {diag['auxiliary_calls']} | "
                f"{diag['primary_forward_ms']['mean']:.2f} | "
                f"{refresh['mean']:.2f} / {refresh['max']:.2f} | "
                f"{tokens['mean']:.1f} |"
            )
        lines.append("")
    cache_records = [r for r in records if r["run"].get("cache_diagnostics")]
    if cache_records:
        lines += [
            "## Computed visual queries and retained KV context",
            "",
            "Query-weighted last-layer counts; skipping query/MLP rows does not remove KV context.",
            "",
            "| Record / method | Backend | Computed queries: mean / min / max | "
            "Retained KV: mean / min / max |",
            "|---|---|---:|---:|",
        ]
        for record in cache_records:
            diag = record["run"]["cache_diagnostics"]
            query, context = diag["computed_visual_queries"], diag["retained_visual_kv"]
            lines.append(
                f"| [{ids[record['run']['run_id']]}](#{ids[record['run']['run_id']].lower()}) "
                f"{display_label(record)} | {record['run']['metadata']['backend']['backend']} | "
                f"{query['mean']:.1f} / {query['min']} / {query['max']} | "
                f"{context['mean']:.1f} / {context['min']} / {context['max']} |"
            )
        lines.append("")
    lines += [
        "## Measurement cohorts",
        "",
        "Each row uses its own paired baseline, never another cohort's denominator.",
        "F = measured control frames; R = repeats; W = warmup queries.",
        "",
        "| Cohort | Backend | Hardware | F / R / W | Source fingerprint | Checkpoint |",
        "|---|---|---|---:|---|---|",
    ]
    for cohort_id, run in cohorts.values():
        meta = run["metadata"]
        hardware = meta["hardware"].get("gpu", meta["hardware"].get("runtime", "unknown"))
        lines.append(
            f"| {cohort_id} | {meta['backend']['backend']} | {hardware} | "
            f"{run['summary']['all_control_steps']['count']} / "
            f"{meta['protocol']['repeats']} / {meta['protocol']['warmup_queries']} | "
            f"`{meta['source'].get('source_sha256', 'unknown')}` | "
            f"`{meta['backend']['checkpoint_sha256']}` |"
        )
    lines += [
        "",
        "<details>",
        "<summary>Run identities, presets and local-record checksums</summary>",
        "",
        "Raw runs: `outputs/<run ID>/`; exported records: `outputs/result_records/<run ID>.json`.",
        "The SHA256 identifies the full canonical JSON, which retains raw-file hashes,",
        "parameters, baseline, runtime/precision and per-episode smoke outcomes.",
        "The preset links are current. Timing revisions show original → cleaned Git ID.",
        "The [commit map](../scripts/history_commit_map.json) resolves pre-cleanup IDs;",
        "a removed documentation-only commit maps to its unchanged-source ancestor.",
        "This is source lookup, not a new measurement: original record hashes, source",
        "fingerprints and dirty flags are unchanged. Dirty runs need their original",
        "working-tree evidence; checking out the mapped commit alone is insufficient.",
        "Smoke-source labels: same / different / unknown / not run. Earlier smoke is",
        "never promoted to a final-source or formal evaluation.",
        "",
        "| Record | Run ID | Cohort | Preset | Timing commit: original → cleaned | Baseline | "
        "Smoke source | Record SHA256 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for record in records:
        run, meta = record["run"], record["run"]["metadata"]
        rid = ids[run["run_id"]]
        baseline_id = ids.get(record["baseline"]["run_id"], record["baseline"]["run_id"])
        revision = revision_text(meta["source"], commit_map)
        smoke = record.get("smoke")
        same = smoke.get("same_source_fingerprint") if smoke else None
        smoke_source = (
            ("same" if same else "different")
            if same is not None
            else ("unknown" if smoke else "not run")
        )
        lines.append(
            f'| <a id="{rid.lower()}"></a>{rid} | `{run["run_id"]}` | '
            f"{cohort_ids[run['run_id']]} | "
            f"[{Path(record['preset']).stem}](../{record['preset']}) | "
            f"{revision} | {baseline_id} | {smoke_source} | "
            f"`{record_digest(record)}` |"
        )
    lines += ["", "</details>"]
    lines += [
        "",
        "## Interpretation boundaries",
        "",
        "- DepthCache rows implement the primary-view progressive merging core only;",
        "  carrying-dependent scene reinitialization and anticipatory wrist gating are excluded.",
        "  Measured metric depth is required; RGB-only traces cannot stand in for this input.",
        "- ASR rows implement the deletion instrument with unconditional dense refresh,",
        "  not KV reuse. CogACT uses native cognition attention as its disclosed gate.",
        "  Speedup includes all auxiliary computation, not an assumed actuation overlap.",
        "- FastV uses the released HF last-query variant; DART is token reduction, not",
        "  speculative decoding. SparseVLM is original v1 rank-adaptive pruning/recycling,",
        "  not SparseVLM+ or the released fixed-budget variant.",
        "- SQAP-VLA is deferred pending clarification of contradictory activation",
        "  quantization granularity; it has no runnable proxy and is not an implemented method.",
        "- SP-VLA scheduling-only rows exclude pruning; pruning + scheduling rows",
        "  include both mechanisms in one implementation. Fitted actions change query",
        "  counts; compare full replay cost and evaluate success loss separately.",
        "- VOTE means trajectory ensemble only: no ACT-token/head retraining and no",
        "  claim to reproduce the full paper's acceleration. H=1 costs substantially",
        "  more than native H=8 even if the voting overhead itself is small.",
        "- AWQ's FP16-kernel row is mixed precision, not the default BF16 kernel;",
        "  boundary casts are timed. Accuracy equivalence is not established.",
        "- Cache regressions, failed smoke episodes and all precision variants remain",
        "  visible. Historical repeats and superseded rows are not new algorithms.",
        "- TEAM-VLA and VLA-IAP are independent paper-based implementations with",
        "  explicit unresolved author-code conventions; consult their method READMEs.",
        "  Their rows are not verified reproductions of the published implementations.",
        "- EfficientVLA token core excludes layer pruning and DiT caching; FlashVLA",
        "  ICS core excludes unresolved action reuse. ST-Merge is a paper-based spatial",
        "  VLA port with shared two-encoder grouping and a timed dense projector bridge.",
        "- VLA-InfoEntropy skips visual QUERY rows while retaining visual KV context.",
        "  It is not context pruning; its query counts do not belong in the retention",
        "  table. Paper-based methods disclose reconstruction choices in their READMEs.",
        "- CogACT native parity does not verify the unidentified training fork's",
        "  evaluation conventions. Its original four-episode Long failure is retained.",
        "- Secondary CogACT native rechecks had observed thermal throttling; do not",
        "  use them as stable references. The category-example cohort retained seven",
        "  sparse thermal-flag samples; its late native drift was 0.435%. VOTE's tiny",
        "  difference is not significant evidence of acceleration or regression.",
        "- No native sampling accelerator is implemented. Do not relabel fewer DDIM",
        "  steps as a faithful paper method. Full success campaigns remain unmeasured.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    for name in ("run-dir", "baseline-dir"):
        export.add_argument(f"--{name}", type=Path, required=True)
    export.add_argument("--records-dir", type=Path, default=Path("outputs/result_records"))
    export.add_argument("--smoke-dir", type=Path)
    export.add_argument("--supersedes", action="append", default=[], metavar="RUN_ID")
    for name in ("section", "label", "preset"):
        export.add_argument(f"--{name}", required=True)
    table = commands.add_parser("render")
    table.add_argument("--records-dir", type=Path, default=Path("outputs/result_records"))
    table.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        if Path(args.preset).is_absolute() or ".." in Path(args.preset).parts:
            raise ValueError("Preset links must be repository-relative")
        record = make_record(
            args.run_dir, args.baseline_dir, args.section, args.label, args.preset, args.smoke_dir
        )
        annotate_supersedes(record, args.supersedes, args.records_dir)
        args.records_dir.mkdir(parents=True, exist_ok=True)
        destination = args.records_dir / f"{args.run_dir.name}.json"
        if destination.exists() and read_json(destination) != record:
            raise ValueError("Refusing to replace a different published evidence record")
        destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    else:
        records = [read_json(path) for path in sorted(args.records_dir.glob("*.json"))]
        if not records:
            raise ValueError("No evidence records; obtain the local archive before regenerating")
        args.output.write_text(render(records))


if __name__ == "__main__":
    main()
