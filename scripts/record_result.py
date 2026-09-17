"""Export audited portable evidence and render the per-category measurement ledger."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


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
    return {
        "run_id": directory.name,
        "metadata": metadata,
        "summary": summary,
        "token_diagnostics": token_evidence(rows),
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


def render(records):
    lines = [
        "# Algorithm measurement ledger",
        "",
        "Generated from the versioned JSON evidence in `results/`. This is a development",
        "measurement ledger, not a final paper comparison or a formal success-rate table.",
        "",
        "Backend, hardware and protocol are recorded per row below. Policy timing",
        "includes preprocessing, transfer, all method/control overhead and postprocessing,",
        "but not model loading, trace I/O or simulation. Fixed replay is not closed-loop",
        "task completion time. Different protocol cohorts must not be pooled.",
        "",
        "Speedup = paired native baseline total replay time / method total replay time.",
        "H is the executed prefix, distinct from the backend's full prediction length.",
        "The present OFT VOTE preset uses its matched H=1 baseline; the other OFT presets",
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
            "| Method / configuration | Backend | H | Queries / replacements | "
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
                f"| [{record['label']}](results/{run['run_id']}.json) | "
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
            "Query-weighted measured counts, including warmup/refresh queries; these",
            "are not configured keep ratios or FLOPs. Final counts do not describe",
            "earlier-layer cost. Per-layer sequence lengths remain in raw diagnostics.",
            "",
            "| Method | Final mean / min / max | Early mean | Mode counts (true/observed) |",
            "|---|---:|---:|---|",
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
                f"| {record['label']} | {final['mean']:.1f} / {final['min']} / "
                f"{final['max']} | {early_text} | {modes} |"
            )
        lines.append("")
    lines += [
        "## Traceability",
        "",
        "Raw files remain under `outputs/<run ID>/`; the linked JSON preserves their",
        "hashes, full metadata, summary, paired baseline and optional smoke records.",
        "Model/trace/artifact identities, precision, kernel, dependencies, source",
        "fingerprint, dirty flag and parameters are in those records. A smoke from an",
        "earlier source revision is explicitly marked, never silently relabeled.",
        "",
        "| Method | Preset | Timing commit | Baseline run |",
        "|---|---|---|---|",
    ]
    for record in records:
        commit = record["run"]["metadata"]["source"]["commit"] or "unknown"
        dirty = " (dirty)" if record["run"]["metadata"]["source"]["dirty"] else ""
        lines.append(
            f"| {record['label']} | [{Path(record['preset']).stem}](../{record['preset']}) | "
            f"`{commit[:12]}`{dirty} | `{record['baseline']['run_id']}` |"
        )
    lines += [
        "",
        "## Timing protocols",
        "",
        "| Method | Hardware | Measured frames | Repeats | Warmup queries |",
        "|---|---|---:|---:|---:|",
    ]
    for record in records:
        run = record["run"]
        meta = run["metadata"]
        hardware = meta["hardware"].get("gpu", meta["hardware"].get("runtime", "unknown"))
        lines.append(
            f"| {record['label']} | {hardware} | "
            f"{run['summary']['all_control_steps']['count']} | {meta['protocol']['repeats']} | "
            f"{meta['protocol']['warmup_queries']} |"
        )
    lines += [
        "",
        "## Interpretation boundaries",
        "",
        "- SP-VLA scheduling-only rows exclude pruning; pruning + scheduling rows",
        "  include both mechanisms in one implementation. Fitted actions change query",
        "  counts; compare full replay cost and evaluate success loss separately.",
        "- VOTE means trajectory ensemble only: no ACT-token/head retraining and no",
        "  claim to reproduce the full paper's acceleration. H=1 costs substantially",
        "  more than native H=8 even if the voting overhead itself is small.",
        "- AWQ's FP16-kernel row is mixed precision, not the default BF16 kernel;",
        "  boundary casts are timed. Accuracy equivalence is not established.",
        "- Cache regressions, failed smoke episodes and all precision variants remain",
        "  visible. Historical runs in other validation reports are separate cohorts.",
        "- TEAM-VLA and VLA-IAP are independent paper-based implementations with",
        "  explicit unresolved author-code conventions; consult their method READMEs.",
        "  Their rows are not verified reproductions of the published implementations.",
        "- EfficientVLA token core excludes layer pruning and DiT caching; FlashVLA",
        "  ICS core excludes unresolved action reuse. ST-Merge is a paper-based spatial",
        "  VLA port with shared two-encoder grouping and a timed dense projector bridge.",
        "- VLA-InfoEntropy skips visual QUERY rows while retaining visual KV context.",
        "  It is not context pruning; its query counts do not belong in the retention",
        "  table. Paper-based methods disclose reconstruction choices in their READMEs.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    for name in ("run-dir", "baseline-dir", "records-dir"):
        export.add_argument(f"--{name}", type=Path, required=True)
    export.add_argument("--smoke-dir", type=Path)
    for name in ("section", "label", "preset"):
        export.add_argument(f"--{name}", required=True)
    table = commands.add_parser("render")
    table.add_argument("--records-dir", type=Path, required=True)
    table.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        if Path(args.preset).is_absolute() or ".." in Path(args.preset).parts:
            raise ValueError("Preset links must be repository-relative")
        record = make_record(
            args.run_dir, args.baseline_dir, args.section, args.label, args.preset, args.smoke_dir
        )
        args.records_dir.mkdir(parents=True, exist_ok=True)
        destination = args.records_dir / f"{args.run_dir.name}.json"
        if destination.exists() and read_json(destination) != record:
            raise ValueError("Refusing to replace a different published evidence record")
        destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    else:
        records = [read_json(path) for path in sorted(args.records_dir.glob("*.json"))]
        if not records:
            raise ValueError("No evidence records")
        args.output.write_text(render(records))


if __name__ == "__main__":
    main()
