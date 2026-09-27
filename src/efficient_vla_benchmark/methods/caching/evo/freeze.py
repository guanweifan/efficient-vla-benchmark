"""Recheck every search rollout and budget before publishing a deployable layout."""

import json
from collections import Counter
from pathlib import Path

from efficient_vla_benchmark.core.provenance import digest_file, stable_hash
from efficient_vla_benchmark.evaluation.libero.independent import load_population
from efficient_vla_benchmark.evaluation.libero.runner import SUITES

from .search import SearchBudget, confirm_target, validate_layout


def read_run(root, run_id, population, group, binding, candidate=None):
    if Path(run_id).name != run_id:
        raise ValueError("Search run IDs must be relative basenames")
    run = root / "runs" / run_id
    metadata = json.loads((run / "metadata.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    records = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
    selected = [e for e in population["episodes"] if e["group"] == group]

    def key(row):
        return row["suite"], row["task_id"], row["episode_id"]

    expected = {key(e["context"]): e for e in selected}
    if (
        metadata.get("kind") != "independent_search_rollout"
        or metadata.get("population_sha256") != population["sha256"]
        or metadata.get("group") != group
        or metadata.get("episodes") != selected
        or metadata["policy"]["binding"] != binding
        or metadata["policy"]["execution_horizon"] != 8
        or metadata["policy"]["candidate_sha256"] != (stable_hash(candidate) if candidate else None)
        or len(records) != len(selected)
        or len({key(r) for r in records}) != len(records)
        or {key(r) for r in records} != set(expected)
    ):
        raise ValueError("Search evidence differs from frozen policy/population/group")
    for record in records:
        entry = expected[key(record)]
        if (
            record["status"] != "completed"
            or type(record["success"]) is not bool
            or record["seed"] != entry["context"]["seed"]
            or record["initial_state_sha256"] != entry["state_sha256"]
        ):
            raise ValueError("Search errors, state/seed substitutions or missing outcomes")
    successes = sum(r["success"] for r in records)
    if (
        summary["episodes"] != len(records)
        or summary["successes"] != successes
        or summary["errors"]
    ):
        raise ValueError("Search summary differs from the actual outcomes")
    return dict(
        run=run_id,
        episodes=len(records),
        successes=successes,
        success_rate=successes / len(records),
        runtime_source_sha256=metadata["source"]["runtime_source_sha256"],
        metadata_sha256=digest_file(run / "metadata.json"),
        episodes_sha256=digest_file(run / "episodes.jsonl"),
        wall_seconds=summary["wall_seconds"],
    )


def freeze_search(root, population_root, output):
    if output.exists():
        raise ValueError("Refusing to overwrite a frozen search artifact")
    report = json.loads((root / "search.json").read_text())
    population = load_population(population_root)
    for group, per_suite in (("reference", 10), ("quick", 5), ("confirmation", 10)):
        entries = [e["context"] for e in population["episodes"] if e["group"] == group]
        if Counter(e["suite"] for e in entries) != dict.fromkeys(SUITES, per_suite):
            raise ValueError("Every search phase must cover all four suites equally")
        if len({(e["suite"], e["task_id"]) for e in entries}) != len(entries):
            raise ValueError("Search phase repeats tasks instead of the frozen task coverage")
    binding = json.loads((root / "runs" / report["reference_run"] / "binding.json").read_text())
    if binding["population_sha256"] != population["sha256"]:
        raise ValueError("Native reference used another population")
    if report["limits"] != {"max_episodes": 640, "max_candidates": 24}:
        raise ValueError("This search artifact exceeds its explicitly reviewed budget contract")
    budget = SearchBudget(**report["limits"])
    for item in report["reservations"]:
        budget.reserve(item["id"], item["kind"], item["episodes"])
    reference = read_run(root, report["reference_run"], population, "reference", binding)
    if reference["episodes"] != 40:
        raise ValueError("Independent native reference must cover all forty tasks")
    checked = {reference["run"]: ("reference", reference)}
    candidates, sources = {}, {reference["runtime_source_sha256"]}
    layouts = {}
    for candidate in report["candidates"]:
        identity = candidate["id"]
        if identity in candidates:
            raise ValueError("Duplicate search candidate")
        descriptor = {k: candidate[k] for k in ("binding", "budget", "layout")}
        layout = validate_layout(
            candidate["layout"], binding["num_blocks"], binding["num_steps"], candidate["budget"]
        )
        if candidate["binding"] != binding:
            raise ValueError("Search candidates use different native model bindings")
        quick = read_run(root, candidate["quick_run"], population, "quick", binding, descriptor)
        if quick["episodes"] != 20:
            raise ValueError("Quick rollout must cover five tasks per suite")
        evidence = dict(quick=quick)
        checked[quick["run"]] = ("candidate", quick)
        sources.add(quick["runtime_source_sha256"])
        confirmation = None
        if candidate.get("confirmation_run"):
            confirmation = read_run(
                root, candidate["confirmation_run"], population, "confirmation", binding, descriptor
            )
            if confirmation["episodes"] != 40:
                raise ValueError("Confirmation must cover all forty tasks")
            checked[confirmation["run"]] = ("confirmation", confirmation)
            sources.add(confirmation["runtime_source_sha256"])
            evidence["confirmation"] = confirmation
        candidates[identity] = {"layout": layout, "evidence": evidence}
        if identity in report["selected"].values():
            if confirmation is None:
                raise ValueError("Selected layout lacks independent confirmation evidence")
            _, accepted = confirm_target(
                quick["success_rate"], reference["success_rate"], confirmation["success_rate"]
            )
            layouts[str(candidate["budget"])] = dict(
                layout=layout,
                candidate=identity,
                target_confirmed=accepted,
                termination="target_confirmed" if accepted else "bounded_search_target_not_met",
                quick=evidence["quick"],
                confirmation=evidence["confirmation"],
            )
    if len(sources) != 1:
        raise ValueError("Search worker source changed during the campaign")
    if set(layouts) != set(report["selected"]) or any(
        v not in candidates for v in report["selected"].values()
    ):
        raise ValueError("Selected schedule does not exist or is inconsistent")
    if len({(c["budget"], stable_hash(c["layout"])) for c in report["candidates"]}) != len(
        candidates
    ):
        raise ValueError("A canonical schedule was reevaluated instead of deduplicated")
    reservations = {r["id"]: (r["kind"], r["episodes"]) for r in budget.reservations}
    if reservations != {k: (kind, evidence["episodes"]) for k, (kind, evidence) in checked.items()}:
        raise ValueError("Launched search budget and complete retained outcomes differ")
    manifest = dict(
        schema_version=1,
        algorithm="evo",
        backend="cogact",
        complete=True,
        **binding,
        layouts=layouts,
        reference=reference,
        uses_dev_or_final_feedback=False,
        search_episodes=sum(r["episodes"] for r in budget.reservations),
        search_candidates=len(candidates),
        search_limits=report["limits"],
        search_sha256=digest_file(root / "search.json"),
        prior_sha256=digest_file(root / "runs" / report["reference_run"] / "prior.npz"),
        runtime_source_sha256=next(iter(sources)),
        interpretation="bounded global four-suite search; target failure is not convergence",
    )
    manifest["sha256"] = stable_hash(manifest)
    output.mkdir(parents=True)
    (output / "search.json").write_bytes((root / "search.json").read_bytes())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
