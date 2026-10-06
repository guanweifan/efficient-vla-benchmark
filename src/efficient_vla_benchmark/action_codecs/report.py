"""Validate codec evidence separately from policy success and speedup records."""

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .corpus import digest


def load_result(directory):
    directory = Path(directory)
    names = (
        "metadata.json",
        "summary.json",
        "provenance.json",
        "quality.json",
        "windows.jsonl",
        "measurement_window.json",
        "telemetry.jsonl",
    )
    metadata, summary, provenance, quality = (
        json.loads((directory / name).read_text()) for name in names[:4]
    )
    rows = [json.loads(line) for line in (directory / "windows.jsonl").read_text().splitlines()]
    if (
        metadata["protocol"] != "libero_codec_reconstruction_v1"
        or metadata["warmup"] != 10
        or metadata["passes"] != 1
        or metadata["batch_size"] != 1
        or not summary["complete"]
        or summary["errors"] != 0
        or not rows
        or len(rows) != summary["windows"]
        or len({row["key"] for row in rows}) != len(rows)
        or any(row["status"] != "completed" for row in rows)
        or metadata["corpus_sha256"] != summary["corpus_sha256"]
    ):
        raise ValueError("Incomplete or inconsistent codec reconstruction evidence")
    if (
        not quality["valid"]
        or not quality["source_unchanged"]
        or not provenance["source_unchanged"]
        or provenance["source"]["dirty"]
        or quality["source"] != provenance["source"]["commit"]
        or (
            metadata["codec"]["device"] != "cpu"
            and (quality["thermal_flags"] or quality["samples_in_pass"] < 2)
        )
    ):
        raise ValueError("Codec latency has no accepted frozen-source quality check")
    if digest(directory / "telemetry.jsonl") != quality["telemetry_sha256"]:
        raise ValueError("Codec telemetry identity differs from its quality check")
    window = json.loads((directory / "measurement_window.json").read_text())
    samples = [
        json.loads(line) for line in (directory / "telemetry.jsonl").read_text().splitlines()
    ]
    measured = [row for row in samples if window["start"] <= row["time"] <= window["end"]]
    if len(measured) != quality["samples_in_pass"] or (
        metadata["codec"]["device"] != "cpu"
        and any(
            row["returncode"]
            or any(part.strip() != "Not Active" for part in row["reading"].split(",")[-2:])
            for row in measured
        )
    ):
        raise ValueError("Codec thermal check disagrees with raw telemetry")
    for row in rows:
        e, d = row["encode_window"], row["decode_window"]
        if not window["start"] <= e["start"] <= e["end"] <= d["start"] <= d["end"] <= window["end"]:
            raise ValueError("Codec measurement intervals differ")
    for key in (
        "encode_ms",
        "decode_ms",
        "mse",
        "mae",
        "gripper_sign_agreement",
        "tokens",
        "nominal_payload_bits",
    ):
        values = [row[key] for row in rows]
        if any(not math.isfinite(value) or value < 0 for value in values) or not math.isclose(
            sum(values) / len(values), summary[key], rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError("Codec summary differs from per-window evidence")
    if not np.allclose(
        np.mean([row["per_dimension_mse"] for row in rows], axis=0),
        summary["per_dimension_mse"],
        rtol=1e-9,
        atol=1e-12,
    ):
        raise ValueError("Codec per-dimension summary differs")
    for result_key, raw_key in (
        ("tokens_per_frame", "tokens"),
        ("nominal_bits_per_frame", "nominal_payload_bits"),
    ):
        if not math.isclose(summary[result_key], summary[raw_key] / metadata["horizon"]):
            raise ValueError("Codec rate uses a different horizon")
    return {
        "run_id": directory.name,
        "metadata": metadata,
        "summary": summary,
        "provenance": provenance,
        "quality": quality,
        "suite_windows": dict(Counter(row["suite"] for row in rows)),
        "window_keys_sha256": hashlib.sha256(
            json.dumps([row["key"] for row in rows]).encode()
        ).hexdigest(),
        "files_sha256": {name: digest(directory / name) for name in names},
    }


def render(records, auxiliary=()):
    if not records:
        return ""
    if len({r["metadata"]["corpus_sha256"] for r in records}) != 1:
        raise ValueError("Codec table requires one shared corpus")
    if len({r["metadata"]["horizon"] for r in records}) != 1:
        raise ValueError("Codec table requires one shared horizon")
    if (
        len(
            {
                r["metadata"]["codec"].get("comparison_scope", "command_reconstruction")
                for r in records
            }
        )
        != 1
    ):
        raise ValueError("Codec table requires one representation scope")
    if len({r.get("window_keys_sha256") for r in records}) != 1:
        raise ValueError("Codec table requires identical ordered evaluation windows")
    if len({json.dumps(r["suite_windows"], sort_keys=True) for r in records}) != 1:
        raise ValueError("Codec table requires identical suite coverage")
    first = records[0]["metadata"]
    lines = [
        "",
        "## Independent action codec reconstruction",
        "",
        f"Frozen corpus `{first['corpus_sha256']}`; {first['horizon']}-frame windows, "
        "one per episode, disjoint calibration/evaluation episodes; "
        f"{len(records[0]['suite_windows'])} evaluated suites.",
        "This is in-domain demonstration reconstruction, not policy success or VLA speedup.",
        "Latency includes representation transforms and CPU/device transfers; native CPU "
        "and GPU implementations are labeled separately. Nominal bits are not entropy-coded size.",
        "Rate variants are retained rate–distortion points, not extra methods. "
        "Representation transfers and state-conditioned results are separate groups.",
        "Pretraining overlap is not excluded; native block sizes/receptive fields differ.",
        "",
        "| Codec / preset | Device | Windows | Raw MSE | Raw MAE | Gripper sign | "
        "Tokens/frame | Bits/frame | Encode ms/window | Decode ms/window |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        s, m = record["summary"], record["metadata"]["codec"]
        params = record["provenance"]["config"]["parameters"]
        label = m["codec"] + (
            " / " + ", ".join(f"{k}={v}" for k, v in params.items()) if params else ""
        )
        device = m.get("gpu") or record["quality"].get("cpu_model", m["device"])
        lines.append(
            f"| {label} | {device} | {s['windows']} | {s['mse']:.7f} | {s['mae']:.5f} | "
            f"{s['gripper_sign_agreement']:.4f} | {s['tokens_per_frame']:.3f} | "
            f"{s['nominal_bits_per_frame']:.3f} | {s['encode_ms']:.3f} | {s['decode_ms']:.3f} |"
        )
    lines += ["", "| Codec run | Measurement commit | Summary SHA256 |", "|---|---|---|"]
    for r in records:
        lines.append(
            f"| `{r['run_id']}` | `{r['provenance']['source']['commit']}` | "
            f"`{r['files_sha256']['summary.json']}` |"
        )
    if auxiliary:
        if any(r["metadata"]["corpus_sha256"] != first["corpus_sha256"] for r in auxiliary):
            raise ValueError("Auxiliary codec device checks use a different corpus")
        lines += [
            "",
            "<details>",
            "<summary>Codec device checks (auxiliary, not default rows)</summary>",
            "",
            "The small spline workload is faster on native CPU; both devices use four OpenMP "
            "threads and the same corpus. Retain the CUDA comparison, not a second primary.",
            "",
            "| Run | Device | Raw MSE | Encode / decode ms per window | Summary SHA256 |",
            "|---|---|---:|---:|---|",
        ]
        for r in auxiliary:
            s, m = r["summary"], r["metadata"]["codec"]
            lines.append(
                f"| `{r['run_id']}` | {m.get('gpu') or m['device']} | {s['mse']:.7f} | "
                f"{s['encode_ms']:.3f} / {s['decode_ms']:.3f} | "
                f"`{r['files_sha256']['summary.json']}` |"
            )
        lines += ["", "</details>"]
    return "\n".join(lines) + "\n"


SCOPES = {
    "command_reconstruction": "Raw-command reconstruction",
    "coordinate_transfer": "Quantizer coordinate-transfer diagnostic",
    "state_conditioned": "State-conditioned representation-transfer diagnostic",
}


def render_groups(records, auxiliary=()):
    """Never compare different horizons or conditional representations as one table."""
    if not records:
        if auxiliary:
            raise ValueError("Auxiliary codec checks require primary evidence")
        return ""
    groups = {}
    for record in records:
        m = record["metadata"]
        scope = m["codec"].get("comparison_scope", "command_reconstruction")
        if scope not in SCOPES:
            raise ValueError("Unknown codec comparison scope")
        groups.setdefault((m["horizon"], m["corpus_sha256"], scope), []).append(record)
    sections = ["", "## Independent action codec reconstruction", ""]
    for (horizon, corpus, scope), members in groups.items():
        checks = [
            r
            for r in auxiliary
            if r["metadata"]["corpus_sha256"] == corpus
            and r["metadata"]["codec"].get("comparison_scope", "command_reconstruction") == scope
        ]
        body = render(members, checks).split("## Independent action codec reconstruction\n", 1)[1]
        sections += [f"### {SCOPES[scope]} — H{horizon}", body]
    for record in auxiliary:
        m = record["metadata"]
        key = (
            m["horizon"],
            m["corpus_sha256"],
            m["codec"].get("comparison_scope", "command_reconstruction"),
        )
        if key not in groups:
            raise ValueError("Auxiliary codec check has no matching comparison group")
    sections += [
        "Only compare rows within a group. H16 is a frozen prefix of H40 with the same "
        "episode identities, but its calibration statistics and workload differ. "
        "Conditional/context-transfer diagnostics are not a codec leaderboard.",
        "",
    ]
    return "\n".join(sections)


def replace_section(original, section):
    """Preserve both policy sections around the standalone codec section."""
    marker = "## Independent action codec reconstruction"
    if original.count(marker) > 1:
        raise ValueError("Ambiguous existing codec section")
    if marker not in original:
        return original.rstrip("\n") + "\n\n" + section.lstrip("\n")
    prefix, old = original.split(marker, 1)
    following = re.search(r"\n## [^#]", old)
    suffix = old[following.start() :] if following else ""
    return prefix + section.lstrip("\n").rstrip("\n") + "\n" + suffix


def main():
    parser = argparse.ArgumentParser(description="Replace only the codec result section.")
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--auxiliary-run", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    args = parser.parse_args()
    records = [load_result(path) for path in args.run]
    auxiliary = [load_result(path) for path in args.auxiliary_run]
    if len({r["run_id"] for r in records + auxiliary}) != len(records + auxiliary):
        raise ValueError("Duplicate codec result")
    section = render_groups(records, auxiliary)
    original = args.output.read_text() if args.output.exists() else ""
    args.output.write_text(replace_section(original, section))
    args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_output.write_text(
        json.dumps({"records": records, "auxiliary": auxiliary}, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
