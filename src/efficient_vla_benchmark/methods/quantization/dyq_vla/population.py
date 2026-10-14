"""Checksummed independent calibration population; evaluation data are rejected."""

import json

import numpy as np

from efficient_vla_benchmark.core.provenance import digest_file, stable_hash
from efficient_vla_benchmark.evaluation.traces import load_trace

from .dispatcher import DyQDispatcher


def manifest(root, backend):
    data = json.loads((root / "population.json").read_text())
    identity = data.get("sha256")
    if stable_hash({k: v for k, v in data.items() if k != "sha256"}) != identity:
        raise ValueError("Independent population checksum mismatch")
    if (
        data.get("kind") != "independent_native_quantization_population"
        or data.get("total_attempts") != sum(len(rows) for rows in data.get("records", {}).values())
        or data.get("failures_replaced") is not False
    ):
        raise ValueError("DyQ requires frozen independent native trajectories, not Dev/Final")
    records = data["records"][backend]
    if len(records) != 40 or {r["flat"] for r in records} != set(range(40)):
        raise ValueError("Incomplete independent task coverage")
    return data


def samples(root, backend, config):
    data = manifest(root, backend)
    for row in data["records"][backend]:
        if not row["success"]:
            continue
        shard = root / backend / row["shard"]
        for field in ("trace", "actions"):
            path = shard / row[field]
            if (
                not path.resolve().is_relative_to(root.resolve())
                or digest_file(path) != row[field + "_sha256"]
            ):
                raise ValueError("Independent trace identity mismatch")
        context, observations, steps = load_trace(shard / row["trace"], True)
        with np.load(shard / row["actions"], allow_pickle=False) as values:
            actions = values["actions"].copy()
            queries = values["queries"].copy()
        if len(actions) != len(steps) or len(queries) != len(steps):
            raise ValueError("Independent action/observation alignment mismatch")
        candidates = np.flatnonzero(queries)
        selected = set(
            candidates[
                np.linspace(
                    0, len(candidates) - 1, min(len(candidates), config.frames_per_trajectory)
                ).astype(int)
            ].tolist()
        )
        state = DyQDispatcher(config, dict(theta_2_4=-1, theta_4_8=-1, theta_fp=config.theta_fp))
        for index, observation in enumerate(observations):
            if index in selected:
                yield (
                    context,
                    observation,
                    dict(
                        flat=row["flat"],
                        step=index,
                        trace_sha256=row["trace_sha256"],
                        sensitivity=state.last_sensitivity,
                    ),
                )
            state.observe(actions[index])
