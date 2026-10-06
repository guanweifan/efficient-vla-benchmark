"""Frozen action-window corpus with episode-level calibration/evaluation separation."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .core import Context, actions_checked


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def write_corpus(directory, rows, *, horizon, provenance):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("Refuse to overwrite a frozen corpus")
    identities, content, manifests = set(), set(), []
    arrays = {}
    for i, row in enumerate(rows):
        identity = (row["suite"], int(row["episode_index"]))
        if identity in identities:
            raise ValueError("One window per episode; calibration/test episodes cannot overlap")
        identities.add(identity)
        if row["split"] not in ("calibration", "evaluation"):
            raise ValueError("Unknown codec corpus split")
        action = actions_checked(row["actions"], horizon).copy()
        state = Context(row["suite"], row["initial_state"]).initial_state.copy()
        checksum = hashlib.sha256(action.tobytes() + state.tobytes()).hexdigest()
        if checksum in content:
            raise ValueError("Duplicate action/state windows across the corpus")
        content.add(checksum)
        key = f"window_{i:05d}"
        arrays[key + "_actions"] = action
        arrays[key + "_state"] = state
        manifests.append(
            {k: row[k] for k in ("suite", "episode_index", "offset", "split")}
            | {"key": key, "content_sha256": checksum}
        )
    if not manifests or {r["split"] for r in manifests} != {"calibration", "evaluation"}:
        raise ValueError("Both nonempty splits are compulsory")
    directory.mkdir(parents=True)
    np.savez_compressed(directory / "windows.npz", **arrays)
    manifest = {
        "schema": 1,
        "kind": "libero_action_reconstruction",
        "horizon": horizon,
        "action_convention": "raw OSC delta commands; gripper -1 open / +1 close",
        "physical_hz": None,
        "provenance": provenance,
        "windows": manifests,
        "arrays_sha256": digest(directory / "windows.npz"),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


class Corpus:
    def __init__(self, directory):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        self.sha256 = digest(directory / "manifest.json")
        if self.manifest["schema"] != 1 or self.manifest["kind"] != "libero_action_reconstruction":
            raise ValueError("Unsupported action corpus")
        if digest(directory / "windows.npz") != self.manifest["arrays_sha256"]:
            raise ValueError("Changed corpus arrays")
        self.arrays = dict(np.load(directory / "windows.npz", allow_pickle=False))
        identities, checksums = set(), set()
        for row in self.manifest["windows"]:
            identity = (row["suite"], row["episode_index"])
            if identity in identities or row["split"] not in ("calibration", "evaluation"):
                raise ValueError("Duplicate/invalid episode assignment")
            identities.add(identity)
            action, context = self.window(row)
            checksum = hashlib.sha256(
                action.tobytes() + context.initial_state.tobytes()
            ).hexdigest()
            if checksum != row["content_sha256"] or checksum in checksums:
                raise ValueError("Changed/duplicate corpus window")
            checksums.add(checksum)

    def window(self, row):
        action = actions_checked(self.arrays[row["key"] + "_actions"], self.manifest["horizon"])
        return action.copy(), Context(row["suite"], self.arrays[row["key"] + "_state"])

    def rows(self, split):
        return [row for row in self.manifest["windows"] if row["split"] == split]


def derive_prefix(source, output, horizon):
    """Freeze a matched shorter workload, never truncate inside a method adapter."""
    corpus = Corpus(source)
    if not 1 <= horizon < corpus.manifest["horizon"]:
        raise ValueError("Derived horizon must be shorter than the frozen source")
    rows = []
    for row in corpus.manifest["windows"]:
        actions, context = corpus.window(row)
        rows.append(
            {k: row[k] for k in ("suite", "episode_index", "offset", "split")}
            | {"actions": actions[:horizon], "initial_state": context.initial_state}
        )
    return write_corpus(
        output,
        rows,
        horizon=horizon,
        provenance={
            "source_corpus_sha256": corpus.sha256,
            "source_horizon": corpus.manifest["horizon"],
            "selection": "fixed prefix; identical episode IDs, offsets and split assignments",
            "limit": "Compare only against methods rerun on this identical derived corpus",
        },
    )
