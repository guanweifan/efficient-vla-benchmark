"""One warmed batch-one pass, including transforms/transfers; no policy speed claim."""

import json
import time
from pathlib import Path

import numpy as np

from .core import Packet, actions_checked


def evaluate(codec, corpus, output, *, warmup=10):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refuse to overwrite codec evidence")
    rows = corpus.rows("evaluation")
    if not rows:
        raise ValueError("No evaluation windows")
    output.mkdir(parents=True)
    identity = codec.metadata()
    start = {
        "protocol": "libero_codec_reconstruction_v1",
        "corpus_sha256": corpus.sha256,
        "horizon": corpus.manifest["horizon"],
        "codec": identity,
        "warmup": warmup,
        "passes": 1,
        "batch_size": 1,
        "timing_scope": "CPU raw actions -> host integer tokens -> CPU raw reconstruction",
        "claim_limit": "In-domain reconstruction, not unseen pretraining data or policy success",
    }
    (output / "metadata.json").write_text(json.dumps(start, indent=2, sort_keys=True) + "\n")
    for i in range(warmup):
        action, context = corpus.window(rows[i % len(rows)])
        packet = codec.encode(action, context)
        actions_checked(codec.decode(packet, context), len(action))
        codec.synchronize()
    results = []
    pass_start = time.time()
    with (output / "windows.jsonl").open("w") as stream:
        for row in rows:
            action, context = corpus.window(row)
            result = {k: row[k] for k in ("key", "suite", "episode_index", "offset")}
            try:
                codec.synchronize()
                encode_start = time.time()
                t0 = time.perf_counter()
                packet = codec.encode(action.copy(), context)
                codec.synchronize()
                t1 = time.perf_counter()
                encode_end = time.time()
                # Only the discrete payload and declared initial observation reach decode.
                packet = Packet(packet.ids.copy(), packet.vocabulary, packet.horizon)
                decode_start = time.time()
                t2 = time.perf_counter()
                recon = actions_checked(codec.decode(packet, context), len(action))
                codec.synchronize()
                t3 = time.perf_counter()
                decode_end = time.time()
                error = recon.astype(np.float64) - action
                result.update(
                    status="completed",
                    encode_ms=(t1 - t0) * 1000,
                    decode_ms=(t3 - t2) * 1000,
                    encode_window={"start": encode_start, "end": encode_end},
                    decode_window={"start": decode_start, "end": decode_end},
                    mse=float(np.mean(error**2)),
                    per_dimension_mse=np.mean(error**2, axis=0).tolist(),
                    mae=float(np.mean(np.abs(error))),
                    gripper_sign_agreement=float(np.mean((recon[:, 6] > 0) == (action[:, 6] > 0))),
                    tokens=packet.token_count,
                    nominal_payload_bits=packet.nominal_bits,
                )
            except Exception as exc:
                result.update(status="error", error_type=type(exc).__name__, error=str(exc))
            stream.write(json.dumps(result, sort_keys=True) + "\n")
            stream.flush()
            results.append(result)
    pass_end = time.time()
    (output / "measurement_window.json").write_text(
        json.dumps({"start": pass_start, "end": pass_end}, sort_keys=True) + "\n"
    )
    if codec.metadata() != identity:
        raise ValueError("Codec/calibration identity changed during evaluation")
    good = [r for r in results if r["status"] == "completed"]
    summary = {
        "windows": len(results),
        "errors": len(results) - len(good),
        "complete": len(good) == len(rows),
        "corpus_sha256": corpus.sha256,
    }
    # No selective averages from only the successfully decoded windows.
    if summary["complete"]:
        for key in (
            "encode_ms",
            "decode_ms",
            "mse",
            "mae",
            "gripper_sign_agreement",
            "tokens",
            "nominal_payload_bits",
        ):
            summary[key] = float(np.mean([r[key] for r in good]))
        summary["per_dimension_mse"] = np.mean(
            [r["per_dimension_mse"] for r in good], axis=0
        ).tolist()
        summary["tokens_per_frame"] = summary["tokens"] / corpus.manifest["horizon"]
        summary["nominal_bits_per_frame"] = (
            summary["nominal_payload_bits"] / corpus.manifest["horizon"]
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
