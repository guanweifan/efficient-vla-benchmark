import json

import pytest

from efficient_vla_benchmark.action_codecs.corpus import digest
from efficient_vla_benchmark.action_codecs.evaluation import evaluate
from efficient_vla_benchmark.action_codecs.report import load_result, render
from tests.test_action_codecs import DummyCodec, fixture_corpus


def evidence(tmp_path):
    class CPU(DummyCodec):
        def metadata(self):
            return {"codec": "test_codec", "device": "cpu"}

    corpus, _ = fixture_corpus(tmp_path)
    run = tmp_path / "run"
    evaluate(CPU(), corpus, run)
    (run / "telemetry.jsonl").write_text("")
    (run / "quality.json").write_text(
        json.dumps(
            {
                "valid": True,
                "source_unchanged": True,
                "source": "a" * 40,
                "samples_in_pass": 0,
                "thermal_flags": [],
                "telemetry_sha256": digest(run / "telemetry.jsonl"),
            }
        )
    )
    (run / "provenance.json").write_text(
        json.dumps(
            {
                "source": {"dirty": False, "commit": "a" * 40},
                "source_unchanged": True,
                "config": {"parameters": {}},
            }
        )
    )
    return run


def test_codec_record_is_independent_and_checksummed(tmp_path):
    run = evidence(tmp_path)
    record = load_result(run)
    assert record["files_sha256"]["telemetry.jsonl"] == digest(run / "telemetry.jsonl")
    table = render([record])
    assert "not policy success" in table and "test_codec" in table
    assert "Speedup |" not in table
    assert "Codec device checks" in render([record], [record])
    altered = dict(record, metadata=dict(record["metadata"], corpus_sha256="b" * 64))
    with pytest.raises(ValueError, match="different corpus"):
        render([record], [altered])


@pytest.mark.parametrize("change", ["summary", "quality", "telemetry", "provenance"])
def test_codec_record_rejects_tampering(tmp_path, change):
    run = evidence(tmp_path)
    if change == "telemetry":
        (run / "telemetry.jsonl").write_text("{}\n")
    else:
        path = run / f"{change}.json"
        data = json.loads(path.read_text())
        if change == "summary":
            data["mse"] += 1
        elif change == "quality":
            data["valid"] = False
        else:
            data["source"]["dirty"] = True
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_result(run)
