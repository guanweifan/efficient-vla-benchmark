import json

import numpy as np
import pytest

from efficient_vla_benchmark.core.provenance import digest_file, stable_hash

from ..artifact import read_manifest
from ..config import TIESConfig


def test_artifact_rejects_foreign_incomplete_and_modified_statistics(tmp_path):
    config = TIESConfig(calibration_samples=4)
    samples = [dict(split="train", suite=str(i), sha256=str(i)) for i in range(4)]
    taus = np.array([-0.5, -0.2, 0.2, 0.5])
    np.savez(tmp_path / "statistics.npz", taus=taus)
    manifest = dict(
        schema_version=1,
        algorithm="ties",
        complete=True,
        backend="cogact",
        calibration_parameters=config.calibration_parameters(),
        samples=samples,
        samples_sha256=stable_hash(samples),
        statistics_sha256=digest_file(tmp_path / "statistics.npz"),
        quantiles=np.quantile(taus, [0.25, 0.5, 0.75]).tolist(),
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert read_manifest(tmp_path, config, "cogact") == manifest
    for changes in [
        dict(complete=False),
        dict(backend="openvla_oft"),
        dict(samples=samples[:-1]),
        dict(quantiles=[0, 0.1, 0.2]),
        dict(statistics_sha256="corrupt"),
    ]:
        path.write_text(json.dumps(dict(manifest, **changes)))
        with pytest.raises(ValueError):
            read_manifest(tmp_path, config, "cogact")
    path.write_text(json.dumps(manifest))
    np.savez(tmp_path / "statistics.npz", taus=taus + 0.01)
    with pytest.raises(ValueError):
        read_manifest(tmp_path, config, "cogact")
