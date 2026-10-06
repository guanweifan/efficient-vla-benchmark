"""Opt-in real released-codec checks; resources are supplied by the caller."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from efficient_vla_benchmark.action_codecs.core import Packet
from efficient_vla_benchmark.action_codecs.corpus import Corpus
from efficient_vla_benchmark.action_codecs.loading import factory
from efficient_vla_benchmark.action_codecs.normalization import fit_quantiles
from efficient_vla_benchmark.action_codecs.verification import independent_decode

CASES = [
    ("uniform_binning", None, None, 40),
    ("minivla_rvq", "minivla_rvq", None, 40),
    ("spatialvla", "spatialvla", "spatialvla", 40),
    ("mint_sdat", "mint_sdat", "mint_sdat", 16),
    ("actioncodec", "actioncodec", None, 40),
    ("actioncodec_rvq", "actioncodec_rvq", None, 40),
    ("actioncodec_rvq_q1", "actioncodec_rvq", None, 40),
    ("actioncodec_rvq_q2", "actioncodec_rvq", None, 40),
    ("fast_plus", "fast_plus", None, 40),
    ("beast", "beast", None, 40),
    ("vq_vla", "vq_vla", "vq_vla", 40),
    ("x_tokenizer", "x_tokenizer", "x_tokenizer", 40),
]


def resources():
    names = (
        "EVB_CODEC_ASSET_ROOT",
        "EVB_CODEC_SOURCE_ROOT",
        "EVB_CODEC_CORPUS",
        "EVB_CODEC_SHORT_CORPUS",
    )
    values = [os.environ.get(k) for k in names]
    if not all(values):
        pytest.skip("Explicit released assets, sources and both frozen corpora required")
    return list(map(Path, values))


@pytest.mark.parametrize("preset,asset,source,horizon", CASES)
def test_released_discrete_only_roundtrip(preset, asset, source, horizon):
    assets, sources, full, short = resources()
    corpus = Corpus(short if horizon == 16 else full)
    cfg = yaml.safe_load(Path("configs/codecs", preset + ".yaml").read_text())
    build = factory(cfg["factory"])
    kwargs = dict(
        assets=assets / asset if asset else None,
        sources=sources / source if source else None,
        corpus=corpus,
        calibration={"corpus_sha256": corpus.sha256, "quantiles": fit_quantiles(corpus)},
        device="cpu"
        if preset in ("uniform_binning", "spatialvla", "fast_plus", "beast")
        else os.environ.get("EVB_CODEC_DEVICE", "cpu"),
        **cfg["parameters"],
    )
    encoder, decoder = build(**kwargs), build(**kwargs)
    try:
        for suite in sorted({r["suite"] for r in corpus.rows("calibration")}):
            row = next(r for r in corpus.rows("calibration") if r["suite"] == suite)
            actions, context = corpus.window(row)
            evidence = independent_decode(encoder, decoder, actions, context)
            assert evidence["max_independent_decode_difference"] <= 1e-6
            packet = encoder.encode(actions, context)
            if preset == "mint_sdat":
                import torch

                with torch.inference_mode():
                    raw = encoder.model(torch.tensor(actions[None], device=encoder.device))[0]
                    grip = (raw[..., 6:8].argmax(-1, keepdim=True) * 2 - 1).to(raw.dtype)
                    native = torch.cat([raw[..., :6], grip], -1)[0].cpu().numpy()
                np.testing.assert_allclose(
                    decoder.decode(packet, context), native, rtol=1e-5, atol=2e-6
                )
            if preset.startswith("actioncodec"):
                native = [
                    encoder.model.encode(block[None], embodiment_ids=encoder.embodiment)[0]
                    for block in np.split(encoder.normalized(actions), horizon // 20)
                ]
                np.testing.assert_array_equal(
                    packet.ids, np.asarray(native)[:, : encoder.token_count]
                )
            with pytest.raises(ValueError):
                decoder.decode(Packet(np.array([0]), packet.vocabulary, horizon + 1), context)
    finally:
        encoder.close()
        decoder.close()
        encoder.close()
        decoder.close()


def test_minivla_matches_pinned_official_dependency():
    assets, sources, corpus, _ = resources()
    # An independent interpreter isolates the author's older vendored quantizer
    # from newer optional packages used by other codecs.
    script = r"""
import json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, sys.argv[2])
from vqvae.vqvae import VqVae
from efficient_vla_benchmark.action_codecs.corpus import Corpus
from efficient_vla_benchmark.action_codecs.normalization import (
    fit_quantiles, normalize, unnormalize,
)
from efficient_vla_benchmark.action_codecs.implementations.minivla_rvq.adapter import MiniVlaRvq
root=Path(sys.argv[1])
directory=root/'pretrain_vq+mx-libero_90+fach-7+ng-7+nemb-128+nlatent-512'
cfg=json.loads((directory/'config.json').read_text())
cfg.update(device=sys.argv[4],eval=True,load_dir=str(directory/'checkpoints/model.pt'))
native=VqVae(**cfg)
native.encoder.eval();native.decoder.eval()
corpus=Corpus(sys.argv[3])
calibration={'corpus_sha256':corpus.sha256,'quantiles':fit_quantiles(corpus)}
codec=MiniVlaRvq(assets=root,calibration=calibration,device=sys.argv[4])
for suite in sorted({r['suite'] for r in corpus.rows('calibration')}):
 row=next(r for r in corpus.rows('calibration') if r['suite']==suite)
 a,ctx=corpus.window(row)
 value=normalize(a,calibration['quantiles'][suite],signed_gripper=False)
 ids=[];recon=[]
 for block in np.split(value,len(value)//8):
  _,code=native.get_code(torch.tensor(block[None],device=sys.argv[4]))
  ids.append(code[0].cpu().numpy())
  with torch.no_grad():
   recon.append(native.get_action_from_latent(native.draw_code_forward(code))[0].cpu().numpy())
 np.testing.assert_array_equal(codec.encode(a,ctx).ids,np.array(ids))
 expected=unnormalize(np.concatenate(recon),calibration['quantiles'][suite],signed_gripper=False)
 np.testing.assert_allclose(codec.decode(codec.encode(a,ctx),ctx),expected,rtol=0,atol=1e-6)
print('Official MiniVLA dependency parity: four suites, all blocks')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(assets / "minivla_rvq"),
            str(sources / "minivla_rvq"),
            str(corpus),
            os.environ.get("EVB_CODEC_DEVICE", "cpu"),
        ],
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
