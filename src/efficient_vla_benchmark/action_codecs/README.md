# Independent action codec evaluation

This component evaluates discrete action representations, not policy acceleration.
It never replaces an OFT/CogACT action head or borrows their success-rate claims.
Each codec owns one adapter directory; common corpus, packet and measurement code
knows no codec/model names. Tensor runtimes are loaded only by selected adapters.

## Contract

- Input/output: CPU float32 raw LIBERO OSC commands, `[frames, 7]`, signed gripper.
- Encode returns integer tokens only. Decode receives those tokens, horizon, suite
  identity and one initial observed state; no future actions/observations or latent
  encoder features. Required initial-state side information must be disclosed.
- Preserve released codec weights/quantizers; no training or codebook/BPE refitting.
  Closed-form calibration uses only the frozen calibration split and is not updated
  during reconstruction. Pretrained codecs may already have seen LIBERO.
- Use forty-frame windows: native five-/twenty-frame codecs tile them exactly;
  flexible codecs receive the complete window. Do not pad, drop or resample actions.
  This compares representations on one workload, not identical model receptive fields.
- A seeded demonstration corpus has disjoint calibration/reconstruction episode IDs,
  one window per episode and equal counts across four suites. It is separate from
  simulator Dev/final episodes, but is NOT claimed as unseen pretraining data.
- Report raw-command error per dimension, aggregate MSE/MAE, gripper agreement,
  tokens/frame and nominal fixed-width bits/frame. Nominal bits exclude shared model,
  statistics and known context; they are not an actual entropy-coded file size.
- Batch one, ten warmups, one measured pass. Include normalization, conversion,
  host/device transfers and token extraction; exclude loading and disk I/O. Encode
  and decode are separately synchronized. Never turn codec latency into VLA speedup.
- A failed decode remains an error; no zero fallback or successful-only averages.
  Record exact corpus, weights, runtime, source, calibration and adapter identities.

Prepare in the existing RLDS data environment. For codec evaluation, use a separate
Python 3.10 environment; do not install over either policy environment:

```bash
python3.10 -m venv .venv-codecs
.venv-codecs/bin/python -m pip install \
  -r src/efficient_vla_benchmark/action_codecs/requirements.txt
.venv-codecs/bin/python -m pip install -e .
```

The validated GPU build is PyTorch 2.5.1+cu124. Choose the matching official CUDA
wheel for the host; do not change policy packages. Assets are supplied separately
and verified against each adapter's pinned manifest. VQ-VLA and X-Tokenizer need
their small pinned source subsets, obtained without cloning whole policy projects:

```bash
python -m efficient_vla_benchmark.action_codecs.fetch_sources \
  --output "$CODEC_SOURCES"
```

Run preparation/evaluation in the corresponding environment:

```bash
python -m efficient_vla_benchmark.action_codecs.prepare \
  --dataset-root "$RLDS_ROOT" --output "$CORPUS_DIR"
python -m efficient_vla_benchmark.action_codecs.run \
  --config configs/codecs/fast_plus.yaml --assets "$FAST_PLUS_DIR" \
  --corpus "$CORPUS_DIR" --output "$RESULT_DIR"
```

Choose `configs/codecs/{fast_plus,beast,vq_vla,actioncodec,x_tokenizer}.yaml`.
VQ-VLA/X-Tokenizer additionally take `--sources` pointing to their fetched subset;
GPU adapters take `--device cuda:0`. FAST+ uses native CPU DCT/BPE. BEAST's basis5,
basis10 and basis20 presets expose rate–distortion choices, not policy tuning.
Only released quantizers are used; each adapter README states representation,
calibration, tiling and decoder-side-information limits.

Each result records per-window timestamps plus a measured-pass interval for
external GPU-quality checks. Use one clean frozen source and an idle GPU, record
thermal flags throughout that interval, and retain failed quality checks separately
from accepted latency. This sampler cannot certify absence of sub-poll transients.
The timing protocol is still one pass, not a best-of-repeat. Codec dependencies
and measurement records stay isolated from policy results.

The dataset's nominal frame rate is not assumed. Native embodiment labels may name
another rate; without a verified conversion, report frames rather than physical Hz.
