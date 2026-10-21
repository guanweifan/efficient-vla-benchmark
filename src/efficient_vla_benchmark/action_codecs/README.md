# Action codec evaluation

Independent reconstruction benchmark for discrete action representations.
It does not replace OFT/CogACT action heads or measure policy success or speedup.

## Supported codecs

Presets are in [configs/codecs](../../../configs/codecs); adapters document pinned
assets, normalization and representation limits.

| Codec | Evaluation scope |
|---|---|
| [Uniform binning](implementations/uniform_binning/README.md) | Classical scalar baseline |
| [FAST+](implementations/fast_plus/README.md) | Frozen DCT/BPE, native CPU implementation |
| [BEAST](implementations/beast/README.md) | B-spline quantization; 5/10/20 basis workpoints |
| [VQ-VLA](implementations/vq_vla/README.md) | Released five-frame residual VQ |
| [MiniVLA RVQ](implementations/minivla_rvq/README.md) | Eight-frame RVQ component, not the full VQ-BeT policy |
| [ActionCodec](implementations/actioncodec/README.md) | Base and released RVQ-finetuned weights; complete-layer prefixes |
| [MINT SDAT](implementations/mint_sdat/README.md) | Separate matched sixteen-frame group |
| [SpatialVLA](implementations/spatialvla/README.md) | Normalized-coordinate transfer diagnostic |
| [X-Tokenizer](implementations/x_tokenizer/README.md) | Initial-state-conditioned command-space diagnostic |

Rate variants are workpoints of one codec, not additional methods. Compare only
within the matching workload/representation group in the
[results](../../../docs/benchmark_results.md#independent-action-codec-reconstruction).

## Setup and inputs

Use an isolated Python 3.10 environment; do not change either policy environment.
The validated GPU build is PyTorch 2.5.1+cu124.

```bash
python3.10 -m venv .venv-codecs
.venv-codecs/bin/python -m pip install \
  -r src/efficient_vla_benchmark/action_codecs/requirements.txt
.venv-codecs/bin/python -m pip install -e .
source .venv-codecs/bin/activate

python -m efficient_vla_benchmark.action_codecs.fetch_sources \
  --output "$CODEC_SOURCES"
```

Obtain weights/processors at the revisions in each adapter README. The explicit
asset downloader covers these four releases; it does not download full policies:

```bash
python -m efficient_vla_benchmark.action_codecs.fetch_assets \
  --output "$CODEC_ASSETS" --codec minivla_rvq --codec mint_sdat \
  --codec spatialvla --codec actioncodec_rvq
```

Prepare demonstrations in the RLDS data environment:

```bash
python -m efficient_vla_benchmark.action_codecs.prepare \
  --dataset-root "$RLDS_ROOT" --output "$CORPUS_DIR"
python -m efficient_vla_benchmark.action_codecs.prepare \
  --source-corpus "$CORPUS_DIR" --horizon 16 --output "$SHORT_CORPUS_DIR"
```

Run reconstruction in the codec environment:

```bash
python -m efficient_vla_benchmark.action_codecs.run \
  --config configs/codecs/fast_plus.yaml --assets "$FAST_PLUS_DIR" \
  --corpus "$CORPUS_DIR" --output "$RESULT_DIR"
```

Source-backed adapters also require `--sources`; GPU adapters take `--device cuda:0`.
Use the short corpus for the matched H16 group, not an adapter-side padding workaround.

## Evaluation contract

- Input/output is CPU float32 raw LIBERO OSC commands, `[frames, 7]`, signed gripper.
  Encode returns integer IDs. Decode receives only IDs, horizon, suite and declared
  initial-state context—not future observations/actions or encoder features.
- Calibration/reconstruction episodes are disjoint, balanced across four suites,
  with one seeded window per episode. Released weights/codebooks/BPE stay frozen;
  calibration statistics never update during reconstruction. Pretraining may include LIBERO.
- The main group uses forty-frame windows, tiled by native blocks without padding,
  dropping or resampling. H16 is a fixed prefix of the same episode population,
  with separately calibrated references. Native receptive fields need not match.
- Report raw per-dimension and aggregate MSE/MAE, gripper agreement, tokens/frame and
  nominal fixed-width bits/frame. Bits exclude shared models/statistics/context and
  are not entropy-coded file size. Report frames, not unverified physical Hz.
- Time batch one, ten warmups and one measured pass, with separate encode/decode
  synchronization. Include transforms, transfers and token extraction; exclude
  loading and I/O. Use an idle device and retain thermal-quality checks.
- Verify serialized IDs with an independently loaded decoder via
  `verification.independent_decode`. Native-equation tests are in
  `tests/test_codec_released.py`. Failed/malformed decodes remain errors, never zero
  fallback or successful-only averages.
- Keep corpus, weights, source, runtime and calibration identities with every result.
  Coordinate-transfer and state-conditioned diagnostics remain separate; reconstruction
  alone establishes neither closed-loop success nor VLA acceleration.

Each codec owns one adapter; corpus/packet/measurement code does not branch on model
names. Optional tensor dependencies load only in the selected adapter.
