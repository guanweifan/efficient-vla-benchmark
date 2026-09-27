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

Prepare in a data environment; run codecs in their isolated optional environment:

```bash
python -m efficient_vla_benchmark.action_codecs.prepare \
  --dataset-root "$RLDS_ROOT" --output "$CORPUS_DIR"
python -m efficient_vla_benchmark.action_codecs.run \
  --config configs/codecs/fast_plus.yaml --assets "$FAST_PLUS_DIR" \
  --corpus "$CORPUS_DIR" --output "$RESULT_DIR"
```

The dataset's nominal frame rate is not assumed. Native embodiment labels may name
another rate; without a verified conversion, report frames rather than physical Hz.
