# efficient-vla-benchmark

Reproduce complete VLA acceleration methods on pinned native model code, with
optional backend packages and one shared control/evaluation/profiling contract.
This repository contains no survey manuscript, model weights or datasets.

## Architecture

- `src/efficient_vla_benchmark/`: lightweight contracts, method catalog and shared harnesses.
- `backends/<backend>/`: independently installed backend plugin; its `upstream/` is a pinned submodule.
- `methods/<category>/<method>/` inside the main package: one paper, with explicit per-backend implementations.
- `configs/`: portable presets; filesystem resources are supplied only on the command line.
- `third_party/`: pinned algorithm references, runtime libraries and benchmark sources.

No backend is a mandatory dependency of the main package. Install only the backend
you use. A source-only integration is not an implemented or validated backend.
One run uses the native baseline or exactly one complete acceleration method.

## Implemented scope

| Backend / variant | Native baseline | Token reduction | Caching | Quantization |
|---|---|---|---|---|
| OpenVLA-OFT / libero_joint | Native SDPA | SpecPrune-VLA | VLA-Cache | AWQ W4A16 |
| CogACT | Source pinned; runtime not implemented | — | — | — |
| starVLA | Source pinned; runtime not implemented | — | — | — |

Algorithms retain their full implemented mechanism and disclose adaptations in
their READMEs. This is a common-checkpoint comparison, not a claim to reproduce
each paper's original numbers. Quantization uses real packed kernels, not fake quantization.

## Start with one backend

Read [installation](docs/installation.md) and the [OFT recipe](backends/openvla_oft/INSTALL.md).
After installation, inspect supported combinations without loading a model:

```bash
evb list
evb validate-config --config configs/experiments/openvla_oft/specprune_vla.yaml

evb evaluate --config configs/experiments/openvla_oft/baseline.yaml \
  --model-path "$MODEL_DIR" --libero-root third_party/benchmarks/libero \
  --output-dir "$SMOKE_DIR"

evb benchmark --config configs/experiments/openvla_oft/specprune_vla.yaml \
  --model-path "$MODEL_DIR" \
  --traces "$SPATIAL_TRACE" "$OBJECT_TRACE" "$GOAL_TRACE" "$LONG_TRACE" \
  --output-dir "$TIMING_DIR"
```

The caller supplies all resource variables. Change the experiment config to select
one algorithm; AWQ also requires `--artifact-path`. Configs expose method parameters
and compose backend/evaluation/profiling presets. CLI overrides are saved in results.
Evaluation defaults to four smoke episodes; formal campaigns require explicit scheduling.

See [architecture](docs/architecture.md), [contribution rules](CONTRIBUTING.md),
[adding a backend](docs/adding_backend.md) and [adding a method](docs/adding_method.md).
Installable templates demonstrate external plugins without editing the core. They
are contract fixtures, not VLA models or benchmark-performance evidence.

Fresh timing, smoke outcomes and migration checks are recorded in
[migration validation](docs/migration_validation.md). Preserve negative results;
these development checks are not formal success-rate measurements.

The subsequent [source audit and retest](docs/performance_validation.md) records
the VLA-Cache attention correction and AWQ's explicit FP16-kernel option, including
precision differences, native controls and remaining negative speed results.
