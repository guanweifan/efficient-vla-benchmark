# efficient-vla-benchmark

Port VLA acceleration methods and independent core contributions onto pinned native code, with
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
One run uses the native baseline or exactly one method with explicitly declared scope.

## Implemented scope

| Backend / variant | Native baseline | Token reduction | Caching | Model routing | Action chunking | Quantization |
|---|---|---|---|---|---|---|
| OpenVLA-OFT / libero_joint | Native SDPA | Nine methods below | VLA-Cache, VLA-InfoEntropy | SP-VLA pruning + scheduling | VOTE ensemble core | AWQ W4A16 |
| CogACT | Source pinned; runtime not implemented | — | — | — | — | — |
| starVLA | Source pinned; runtime not implemented | — | — | — | — | — |

Algorithms retain their full implemented mechanism and disclose adaptations in
their READMEs. This is a common-checkpoint comparison, not a claim to reproduce
each paper's original numbers. Quantization uses real packed kernels, not fake quantization.

### Token-reduction methods

Every row is one independent plugin under `methods/token_reduction/`, selected by
the correspondingly named experiment YAML. Shared backend hooks do not contain
paper-specific policy, and no method imports another method's implementation.

| Method ID | Mechanism | Implementation basis |
|---|---|---|
| `specprune_vla` | Temporal/global and layerwise pruning with action control | Pinned source port |
| `adp` | Text Q/K selection and observed-motion pruning gate | Pinned source port |
| `vla_pruner` | Semantic/action saliency, history and diversity | Pinned source port |
| `safe_pruner` | Forecast saliency and adaptive full-reference refresh | Pinned source port |
| `team_vla` | Early spatial expansion and mid-layer soft merging | Independent paper implementation |
| `vla_iap` | Semantic/motion alignment and geometric token protection | Independent paper implementation |
| `efficient_vla` | Task-relevance and diversity token selection only | Independent paper implementation |
| `st_merge` | Shared spatial token groups in both vision encoders | Independent paper implementation |
| `flash_vla` | Attention-output SVD/ICS pruning only | Independent paper implementation |

Paper-based methods retain explicit unresolved conventions in their method READMEs;
they are not verified author-code reproductions. EfficientVLA excludes layer pruning
and DiT caching. FlashVLA excludes action reuse pending clarification of contradictory
paper equations. ST-Merge retains a dense projector bridge and reports its cost.
Integration and speed do not prove
task reliability. Failed smoke episodes remain visible in the measurement ledger.

SP-VLA also contributes token pruning, but lives once under `model_routing/sp_vla/`.
Its `sp_vla_scheduling` experiment remains a scheduling-only ablation; both presets
use method ID `sp_vla`. VLA-InfoEntropy lives under `caching/vla_infoentropy/`: it
skips visual query computation while retaining visual KV context. Both cross-reference
Section 5.1 without duplicating their implementations or evidence rows.

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

The [measurement ledger](docs/benchmark_results.md) groups measured methods by
survey section/category and links configuration, source identity, paired baseline,
raw-result hashes and smoke records. Use [result recording](docs/recording_results.md)
for every future integration; formal success rate stays separate from smoke.

The [control-method validation report](docs/control_methods_validation.md) details
the cache follow-up, the two independent core contributions, matched-cadence
comparisons and clean-checkout verification.

The [token-reduction validation report](docs/token_reduction_validation.md) records
the five-method Section 5.1 batch, source-fidelity limitations, smoke failures,
observed token budgets, paired measurements and independent regression checks.

The [second token-reduction batch report](docs/token_reduction_batch2_validation.md)
records EfficientVLA's token core, full SP-VLA integration, ST-Merge, VLA-InfoEntropy
and FlashVLA's ICS core, with matched ablations, retained regressions and the
unresolved FlashTrigger action-reuse boundary.
