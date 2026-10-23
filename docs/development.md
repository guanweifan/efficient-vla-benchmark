# Development guide

[README](../README.md) covers installation and supported methods;
[AGENTS.md](../AGENTS.md) defines contribution and resource-use rules.

## Architecture

| Component | Responsibility |
|---|---|
| `core` | Types, lifecycle, configuration, discovery and provenance |
| `model_ops` | Shared tensor operations |
| `backend` | Native loading, preprocessing, model access and action conversion |
| `method` | One acceleration mechanism and its backend bindings |
| `control` | Action queue, executed history and query scheduling |
| `calibration` | Read-only training-observation sampling |
| `evaluation` | LIBERO manifests, seeds, stopping and outcomes |
| `profiling` | Replay, timing boundaries and aggregation |
| `action_codecs` | Independent action reconstruction |

Backends live under `backends/<id>/`; methods live under
`src/efficient_vla_benchmark/methods/<category>/<paper_id>/`.
Core and discovery load without tensor or simulator dependencies. Methods use backend
interfaces; shared infrastructure remains independent of specific models and papers.

## Policy and method lifecycle

Policies expose `reset`, `observe`, `predict_chunk`, `after_action`, `metadata` and
`close`. State belongs to each policy; reset clears episode state and close restores hooks.

1. `observe` receives every control frame.
2. With an empty queue, `before_plan` may supply a replacement action; the first step queries the model.
3. `on_plan` receives the predicted chunk. `select_execution_horizon` may select an existing prefix.
4. `transform_action` may aggregate predictions; `after_action` may clear the remaining queue.
5. `control_diagnostics` reports counters without changing execution.

The backend owns auxiliary queries requested through `wants_auxiliary_query`, including
synchronization and RNG restoration. All auxiliary work remains inside timing.

## Model-access boundaries

- [OFT](../backends/openvla_oft/DECODER.md): preserve native positions, bidirectional visibility and action readout.
- [CogACT](../backends/cogact/README.md): preserve causal visibility, cognition EOS, preprocessing and diffusion conventions.
- `TokenReduction` selects or merges visual rows. Causal groups anchor at their latest original member.
- Indexed KV reuse retains context while reducing computed queries; it differs from token deletion.
- Attention probes preserve native outputs. Instance-scoped patches restore on detach; caches reset at their declared boundaries.
- Candidate generation and diffusion samplers account for all candidates, probes, CFG work and RNG advancement.

## Add a method

Start from [templates/method](../templates/method/README.md). Provide a mechanism README,
`metadata.yaml`, typed configuration and backend-specific factories under `implementations/`.
A factory receives `(backend, method_config, RuntimeResources)`; preflight validates
resources before model loading. Optional `validate_experiment` checks cadence and protocol.

Independent cores use `components/<scope>.yaml` with their own IDs and boundaries.
Add final presets under `configs/experiments/`. Test equations, active execution,
reset/detach, unsupported inputs and native behavior before evaluation.

## Add a backend

Start from [templates/backend](../templates/backend/README.md). Register a lazy API-v1
`BackendSpec`, implement policy lifecycle and `RuntimeMonitor`, and expose method-facing
capabilities. Pin upstream sources and provide an isolated installation recipe.
Validate loading, normalization, native action parity, checkpoint immutability and timing.

## Training-dependent boundary

Learned selectors, routers and distilled heads require trained parameters. Calibration
samples training observations and is separate from gradient-based policy training.
Method READMEs specify the required assets and implemented scope.

## Final configuration contract

`configs/evaluation/primary_selection.yaml` records one selected preset per method/backend,
plus fixed controls and useful alternatives. Hyperparameters live in YAML; `measured_preset`
preserves the original name when a preset is renamed. Effective settings and measurement
identities remain attached to the results they produced.

## External assets

Use CLI arguments for resource locations. Model download instructions are in the
[OFT](../backends/openvla_oft/README.md#model-asset) and
[CogACT](../backends/cogact/README.md#model-asset) guides.
Ordinary rollouts need LIBERO resources; calibration additionally uses training data.

```bash
hf download openvla/modified_libero_rlds --repo-type dataset \
  --revision 6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551 --local-dir "$RLDS_DIR"
evb calibrate --config "$PRESET" --model-path "$MODEL_DIR" \
  --dataset-path "$RLDS_DIR" --artifact-dir "$ARTIFACT_DIR"
evb check-model --config "$PRESET" --model-path "$MODEL_DIR" \
  --artifact-path "$ARTIFACT_DIR"
```

Add `--runtime-path` for external engines. Calibration artifacts remain separate from weights.

| Methods | Preparation |
|---|---|
| AWQ, GPTQ, SmoothQuant, DA-MPA, QuantVLA, HoloQ, TIES, EcoVLA, EfficientVLA layer core, BAC, TeaCache | RLDS calibration; method README specifies budget/runtime |
| PACE | Demonstration-action calibration on the pinned LeRobot-v2 conversion |
| ActionCache | Native population collection and freeze CLIs |
| DyQ-VLA | Independent trajectories and mixed-integer runtime; historical population is not distributed |
| EVO | External frozen search artifact; complete search driver and historical population are not distributed |
| ActQuant | HSIC/AMF calibration, GGUF conversion and author-engine packing; historical engine bundle is not distributed |
| Action codecs | [Codec environment and assets](../src/efficient_vla_benchmark/action_codecs/README.md) |

### Rebuilding evaluation inputs

Generate manifests before evaluation:

```bash
evb evaluation-manifest --libero-root "$LIBERO_ROOT" --output "$DEV_MANIFEST"
evb derive-manifest --development-manifest "$DEV_MANIFEST" \
  --purpose timing --output "$TIMING_MANIFEST"
evb derive-manifest --development-manifest "$DEV_MANIFEST" \
  --purpose final --output "$FINAL_MANIFEST"
```

Collect native traces and replay the same workload for each comparison:

```bash
evb evaluate --config "$BASELINE_PRESET" --model-path "$MODEL_DIR" \
  --libero-root "$LIBERO_ROOT" --episode-manifest "$TIMING_MANIFEST" \
  --record-traces --depth --output-dir "$REFERENCE_RUN"
python scripts/timing_workload.py --manifest "$TIMING_MANIFEST" \
  --run-dirs "$REFERENCE_RUN" --output "$WORKLOAD"
evb benchmark --config "$PRESET" --model-path "$MODEL_DIR" \
  --traces "$REFERENCE_RUN"/traces/*.npz --workload-manifest "$WORKLOAD" \
  --repeats 1 --warmup 10 --output-dir "$TIMING_DIR"
```

Add artifact/runtime arguments as required. Measure paired native references on the
same GPU and trace bank. Historical traces are not included in the repository.

## Calibration, evaluation and timing

RLDS sampling balances training episodes across suites and selects one seeded observation
per episode. Preserve policy-oriented RGB, raw state and language. Independent rollout
calibration uses placements disjoint from evaluation states and retains failed trajectories.

Smoke runs one episode per suite with ten settling steps and control limits
220/280/300/520 for Spatial/Object/Goal/Long. Depth methods require aligned metric depth.
CUDA and EGL must address the same physical GPU.

### Fixed development validation

`libero_dev200_v1` selects five of the fifty standard states per task for configuration
selection; the complete evaluation contains 2,000 episodes across forty tasks.

```bash
evb evaluate --config "$PRESET" --model-path "$MODEL_DIR" \
  --libero-root "$LIBERO_ROOT" --episode-manifest "$DEV_MANIFEST" \
  --shard-index 0 --num-shards 8 --output-dir "$RUN_DIR" --no-record-traces
```

Each shard uses a separate output directory. `scripts/evaluation_results.py` plans and
merges jobs, checking duplicate, missing and incompatible episodes. Resume preserves
existing outcomes and verifies configuration, checkpoint and environment identities.

### Selection and final-measurement workflow

Verify the mechanism and smoke behavior, select parameters, then freeze the configuration,
checkpoint, calibration and runtime. Follow the approval and outcome-retention rules in
[AGENTS.md](../AGENTS.md) before launching full evaluations.

`libero_replay4x128_v1` measures four frozen native prefixes of at most 128 controls each,
with ten warmup queries and one measured pass. Mean cost includes preprocessing, transfers,
method/control work and postprocessing; loading, disk I/O and simulation are excluded.
Use fresh processes, fixed CPU threads and same-GPU native pre/post references. Accepted
pairs have at most 3% latency drift and no thermal slowdown.

Primary references execute eight actions per query. Adaptive methods use their selected
prefixes. Peak CUDA allocation excludes external-engine memory; unavailable values remain
missing. Query counters distinguish model calls, reused actions and auxiliary computation.

## Quantization runtime admission

Quantized timing uses actual packed execution. Record weight/activation precision,
accumulation type, target layers and conversion costs. Tile-dequantized floating GEMM
and integer MMA are distinct runtimes. Shared kernels live in
[model_ops/quantization](../src/efficient_vla_benchmark/model_ops/quantization/README.md);
method READMEs own calibration and numerical conventions.

The supported catalog is in [README](../README.md#supported-methods); unresolved scopes
and their reasons remain in the selection manifest. Runtime changes require affected
parity, smoke and timing checks before results are reused.

## Record results

Canonical JSON and raw runs stay in ignored output directories. Publish the concise
[results ledger](benchmark_results.md) and selected presets. The reporter checks actual
source revisions, checkpoint/configuration identity and each method's paired native.

```bash
python scripts/record_result.py export \
  --run-dir "$TIMING_DIR" --baseline-dir "$BASELINE_TIMING_DIR" \
  --smoke-dir "$SMOKE_DIR" --section 'Dynamic Computation' \
  --label 'SP-VLA pruning + scheduling' \
  --preset configs/experiments/cogact/sp_vla.yaml
python scripts/readiness_report.py --selection configs/evaluation/primary_selection.yaml \
  --development-summary "$DEV_SUMMARY" --development-manifest "$DEV_MANIFEST" \
  --records-dir "$RECORDS_DIR" --timing-quality "$TIMING_QUALITY" \
  --output "$GENERATED_LEDGER" --evidence-output "$READINESS_JSON"
```

Supply each referenced summary and complete final campaign with `--development-summary`
and `--final-summary`; add codec inputs with `--codec-run` and `--codec-auxiliary-run`.
Generate to ignored output, then reconcile the independent native/runtime controls.
The JSON output retains full selection, measurement and provenance records.
[history_commit_map.json](../scripts/history_commit_map.json) resolves historical sources.

## Local checks

```bash
python -m pytest tests -m 'not gpu and not libero'
python scripts/check_boundaries.py
python scripts/check_private_paths.py
python scripts/check_repository.py
```

Run backend-specific tests in its own environment; OFT's Transformers fork and CogACT's
stock Transformers have separate dependency stacks.
