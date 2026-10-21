# Development guide

[README](../README.md) covers installation and supported methods.
[AGENTS.md](../AGENTS.md) defines contribution, Git and resource-use rules.
This guide covers interfaces, evaluation and result provenance.

## Architecture

| Component | Owns | Must not own |
|---|---|---|
| core | Types, lifecycle, config, discovery and provenance | Model internals or paper decisions |
| model_ops | Optional shared tensor operations | Backend imports or paper policy |
| backend | Native loading, preprocessing, model access and actions | Paper algorithms or simulator loops |
| method | One declared mechanism, bindings and per-instance state | Other methods or environment stepping |
| control | Action queue, executed history and scheduling | Backend/method-name branches |
| calibration | Read-only training-observation sampling | Quantizer search or packing |
| evaluation | LIBERO protocol, seeds, stopping and outcomes | Concrete model or method imports |
| profiling | Replay, timing boundaries and aggregation | Policy changes |
| action_codecs | Independent action reconstruction | Policy heads or closed-loop success claims |

Backends are optional packages under `backends/<id>/`; `upstream/` pins native source.
Discovery is lightweight and loads tensor runtimes only when selected. Unsupported
method/backend pairs fail explicitly, never fall back to baseline. External plugins
register through entry points without editing core. Methods may use backend APIs;
backends and shared infrastructure must not import methods.

## Policy and method lifecycle

Policies expose `reset`, `observe`, `predict_chunk`, `after_action`, `metadata` and
`close`. `MethodHost` delegates method callbacks. State and hooks belong to one
policy; reset clears episode state, and idempotent close releases or restores it.
Instruction changes clear the action queue and reset the predictor.

1. `observe(observation)` receives every control frame.
2. At an empty queue, `before_plan(observation, step)` may supply one environment-ready
   replacement action. The first step after reset always queries the model.
3. A model query invokes `on_plan(full_chunk_copy, step)` before prefix selection.
   Optional `select_execution_horizon(full_chunk_copy, default_horizon)` returns an
   integer from 1 to the actual predicted length. It cannot pad or extend predictions.
4. `transform_action(action_copy, step)` may aggregate predictions.
   `after_action(final_action_copy, remaining)` receives the emitted command;
   returning true clears the remaining queue before the next step.
5. `control_diagnostics()` reports counters without changing behavior.

Hooks do not run the simulator or invoke inference themselves. Actions already use
the backend's environment convention; core assumes neither seven dimensions nor a
gripper index. Invalid/nonfinite outputs fail before execution. Record predicted
length, selected prefix, emitted actions and served query counts separately.

`wants_auxiliary_query()` requests one same-observation auxiliary pass owned by the
backend. Its output is discarded; all compute and joins remain timed. The worker
must finish before the next native query, reset or detach, and restore stochastic
head RNG. One policy owns the process; concurrent global-RNG manipulation is unsupported.

## Model-access boundaries

- [OFT decoder](../backends/openvla_oft/DECODER.md): preserve original positions,
  protected nonvisual rows, bidirectional visibility and the native action readout.
- [CogACT](../backends/cogact/README.md): preserve causal visibility, cognition EOS,
  native preprocessing, CFG, DDIM coordinates and RNG. Fresh-prefill selection does
  not imply arbitrary past-KV decoding or action-to-image attention inside DiT.
- `TokenReduction` selects/merges visual rows only. Causal groups anchor at or after
  their latest original member. Fused vision branches must preserve corresponding
  patch membership, not merely equal output counts.
- `NativeAttentionProbe` and `VisionAttentionProbe` inspect native Q/K without
  changing attention output. Statistics normalize over all current keys; transient
  tensors clear after the call. Probing costs belong in timing.
- Indexed KV reuse retains the full key context while skipping selected query/MLP
  rows. It is not context-token deletion. Original causal rows remain authoritative.
- `ForwardPatches` restores instance attributes and never patches model classes.
  Denoiser caching retains native conditioning, residuals and CFG rows; within-query
  caches clear at action-query boundaries.
- Candidate generation shares one cognition encoding. Count every candidate,
  discarded probe and verification branch; return an actual selected chunk.
- `EpsilonSamplerContract` and `model_ops.scheduled_diffusion` preserve the native
  finite-noise start, training-noise table, CFG pairing and RNG advancement.
  Methods own update equations; histories reset per query and duplicate grid indices fail.

Native/no-op and detach checks are mandatory. Eager/SDPA and packed-kernel precision
differences must be explicit, not asserted bitwise equal. Method READMEs own
paper-specific conventions, backend adaptations and exclusions.

## Add a method

Use [templates/method](../templates/method/README.md). Create one directory at
`src/efficient_vla_benchmark/methods/<category>/<paper_id>/`:

- `README.md`: source revision, mechanism, adaptations, supported variants and usage.
- `metadata.yaml`: ID/category, typed config, capabilities and lazy binding factories.
- `config.py`: portable validated parameters; native geometry checks belong in bindings.
- `implementations/<backend>/`: backend-specific factories and behavior.
- Shared equations only where actually shared; tests beside their owning method.

Factory `(backend, method_config, RuntimeResources)` returns an `InferenceMethod`.
Preflight `(method_config, RuntimeResources)` rejects missing resources before loading.
Optional `validate_experiment(ExperimentConfig, method_config)` checks cadence/protocol.
Calibration uses the factory signature and writes a separate artifact.

Independently evaluated cores use `components/<scope>.yaml` with explicit IDs,
backends and exclusions. An owner with no complete selectable method may declare
`selectable: false`; it must expose at least one valid component. Each run still
selects one core, never an implicit composition.

Add reusable defaults under `configs/methods/` and final compatible presets under
`configs/experiments/`. A second backend extends the same method, not a copied directory.
Test equations, active behavior, reset/detach, unsupported inputs and native boundaries;
then use the shared smoke, Dev and timing workflow.

## Add a backend

Use [templates/backend](../templates/backend/README.md). Name its distribution
`efficient-vla-backend-<name>` and package `efficient_vla_backend_<name>`.

1. Register API-v1 `BackendSpec` with a lazy factory, validator, variants and capabilities.
2. Implement policy lifecycle and `RuntimeMonitor`, including synchronization and reset.
3. Expose APIs required by supported methods without moving paper decisions into the backend.
4. Pin fetchable native sources, provide an isolated installation recipe and explicit bindings.
5. Test loading, native actions/normalization, RNG, checkpoint immutability and shared LIBERO/timing.

Registration or a CPU fixture alone is not validated model support.

## Training-dependent boundary

Learned selectors, routers, exits and distilled denoisers require trained parameters
and an explicitly agreed training budget. Untrained heuristics are not reproductions.
Calibration without optimizer updates is distinct from policy training; neither may
use Dev/Final outcomes as training labels.

## Final configuration contract

Keep one final measured preset per backend/method scope, shared across four suites.
Use short filenames; parameters, precision and scope remain in YAML. Fixed published
controls and useful codec workpoints are separate. Remove obsolete candidate presets
after retaining selection evidence.

`configs/evaluation/primary_selection.yaml` is the sole primary selection list,
not a launcher. `measured_preset` retains original measurement names after renaming.
Compare fully resolved settings before consolidation; never rewrite source, config,
checkpoint or artifact identities to make old results appear current.

## External assets

All locations are CLI arguments. [OFT weights](../backends/openvla_oft/README.md#model-asset)
are publicly downloadable; the exact [CogACT export](../backends/cogact/README.md#model-asset)
is not. Ordinary rollouts need LIBERO resources, not demonstration datasets.

RLDS calibration uses the four `*_no_noops/1.0.0` training datasets:

```bash
hf download openvla/modified_libero_rlds --repo-type dataset \
  --revision 6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551 --local-dir "$RLDS_DIR"
```

| Scope | Required preparation |
|---|---|
| AWQ, GPTQ, SmoothQuant, DA-MPA, QuantVLA, HoloQ, TIES, EcoVLA, EfficientVLA layer core, BAC, TeaCache | `evb calibrate` on RLDS; method README specifies budget and optional runtime |
| PACE | Separate demonstration-action calibration on the pinned LeRobot-v2 conversion |
| ActionCache | Independent native population and freeze CLIs; no Dev/Final collection |
| DyQ-VLA | Frozen complete independent trajectories and mixed-integer runtime; historical population is not distributed |
| EVO | Externally supplied frozen search artifact; historical population/layout and a complete search driver are not distributed |
| ActQuant | HSIC/AMF calibration, floating GGUF conversion and author-engine packing; historical floating/runtime bundle is not distributed |
| Action codecs | [Separate environment, corpus and asset preparation](../src/efficient_vla_benchmark/action_codecs/README.md) |

```bash
evb calibrate --config "$PRESET" --model-path "$MODEL_DIR" \
  --dataset-path "$RLDS_DIR" --artifact-dir "$ARTIFACT_DIR"
evb check-model --config "$PRESET" --model-path "$MODEL_DIR" \
  --artifact-path "$ARTIFACT_DIR"
```

Add `--runtime-path "$RUNTIME_DIR"` when required. Original data/checkpoints are
read-only. A rebuilt artifact has new checksums; it is not the historical artifact
by filename alone. Missing assets must remain explicit, not replaced by a baseline.

### Rebuilding evaluation inputs

Create fixed manifests before observing outcomes; these commands run no policy:

```bash
evb evaluation-manifest --libero-root "$LIBERO_ROOT" --output "$DEV_MANIFEST"
evb derive-manifest --development-manifest "$DEV_MANIFEST" \
  --purpose timing --output "$TIMING_MANIFEST"
evb derive-manifest --development-manifest "$DEV_MANIFEST" \
  --purpose final --output "$FINAL_MANIFEST"
```

Collect a native four-episode timing bank for one backend, then replay it:

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

Supply artifact/runtime arguments as needed. Measure native references before and
after methods on the same GPU and frozen bank. Historical traces are not distributed;
a new bank supports within-bank comparisons, not direct pairing with published latency.

## Calibration, evaluation and timing

The RLDS sampler balances distinct training episodes across suites and selects one
seeded observation per episode. RGB already has policy orientation: do not rotate it
again. Preserve raw state, language, native preprocessing and content digests.
TFDS episode indices are not LIBERO task IDs. Search and packing belong to methods.

Independent rollout calibration/search uses frozen placements disjoint from all
fifty standard states per task and other independent groups. Simulator time alone
does not make placements distinct. Reserve the complete budget, retain failures
without replacement, freeze artifacts before evaluation and report collection/search
cost separately. Method READMEs define the specific populations and freezing rules.

Smoke runs task/state/seed zero in each suite with ten settling steps and control
limits 220/280/300/520 for Spatial/Object/Goal/Long. Four episodes validate connectivity,
not accuracy. `--libero-root` and output-local configuration avoid global LIBERO edits.
Map CUDA and EGL to the same physical GPU; numeric IDs can differ after remapping.
Depth methods require aligned real metric depth via `--depth` or `evaluation.depth`.

### Fixed development validation

`libero_dev200_v1` freezes five sampled states from indices 0–49 for each of forty
tasks, with explicit seeds and resource hashes. Dev-200 is a tuning subset of
Final-2000, not an independent test. Retain the remaining 1,800 outcomes as untuned
evidence from the same final run.

```bash
evb evaluate --config "$PRESET" --model-path "$MODEL_DIR" \
  --libero-root "$LIBERO_ROOT" --episode-manifest "$DEV_MANIFEST" \
  --shard-index 0 --num-shards 8 --output-dir "$RUN_DIR" --no-record-traces
```

Run every shard once with its own index/output. The manifest owns episodes; do not
combine it with `--formal` or task/trial/seed overrides. Identical-run resume preserves
recorded outcomes and rejects changed identities. Infrastructure retries retain
attempt history; ordinary unsuccessful episodes remain failures.

`scripts/evaluation_results.py plan` creates portable jobs; `merge` rejects duplicate,
foreign or incompatible episodes and reports missing records/errors. Machine-specific
scheduling stays outside Git. Freeze worker source throughout a campaign and verify
environment/extension identities before assigning shards. A complete Dev result needs
200 records and zero runtime errors. Parallel wall time is not policy latency.

### Selection and final-measurement workflow

1. Verify the core mechanism, declared adaptations, active execution, native/reset
   boundaries and shared smoke before selecting parameters.
2. Evaluate on fixed Dev-200 and audit meaningful speedward headroom. Prefer faster
   full-cost candidates targeting at most 3 percentage points of native-relative
   success loss. This is not a hard cutoff or an outcome filter: justify near-target
   trade-offs with suite-level evidence. Timing differences within variability are ties.
3. Close search with an accuracy boundary, latency plateau or concrete mechanism limit,
   not merely a passing candidate. Faithful slowdowns/low-success results remain valid;
   an unresolved core is not a certified negative finding.
4. Freeze one configuration, checkpoint, artifact, runtime and selection rationale.
   Final-2000 needs separate explicit owner approval and `--authorize-final`;
   general development permission never authorizes an automatic final queue.
5. Run all 2,000 episodes once, preserving failures and exact resume identities.
   Do not retune from final outcomes, select a favorable repeat or splice old Dev
   results into a new 1,800-episode run. Derive the untuned subset from that same final run.
6. Publish one primary row with separate Dev, Final and timing evidence. Keep fixed
   native controls and useful alternatives explicitly supplementary.

Timing uses `libero_replay4x128_v1`: one frozen native prefix per suite, at most
128 controls each, ten warmup queries and one measured pass. Select prefixes by
fixed episode IDs, not outcomes; reset history after warmup. Include preprocessing,
transfers, method/control work, auxiliary passes and postprocessing; exclude loading,
disk I/O and simulation. Average all controls, including queue-consumption steps.
This is representative-input policy compute, not closed-loop completion time.

Use fresh model processes, fixed CPU threads, idle GPUs and disjoint CPU affinities.
Accept same-GPU native pre/post pairs only with at most 3% latency drift and no
thermal slowdown during measured intervals. Reject affected measurements, preserve
reasons and use the first valid replacement, never the fastest repeat. Do not change
power, clocks or fans. A >10% change from compatible old timing requires diagnosis.

Primary references use native H8. Adaptive methods execute their real selected
prefixes; include selection and candidate costs. Record query/first/steady latency,
control latency, percentiles, query/replacement counts and allocated/reserved memory.
Do not infer repeat variance from a single pass or compare absolute latencies from
different GPU pairs as if paired. Query/prepared-only diagnostics cannot replace
stateful control replay. The [native controls](../backends/cogact/README.md#fixed-native-controls)
retain their declared contrasts and are not extra primary methods.

## Quantization runtime admission

Formal results require real packed execution, not dense fake-quantization latency.
Disclose packing, scale axes, activation storage, product/accumulation precision,
target coverage and all casts/reorders. Tile-dequantized floating GEMM is not integer
MMA. External allocations invisible to Torch are unavailable, never zero.

[Shared operations](../src/efficient_vla_benchmark/model_ops/quantization/README.md)
own kernels, not quantizer algorithms. Method READMEs own calibration, numerical
conventions and selected runtime. Changes require affected parity/reset/reload,
smoke, Dev and timing checks; unchanged evidence keeps its original identity.
`pending_engineering` excludes unresolved ports from primary/final queues;
`excluded` records a fidelity/source boundary, not an unfavorable outcome.

| Unsupported scope | Remaining condition |
|---|---|
| ActQuant on CogACT | Floating GGUF bridge fails action parity; not mathematical incompatibility |
| MABR / HBVLA | Primary equations / probe bootstrap and adaptive grouping remain unverified |
| Full DA-PTQ | CSRC weighting and regularization unresolved; only DA-MPA is supported |
| Full VQVLA | Spatial/temporal multiplication reuse requires a separate engine |
| Mix-QVLA | Native heads lack its action-token-likelihood objective |
| SQAP-VLA | Contradictory activation granularity remains unresolved |
| Trajectory-curvature/attention-guided zero-shot quantization | Required native flow-velocity and action-to-image/text objectives are absent |
| SQIL / BitVLA / QuantWAM | Policy training or another baseline is required |

These boundaries use the 2026-08-17 literature cutoff and inspected primary sources.
They do not establish that every future adaptation is impossible.

## Record results

Canonical JSON and raw runs stay in ignored `outputs/result_records/` or explicit
caller-owned output directories. Commit only the generated
[results ledger](benchmark_results.md), selected presets and necessary source notices.
Never recreate `docs/results/` or store private paths/launchers in Git.

`record_result.py export` validates checkpoint, config, source, dependencies,
hardware, protocol and trace compatibility. Each speedup has its own native pair.
`readiness_report.py` joins selection, actual Dev outcomes, search closure, accepted
timing quality and final evidence; it never launches evaluations.

```bash
python scripts/record_result.py export \
  --run-dir "$TIMING_DIR" --baseline-dir "$BASELINE_TIMING_DIR" \
  --smoke-dir "$SMOKE_DIR" \
  --section '5.3 Layer Skipping, Early Exit, and Model Routing' \
  --label 'SP-VLA pruning + scheduling' \
  --preset configs/experiments/cogact/sp_vla.yaml

python scripts/readiness_report.py --selection configs/evaluation/primary_selection.yaml \
  --development-summary "$DEV_SUMMARY" --development-manifest "$DEV_MANIFEST" \
  --records-dir "$RECORDS_DIR" --timing-quality "$TIMING_QUALITY" \
  --output "$GENERATED_LEDGER" --evidence-output "$READINESS_JSON"
```

Generate to ignored output first. Repeat `--development-summary` for all referenced
selection/coverage summaries; add retained `--final-summary` inputs for complete final
campaigns and `--codec-run` / `--codec-auxiliary-run` for codecs. Preserve independent
native/runtime-control sections when updating the public ledger: the reporter refuses
to overwrite those in place. Legacy `record_result.py render` cannot overwrite a
primary ledger with a selection manifest.

Cross-revision reuse needs identical effective settings/assets and an explicit scoped
source audit. Record actual measured revisions and dirty flags; never relabel outcomes
as a new clean source. [history_commit_map.json](../scripts/history_commit_map.json)
maps cleaned history without changing original evidence identities.
`final_partition()` derives untuned outcomes from one complete final aggregate.

External engines require `--allow-execution-runtime-change` and checksummed runtime
metadata; label cross-runtime gains rather than quantization-only gains. For nonruntime
edits, `--allow-nonruntime-source-change` still requires identical runtime fingerprints.
Neither flag permits changed inference to inherit old accuracy.

Retained evidence is immutable. Remove obsolete trials only after checking selection,
calibration, paired-baseline, quality and provenance dependencies; keep every outcome
of retained campaigns. Invalid/superseded runs cannot be primary. Preserve compact
retirement reasons without backups or duplicate result tables. `--supersedes` is for
corrected implementations, not unfavorable results or valid ablations.

Token diagnostics distinguish retained context (`visual_tokens_final` /
`visual_kv_tokens`) from computed queries (`visual_query_tokens_final`).
Auxiliary work has separate counters but remains inside full-cost timing.

## Local checks

```bash
python -m pytest tests -m 'not gpu and not libero'
python scripts/check_boundaries.py
python scripts/check_private_paths.py
python scripts/check_repository.py
```

Run method/backend tests in their own runtime. OFT's Transformers fork is not CogACT's
stock Transformers. GPU/LIBERO checks need explicitly assigned resources; lightweight
CI alone does not certify model correctness or benchmark performance.
