# Development guide

Read [AGENTS.md](../AGENTS.md) for workflow rules and [README](../README.md) for
installation and usage. Backend-native geometry and numerical conventions live
with the [OFT](../backends/openvla_oft/README.md) and
[CogACT](../backends/cogact/README.md) plugins; paper adaptations live with methods.

## Architecture

| Component | Owns | Must not own |
|---|---|---|
| core | types, lifecycle, lazy discovery, config validation, provenance | model internals or paper decisions |
| model_ops (optional) | shared tensor/vision inspection and bounded storage | backend imports, paper policy or simulator loops |
| backend plugin | native loading, model access, preprocessing/actions, synchronization | paper algorithms or LIBERO loops |
| method | one paper's declared mechanism, compatibility and per-instance state | another method or environment stepping |
| control | action queue, executed history and scheduling callbacks | paper/backend-name branches |
| calibration | read-only training-observation sampling | quantizer-specific search/packing |
| evaluation | environment protocol, seeds, stopping and outcomes | concrete model or method imports |
| profiling | timing boundaries, replay and aggregation | model internals or policy changes |

Backend plugins depend on core. A selected method binding may depend on its backend,
but shared method equations must not. Entry-point and method-metadata discovery is
lightweight: import the runtime only when selected. No fallback from an unsupported
method to baseline. External plugins can register without modifying core.

`backends/<id>/upstream/` owns the pinned native source; the adjacent installable
plugin owns our adaptation. `third_party/` separates reference methods, runtime
dependencies and simulators. Only implemented backends are installable/registered.
The predicted chunk and executed prefix are distinct; neither dimensions nor
layer counts belong in the shared controller.

## Policy and method lifecycle

Policies expose reset, observe, predict_chunk, after_action, metadata and close.
`MethodHost` delegates the selected method's lifecycle. Hooks and caches are
instance-local; reset clears episode state and close restores hooks/resources.
An instruction change clears the queue and resets the predictor before observing
the new task. API-v1 predictors without optional control hooks keep native behavior.

1. `observe(observation)` receives every control frame, not only model-query frames.
2. At an empty queue, `before_plan(observation, step)` returns None or exactly one
   environment-ready replacement action. The first step after reset always queries
   the model to establish fresh state; this replacement hook is not consulted.
3. A real query invokes `on_plan(full_chunk_copy, step)` before prefix selection.
   `step` is the absolute control index. Predictions are not executed history.
4. `transform_action(action_copy, step)` may aggregate predictions, then
   `after_action(final_action_copy, remaining)` receives the actual emitted command.
   Returning true requests clearing the remaining queue before the next step.
5. `control_diagnostics()` reports counters without changing behavior.

Hooks never invoke inference or the environment. Actions at this boundary already
use the backend's environment convention. A method requiring a special representation
declares a capability and backend codec; core assumes neither a gripper index nor
seven dimensions. Nonfinite/invalid hook outputs fail before environment execution.
`queried` and query counts mean served model calls, not replacements or queue refills.
Discarded auxiliary native calls are reported separately and included in total work.

A method may request one same-observation auxiliary pass through
`wants_auxiliary_query()`. The backend owns execution; begin/end callbacks only
select the method's auxiliary mode, and its action output is never served.
The instance-local worker is serialized before the next native query/reset/detach.
Its RNG scope restores stochastic-head noise. One policy owns the process; concurrent
policies sharing global framework RNG in the same process are unsupported.
The CUDA monitor drains the worker before returning, so common replay timing includes
ALL refresh cost. Primary-forward latency, auxiliary latency/join wait and auxiliary
call counts are separate diagnostics. Simulation stepping can overlap refresh, but
unpaced replay is not proof of real-robot deadline feasibility.

## Model-access boundaries

- OFT's [decoder contract](../backends/openvla_oft/DECODER.md) exposes current-layer
  hidden states only during callbacks. A boolean keep mask or `TokenReduction`
  can select/merge visual rows; original positions and nonvisual/action readout
  stay protected. Do not retain full transient hidden states across queries.
- CogACT's `CausalTokenAdapter` keeps native attention, original RoPE positions,
  causal/padding visibility and final cognition EOS. It supports fresh batch-one
  prefill, not arbitrary past-KV decoding or OFT action-slot reconstruction.
  Both token adapters accept the shared `model_ops.token_reduction.TokenReduction` result
  (defined in `model_ops/token_reduction.py`). Merged features replace visual
  rows only. A causal merge must anchor each group at or after its latest member;
  the method owns this grouping constraint, not a relaxed attention mask.
- CogACT's separate `attach_reuse_strategy()` retains indexed visual KV while
  skipping selected query/MLP rows. All nonvisual rows remain active. Sparse queries
  use original causal rows against retained keys; errors invalidate caches and
  close restores attention. Suppressing the unused append cache of the single
  cognition call does not permit changes to generation/readout or DiT sampling.
- `VisionAttentionProbe` provides reversible per-view Q/K inspection without
  changing attention output. Paired vision grouping preserves patch membership,
  prefix/register tokens and anchors across SigLIP/DINO. The OFT/CogACT bridge
  restores native patch slots for projection; do not claim dense projector savings.
  `PairedVisionTransform` owns grouping/mass bookkeeping; backend bindings only
  unpack native per-camera images. Both branches must share original patch members,
  not merely the same number of anchors. CogACT exposes `attach_vision_strategy()`.
- CogACT `policy_image()` exposes native spatial RGB transforms before normalization,
  not an extra crop or reconstruction from normalized pixels.
- `model_ops/` is optional tensor infrastructure. It may import Torch/Transformers,
  but never backends or paper policy; discovery/core/harnesses may not import it.
- `NativeAttentionProbe` reads selected native Llama Q/K rows without replacing
  attention output. It preserves RoPE and causal/padding visibility; statistics
  normalize over all current keys. Its transient tensors clear after each call.
- CogACT exposes `action_denoiser` under `dit_block_features`. A method may
  reversibly cache per-block outputs; it must retain native sampling, timestep
  conditioning, CFG branches and residual paths. Denoiser calls are not policy
  queries: action-query boundaries must clear any within-sampling feature cache.

Full-retention/no-op adapters must match the appropriate native reference and
detach cleanly. Eager-versus-fused attention and quantized kernel precision
differences must be declared, not hidden behind an assertion of bitwise parity.

## Add a method

Create `src/efficient_vla_benchmark/methods/<category>/<paper_id>/`:

- `README.md`: references/pinned revisions, core scope, adaptations, limitations,
  supported variants and validation commands.
- `metadata.yaml`: schema version, ID/category, config, explicit implementation
  mappings, variants/capabilities and lazy factories.
- `config.py`: types/defaults/ranges; backend-specific layer/token bounds belong
  in that binding's preflight.
- `implementations/<backend>/`: factory and backend-specific behavior.
- Shared equation/controller modules only where actually shared; no empty scaffolds.
- Independently evaluated core contributions may add `components/<scope>.yaml`
  in the same paper directory, with ID `<paper_id>_<scope>` and explicit supported
  backends, category and exclusions. This selects ONE core, not a combination;
  it never duplicates the paper directory or silently upgrades partial coverage.
  If no complete/default method is supported, owner `metadata.yaml` may declare
  `components_only: true` with its schema/name/category and no implementations.
  At least one valid component is required; the owner itself is not selectable.
- `tests/`: independent mechanism references, reset/detach, unsupported inputs
  and real model/kernel checks. Optional backend tests skip if that plugin is absent.

Factory `(backend, method_config, RuntimeResources)` returns one `InferenceMethod`.
Preflight `(method_config, RuntimeResources)` rejects missing conditions before
model loading. Optional `validate_experiment(ExperimentConfig, method_config)`
rejects incompatible cadence/protocol; calibration uses the factory signature
and produces separate artifacts from read-only training data.

Method callbacks include reset, observe, before_query, after_query and after_action,
plus the optional control hooks above. Weight conversion declares reversible=false.
Add `configs/methods/<paper_id>.yaml` and compatible experiment presets. To port a
second backend, extend the same method's implementation map, not its directory.

Examples: ADP/FlashVLA share scorers with separate native bindings; VOTE shares its
codec and physical-time committee; SP-VLA shares pruning and executed-action
scheduling; VLA-Cache shares complete statistics/reuse policy; AWQ separates native
decoder targets from common calibration/packing. Read their tests and READMEs.
[External method template](../templates/method/README.md) demonstrates the
`efficient_vla_benchmark.methods` entry point, not a useful acceleration algorithm.

## Add a backend

Use the [backend template](../templates/backend/README.md) as an interface example.
Rename distribution `efficient-vla-backend-<name>`, package
`efficient_vla_backend_<name>` and entry-point ID `<name>`.

1. Register API-v1 `BackendSpec` with a lazy factory/validator, explicit variants
   and capabilities. Validate settings before expensive loading.
2. Implement native policy lifecycle and `RuntimeMonitor`, including framework
   RNG/state reset, synchronization and idempotent close.
3. Expose only APIs needed by real methods. Preserve native processors, layers,
   token layout and action conventions; different families need not share internals.
4. Add compatible method bindings explicitly, backend/experiment configs, installation
   recipe and pinned dependency closure. Conflicting runtimes use separate environments.
5. Test native parity, action normalization, reset, checkpoint immutability, repeated
   loading and real predicted/executed horizons; use the same LIBERO/timing harness.

A registered name or CPU fixture is not validated model support. Pin native changes
through fetchable submodule revisions; never depend on another developer's checkout.

## Calibration, evaluation and timing

Calibration reads the four `*_no_noops/1.0.0` training datasets in caller-supplied
`modified_libero_rlds`. Seeded index strata choose distinct episodes equally across
suites and one seeded step per episode; sample count is divisible by four. RLDS RGB
already has policy orientation: do not rotate it again. Preserve actual raw state,
language and sample/dataset content digests, never private episode paths. TFDS episode
indices are not LIBERO task IDs (calibration task ID is -1). Backend preprocessing
remains native. Algorithm search/packing belongs to the method; no evaluation-sample
fitting, gradient training or silent data download is implied.

LIBERO smoke uses task 0 / initial state 0 / seed 0 in each suite, ten settling steps,
then limits of 220/280/300/520 control steps for Spatial/Object/Goal/Long. CUDA/EGL
selection must agree. Outputs retain metadata, episodes (including errors), summary,
optional traces and private path configuration. Identical-run resume skips recorded
episodes; changed identities are rejected. Preserve a failed run when fixing it.
Formal mode (`--formal`, `--tasks`, `--suites`, `--trials`) requires owner scheduling.

Depth-based methods require real aligned metric depth: enable `evaluation.depth`
or use `evaluate --depth`. Schema-3 traces preserve this modality; no synthetic
depth is inferred from old RGB-only traces. The native model still receives only
its original inputs. Depth-aware methods read the retained observation context.

Timing defaults to full control replay: four complete schema-2/3 traces, ten warmup
queries and three repeats, with fresh episode state. Use one otherwise idle GPU,
fresh policy/model per method, fixed CPU threads and synchronized measurements.
It includes preprocessing, transfers, all method/control work and postprocessing;
it excludes loading, disk I/O and simulation. Warmup history is reset before timing.
Query/prepared modes are diagnostics; prepared excludes input preparation/transfer.
Neither can replace full replay for history-dependent methods.

Record query/first/steady latency, percentiles, repeats, total/control latency,
queries/replacements and allocated/reserved memory. Compare matched cadence for
aggregation, including VOTE H1. Checkpoint/source/trace hashes, dependencies,
attention/precision and driver/kernel identities accompany every result. Dirty
measurements keep their real status. Replay is not closed-loop completion time;
four smoke episodes never establish formal accuracy.

## Record results

Keep canonical JSON records in ignored `outputs/result_records/`, raw runs under
their run IDs, and only the generated summary in the current Git tree. Obtain the
complete local record collection from the owner
before regenerating the complete table. An empty record directory is an error.

```bash
python scripts/record_result.py export \
  --run-dir "$TIMING_DIR" --baseline-dir "$BASELINE_TIMING_DIR" \
  --smoke-dir "$SMOKE_DIR" \
  --section '5.3 Layer Skipping, Early Exit, and Model Routing' \
  --label 'SP-VLA pruning + scheduling' \
  --preset configs/experiments/cogact/sp_vla.yaml

python scripts/record_result.py render --output docs/benchmark_results.md
```

The default records directory is `outputs/result_records/`; `--records-dir` may
select another ignored local directory. Export verifies counts, sums and compatible
checkpoint/source/runtime/hardware/traces/protocol. Omit `--smoke-dir` if unavailable.
Each summary row retains a run ID, paired reference and record digest. Records are
immutable: do not replace earlier evidence or manually edit generated metrics.

The pre-publication history cleanup removed retired documents and exported JSON.
[scripts/history_commit_map.json](../scripts/history_commit_map.json) maps all original
commits to cleaned revisions; the one removed documentation-only commit resolves to
its unchanged-source ancestor. Rendering shows both IDs without modifying evidence.
Original source fingerprints and dirty flags remain authoritative; a mapped clean
checkout cannot reconstruct uncommitted changes from a dirty measurement.

Use `--supersedes OLD_RUN_ID` only for a corrected implementation of the same
backend/variant/method. Rendering keeps obsolete rows visibly marked. Do not
supersede a valid ablation or slower precision variant. Historical cohorts remain
separate. Negative results and failed episodes stay visible.

Token diagnostics count served query rows, including dense served warmup/refresh,
not queued actions. Auxiliary passes have separate full-context/call/timing evidence.
`visual_tokens_final` describes retained context, not full FLOPs;
`visual_query_tokens_final` with `visual_kv_tokens` describes computation reuse
with retained context. Do not put the latter in a context-pruning table.

## Local checks

```bash
python -m pytest tests -m 'not gpu and not libero'
python scripts/check_boundaries.py
python scripts/check_private_paths.py
python scripts/check_repository.py
```

Run method/backend tests in the selected runtime, with GPU tests only when explicitly
selected resources are available. OFT-specific tests require its Transformers fork;
CogACT bindings have separate `test_cogact.py` suites. Do not interpret an OFT-only
reference under stock Transformers as a CogACT correctness test.
CI checks lightweight tests and scoped first-party
lint/format; it does not install all backends or certify GPU/robot performance.
