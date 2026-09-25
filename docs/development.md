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
episodes; changed identities are rejected. Preserve a failed run during diagnosis;
after resolution apply the dependency-aware retirement policy below.
Formal mode (`--formal`, `--tasks`, `--suites`, `--trials`) requires owner scheduling.

### Fixed development validation

`evaluation-manifest` freezes `libero_dev200_v1`: five seeded samples without
replacement from initial-state indices 0–49 for each of forty tasks. Task order,
BDDL/init-state digests and policy seeds are explicit. It is a tuning subset of
the full 2,000-episode protocol, not an independent final test. Retain the remaining
1,800 episodes as untuned evidence and disclose overlap in later full-set reports.

```bash
evb evaluation-manifest --libero-root "$LIBERO_ROOT" --output "$MANIFEST"
evb evaluate --config "$PRESET" --model-path "$MODEL_DIR" \
  --libero-root "$LIBERO_ROOT" --episode-manifest "$MANIFEST" \
  --shard-index 0 --num-shards 8 --output-dir "$RUN_DIR" --no-record-traces
```

The manifest owns episode selection; `--formal`, task/trial/seed overrides cannot
be combined with it. Shards retain original initial-state IDs and policy seeds;
every algorithm uses the same manifest. Runtime-source, active backend dependency,
extension, checkpoint and effective-config identities protect resume/aggregation.
Full source revisions and hardware remain recorded separately for every shard.
No framework, algorithm or backend branch belongs in sampling or aggregation.

`evb derive-manifest --development-manifest "$MANIFEST" --purpose timing --output
"$TIMING_MANIFEST"` freezes one episode per task (the lowest selected Dev initial-
state ID), before examining outcomes. `--purpose final` instead freezes all fifty
states per task. Derivation never runs a policy. Timing-reference rollouts may
record real depth with `--depth --record-traces`; failed tasks remain in the workload.
A final-manifest rollout additionally requires `--authorize-final` after owner
scheduling. The common shard/merge path keeps timing, Dev and final kinds distinct.

`configs/evaluation/dev200_campaign.yaml` defines the original development campaign,
including both AWQ kernel variants; it is NOT the final report selection manifest.
`scripts/evaluation_results.py plan` generates portable jobs;
deployment, scheduling and environment bootstrap are private operational tools,
kept outside Git with caller-owned resource inventories. Validate package origins,
CUDA-extension identities and matched real-model episodes before admitting another
host. One coordinator owns campaign state; inspect interrupted jobs before resuming.
Machine-local paths are runtime inputs only, never committed deployment settings.

The local launcher must resolve the assigned GPU to a UUID and map its EGL device;
never assume EGL enumeration equals the GPU index shown by `nvidia-smi`. It keeps
one CUDA-visible device and defers the EGL override until the pinned robosuite
binding has imported, avoiding its incorrect cross-namespace index assertion.
Its import-scoped log redirection restores the logging factory immediately and
keeps legacy simulator logs under the run's `_private/`, not a shared global file.
No model/simulator equations or installed packages are patched. Verify rendered
RGB/depth and real-policy parity when changing launch/bootstrap behavior.

`scripts/evaluation_results.py merge` rejects duplicate, foreign or incompatible
episodes and reports missing episodes/errors. A comparable Dev-200 success rate
requires all 200 records and zero runtime errors. Normal unsuccessful episodes
remain failures; infrastructure retries preserve the exact episode and attempt
history. Concurrency wall time is not a policy-speed benchmark. Keep raw evidence
ignored and add development success columns to the existing result ledger only.

For historical diagnostics only, render a separate ignored ledger with
`scripts/record_result.py render`, supplying
`--development-summary` and `--timing-selection` together. The latter is a local
JSON map from `backend/preset` to an explicitly audited historical timing run ID;
choose the current valid implementation cohort, never the fastest repeat. The join
accepts repeated pairs of `--development-summary` / `--timing-selection` in round
order, preserving the original table alongside later bounded diagnostic rounds.
Retain unsuccessful candidates while they support selection or diagnosis; retire
unneeded ones under the policy below. A changed parameter configuration needs its own
paired timing measurement; never attach its parent's speed to the new preset.
The join
requires matching checkpoint, precision, method parameters and execution cadence,
and refuses superseded/incomplete evidence. Timing and closed-loop source revisions
remain separate: a metadata match does not certify source equivalence. Dev-200
delta uses the same backend's matched-cadence baseline. Do not overwrite old timing
records or promote parallel evaluation wall time to a latency measurement.

Depth-based methods require real aligned metric depth: enable `evaluation.depth`
or use `evaluate --depth`. Schema-3 traces preserve this modality; no synthetic
depth is inferred from old RGB-only traces. The native model still receives only
its original inputs. Depth-aware methods read the retained observation context.

Existing development timing uses full control replay: four complete schema-2/3 traces, ten warmup
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

### Selection and final-measurement workflow

1. Establish fidelity: identify the paper's core, compare executable reference or
   explicit equations, test active behavior and native/reset boundaries, and label
   independent cores and unverified conventions. Run the common smoke; runtime
   completion alone is neither correctness nor accuracy validation.
2. Evaluate the existing candidate on the fixed Dev-200 manifest. Reuse compatible,
   complete evidence; do not rerun merely because documentation changed. All four
   suites share one parameter configuration. Calibration still uses training data.
3. Investigate only warranted cases: unexplained accuracy loss, poor full-cost
   latency, almost-inactive mechanisms or unresolved fidelity. Select informative
   changes to the implicated parameters; no fixed candidate quota or exhaustive
   sweep. A mechanism change is a new implementation audit, not just tuning.
4. Selection target: `Dev_SR >= matched_baseline_Dev_SR - 3 percentage points`.
   On the current H8 references this is at least 192/200 for OFT and 177/200 for
   CogACT. These counts follow the observed baseline, not a universal constant.
   Prefer faster full-cost candidates meeting the target; differences within
   timing variability are ties, with higher success and then the simpler/reference-
   closer setting preferred. This is a development decision, not a significance test.
5. If no tested active candidate meets the target, or none accelerates, do not
   exclude a faithfully implemented mechanism for its outcome. Stop when further
   useful tuning is not justified; nominate a representative, preferably reference-
   close or accuracy-recovered, active configuration with a recorded reason. An
   unresolved core cannot be certified simply because a favorable preset was found.
   Record every attempted measurement during this process; keep only useful evidence
   after decisions settle. An intentional schedule/precision change stays explicit.
6. Freeze ONE primary configuration per backend/method scope, its selection evidence,
   calibration identity if applicable, checkpoint, runtime and evaluation protocol.
   Valuable auxiliary configurations may remain separately, without automatically
   scheduling extra full campaigns. No separate best preset for each suite.
7. Following owner scheduling, run all 2,000 fixed episodes once for each frozen
   primary and matched baseline. Never tune using those outcomes, replace failures
   or choose the best repeat. Extract the untuned 1,800-episode subset from THAT SAME
   run for internal overlap-aware analysis; retain the original Dev-200 selection
   evidence separately. Do not assemble a final result from old Dev runs plus 1,800
   new episodes. Infrastructure resume is allowed with exact identities; a real
   implementation bug invalidating a final run needs an explicit owner decision.
8. Publish one primary row joining its Dev selection reason, actual final success,
   separately measured full-cost speed/memory, scope and evidence IDs. Low success
   and slowdowns remain reportable. A dash means not measured, never zero or baseline.

The formal timing target is a fixed **40-task replay workload**, one deterministic
native-baseline trace per task drawn by ID from the existing Dev manifest, never
selected by favorable success or speed. Freeze IDs, content hashes and length rules
before comparing methods; record each backend's own traces, shared by its methods.
Depth-aware methods and their paired native reference need the same real depth-bearing
traces. Keep the same CPU/GPU/runtime conditions and existing full-cost boundary,
with warmup reset, repeated timing and native pre/post checks for drift/throttling.
Compute mean ms/control within each task, macro-average the 40 task means, then
divide the native macro-mean by the method macro-mean for speedup. Do not average
task speedup ratios or weight tasks by their rollout lengths. Retain repeat-level
variation; an approximately 1x historical result is not proof of acceleration.

VOTE uses a separately labeled H1 protocol and H1 native reference, not the H8
ranking. AWQ precision/memory and offline calibration costs remain explicit. ASR's
auxiliary refresh stays in total compute timing; real-robot overlap/serve latency
is a different, currently unmeasured claim.

**Implementation status (2026-09-25):** Dev-200 sampling/sharding/merging, derived
40-task/final manifests, hashed timing workloads, task-macro aggregation, explicit
primary-report joins and same-run final-2,000/untuned-1,800 partitioning are implemented
and tested. Native timing references and the additional Dev candidates have completed;
quality-controlled timing and primary selection are being populated. Existing
four-trace records remain development evidence, not completed measurements under
the new timing protocol. No final campaign has been launched.

Create each fixed workload with `scripts/timing_workload.py` from the complete
native timing-reference shards. Pass it to `benchmark --workload-manifest`; missing,
changed or extra traces are rejected. `scripts/readiness_report.py` consumes an
explicit selection, retained Dev summaries, timing records and accepted native
pre/post quality groups. It refuses missing evidence, unknown runtime changes,
configuration/checkpoint/cadence mismatches and multiple primary presets for one
scope. It rechecks the actual 200 outcomes and seeds, backend metadata, active
dependency/extension identities and coverage of the frozen reviewed campaign.
Its launch list is descriptive only; it never launches final evaluations.
`final_partition()` accepts only one complete final-2000 aggregate and derives the
untuned 1800 from those same episode outcomes, not an older Dev run.

The readiness source audit reconstructed runtime identities at `0cc594e`, `1d954ec`,
`557dac8` and `83cd166`. Their 307 backend/core/control/model-ops/method source files
are identical (SHA256 `c118623603d159cb90bbc445e91d2e7cc33672cfc07856f4afd552348b402ef4`).
Changed runtime hashes concern evaluation/profiling/CLI infrastructure; algorithm,
native backend and controller logic did not change. Cross-version Dev reuse requires
this explicit scoped audit plus identical effective policy configuration, checkpoint
and method/artifact metadata. It is not a general permission to ignore code identity.

## Record results

Keep canonical JSON records in ignored `outputs/result_records/`, raw runs under
their run IDs, and only the summary in the current Git tree. The primary table is
rendered by `scripts/readiness_report.py` from
`configs/evaluation/primary_selection.yaml`, grouped by backend and labeled by
survey section. Keep the actual Dev count/rate, selection reason, scope and exact
timing record; a matched native pair belongs to each speedup, not to the whole table.
`record_result.py render` enumerates historical records and cannot overwrite the
primary table once its selection manifest exists. Obtain the retained dependency-
complete collection before regeneration; an empty directory is an error.
For historical diagnostics, the dated audit between
`benchmark-review:start/end` comment markers is hand-maintained and preserved by
the renderer, separately from generated measurements. Refresh its decisions when
evidence changes; regeneration alone does not revalidate them.

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
  --output docs/benchmark_results.md --evidence-output "$READINESS_JSON"
```

Repeat `--development-summary` for every referenced selection/coverage summary.
Historical or diagnostic rendering must use a separate ignored output, not the
primary table. Neither rendering command executes a policy or schedules a campaign.

The default records directory is `outputs/result_records/`; `--records-dir` may
select another ignored local directory. Export verifies counts, sums and compatible
checkpoint/source/runtime/hardware/traces/protocol. Omit `--smoke-dir` if unavailable.
Each summary row retains a run ID, paired reference and record digest. Records are
immutable while retained: do not rewrite earlier evidence or manually edit generated
metrics. Immutability does not require indefinite retention of retired experiments.

The pre-publication history cleanup removed retired documents and exported JSON.
[scripts/history_commit_map.json](../scripts/history_commit_map.json) maps all original
commits to cleaned revisions; the one removed documentation-only commit resolves to
its unchanged-source ancestor. Rendering shows both IDs without modifying evidence.
Original source fingerprints and dirty flags remain authoritative; a mapped clean
checkout cannot reconstruct uncommitted changes from a dirty measurement.

Use `--supersedes OLD_RUN_ID` only for a corrected implementation of the same
backend/variant/method. The legacy renderer marks those rows while retained; the
final primary view must omit retired/invalid rows. Do not supersede a valid
ablation or slower precision variant. A lower score is not an invalid measurement.

Keep primary, useful auxiliary and still-open candidate evidence distinct. Before
deleting anything, resolve dependencies from selection decisions, paired baselines,
native drift rechecks, timing-quality checks, calibration, source identities and
campaign manifests. The absence of a record from a main table is NOT proof it is
unused. Retained campaigns keep every episode, including failures and warnings.
Invalid timings and superseded buggy runs cannot be selected as current results.
After their causes and replacements are recorded, genuinely unnecessary logs,
traces, duplicates and retired records may be removed; no extra archive or backup.
Keep only compact run IDs/reasons where history is needed, not redundant full tables.
Never break hashes or references by editing retained raw records to simulate cleanup.

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
