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
- `ForwardPatches` wraps named callable attributes on individual module/sampler
  instances and restores whether the attribute originally existed. It never patches
  a model class. Methods own cache policy, shape guards and lifecycle; shared code
  owns only reversible binding. Visual attention reuse preserves native grid/MLPs,
  while denoiser output and cross-chunk residual caches are separate methods.
- A candidate-generation method may expand the native action-sampler batch after
  one cognition encoding. Preserve conditional/unconditional pairing, timestep
  convention and native postprocessing; return one real selected chunk. Shared
  context is not K repeated VLM passes. All candidates and selection cost belong
  in full-cost timing. KeyStone owns its geometric selection within
  `methods/action_generation/keystone/`; it does not change the DDIM schedule.
- V-AEFusion overlaps a separate action-head stream with the current native VLM
  call. Previous cognition conditions only a fixed early denoising window; the
  fresh tail remains compulsory. Its private worker RNG cannot fork/restore global
  RNG concurrently. Native draws, sampler steps and all joins remain accounted for.
  CUDA stream-interval overlap is diagnostic, not proof of simultaneous kernels
  or reduced end-to-end latency. Reset/close join work before releasing state.
- An adaptive solver port must preserve the native finite-noise initial state,
  conditioning, CFG pairing and action interface. ProbeFlow first verifies its
  DDIM-to-linear-mixture coordinate bridge on the original grid; its adaptive grid
  is then a separate accuracy question. Count discarded lookahead evaluations and
  all coordinate/RNG overhead. Fixed-grid diagnostic controls are not algorithm
  results, and an epsilon-prediction model is not relabeled as FM-trained.

Full-retention/no-op adapters must match the appropriate native reference and
detach cleanly. Eager-versus-fused attention and quantized kernel precision
differences must be declared, not hidden behind an assertion of bitwise parity.

`model_ops.indexed_linear` provides real indexed row/column GEMMs and fused
gate/up/SiLU/product kernels. It owns no channel-ranking policy. EcoVLA owns
coupled FFN/head masks, trigger history and its interleaved statistics stream;
all stream joins belong inside the shared full-cost timing boundary. Tests compare
FP16/BF16 kernels against materialized native linear algebra. Token count is a
non-specializing runtime argument: warmed kernels must not recompile for another
instruction length. GPU regression tests check bitwise static/dynamic parity and
cache-size invariance. Pre-fix EcoVLA timings are retired, not mixed into selection;
development runs keep their actual source hashes under an explicit arithmetic
compatibility audit. Triton JIT requires
matching Python development headers; machine-specific include locations stay in
ignored environment launchers, never public configuration.

EfficientVLA layer pruning physically shortens the decoder and offloads removed
weights; reversible detach restores modules, indices and native depth. Its static
ranking and EcoVLA's initial energies use separate checksummed training-only
calibration artifacts. Retention/depth candidates reuse the same frozen artifact.
ASR KV reuse keeps an immutable complete dense base: sparse active-row updates
are temporary, and only the full auxiliary pass publishes the next base/gate.

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

## Training-dependent candidate boundary

These audited candidates are NOT registered implementations. Their core requires
learned parameters or independently trained control policies; a random/rule-based
stand-in would change the contribution. Large fine-tuning remains owner-scheduled.
Published costs below are source recipes, not estimates for our joint checkpoint.

| Candidate / intended first backend | Core missing without training | Verified cost or unresolved condition |
|---|---|---|
| [LAC](https://arxiv.org/html/2602.00686v1) / OFT, then CogACT | RAFT-small-conditioned selector + ratio predictor; attention alignment then task-loss training | Backbone frozen, but stage-II gradients pass through it; no verified duration or complete executable recipe |
| [DySL-VLA](https://github.com/PKU-SEC-Lab/DYSL_VLA/tree/cabc0d8bbd2f90807695182a502e64160efeaa42/DySL-VLA-openvla-oft) / OFT | Learned skip adapters and router, static anchors, action-consistency verification | Default spatial script resumes at 150k and stops at 350005; roughly 200k additional optimizer updates, not a short router calibration |
| [MoLe-VLA](https://github.com/RoyZry98/MoLe-VLA-Pytorch/tree/8ae45f8bd513e9901528ff92468dd08f3ab309f6) / CogACT | STAR router, top-k layers, task/distillation/load-balancing losses | Checked launcher trains LLM, batch 512, 100 epochs on eight GPUs; not router-only |
| [DeeR-VLA](https://arxiv.org/html/2411.02359v1) / OFT | Trained multi-exit shared temporal head, auxiliary exits, randomized exits and threshold optimization | Original 3B recipes: 14–25 h on eight V100s; native final head at intermediate layers is not equivalent |
| [A1 MLP core](https://arxiv.org/html/2604.05672v2) / OFT | Shared-head random-depth training and early-exit criterion | Published LIBERO fine-tuning: 50k steps, batch 128; FM variant is not a drop-in CogACT DDIM port |
| [ActDistill](https://arxiv.org/html/2511.18082v1) / CogACT, then OFT | Graph teacher probes plus jointly trained student/router | One teacher-probe epoch, then five end-to-end student epochs; not merely a frozen-backbone gate |
| [MoDE](https://arxiv.org/html/2412.12953v1) / CogACT | Learned noise-conditioned expert denoisers before routing-cache/fusion | Original LIBERO-10 recipe: 50 epochs, four A6000s, about two hours; joint-40-task CogACT cost unmeasured |
| [ElegantVLA](https://arxiv.org/html/2605.29438v1) / CogACT | Two-stage Maskable PPO scheduler over backbone and denoiser reuse | Requires independent training rollouts; verified rollout budget/weights unavailable; teacher rules alone are not the method |

These are training/budget deferrals, not claims that the native backends are
fundamentally incompatible. Revisit after agreeing a train/validation split,
compute budget and trained-artifact contract; never use Dev/final outcomes as
training rewards. Keep SP-VLA's already integrated scope separate from this queue.

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

ActionCache uses an additional, separately authorized native population: two fresh
seeded placements per task, distinct from ALL fifty standard evaluation states.
The method-specific collector calls the same H8 episode runner; it does not create
another success protocol. Commit pairs only after trajectory success, retain every
collection outcome, then freeze and checksum the bank before Dev/timing.
The freezing command rechecks saved states and complete pair accounting. Failed
population episodes are not replaced. Evaluation never updates this bank. Report
population cost separately from steady-state lookup/refinement; an 80-episode bank
is not free training data or part of the final 2,000-episode success measurement.

`evaluation.libero.independent` freezes caller-declared independent placement
groups before outcomes are observed. It excludes all fifty standard states per
task and cross-group physical duplicates (simulator time does not create a new
placement). Search quick/confirmation groups use this separate population, never
Dev/final outcomes. The runner calls the same episode loop, preserves failures,
verifies resource/state hashes, and refuses silent retries. Search algorithms own
their fitness and authorized episode budget; this shared utility owns neither.

EVO's prior/native reference, quick search and confirmation groups are all disjoint.
The native rollouts supply only initialization dissimilarity and the reference
success target; rollout fitness drives schedule selection. Reserve the complete
episode allocation before dispatch, including failures. Freeze and checksum layouts
before Dev/timing. Report bounded-search target failures honestly, not as convergence.
First/final branch refreshes count toward the actual global budget; native DDIM,
CFG and action/control conventions stay unchanged.

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
"$TIMING_MANIFEST"` freezes FOUR episodes: task 0 in each suite, using its lowest
selected Dev initial-state ID before examining outcomes. `--purpose final` freezes all fifty
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

Freeze each worker checkout's source for the entire queued campaign. Adding an
unrelated method still changes the global runtime identity of later shards.
Do not synchronize new code into a running worker or rewrite recorded identities.
If this happens, repeat the affected shards under one frozen source; retain the
replacement reason and compare outcomes before retiring the invalid mixed cohort.

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

The standard timing protocol is `libero_replay4x128_v1`: four schema-3 native traces,
one per suite, each restricted to its first min(128, native length) controls. Select
the lowest task ID and its frozen reference episode by ID, never by outcome. Use
ten warmup queries and ONE measured pass, with fresh episode state. Use one otherwise idle GPU,
fresh policy/model per method, fixed CPU threads and synchronized measurements.
It includes preprocessing, transfers, all method/control work and postprocessing;
it excludes loading, disk I/O and simulation. Warmup history is reset before timing.
Query/prepared modes are diagnostics; prepared excludes input preparation/transfer.
Neither can replace stateful control replay for history-dependent methods.

Record query/first/steady latency, percentiles, total/control latency,
queries/replacements and allocated/reserved memory. Compare matched cadence for
aggregation at the fixed H8 comparison cadence. Checkpoint/source/trace hashes, dependencies,
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
3. Audit speedward headroom for every admitted backend/method scope, including
   candidates already meeting the accuracy target. Select meaningful stronger
   settings of the main compute parameter, then refine measured accuracy boundaries
   or inspect overhead plateaus. Investigate secondary parameters when justified;
   no mandatory candidate count or exhaustive Cartesian sweep. Mechanism changes
   require a fresh fidelity audit. Meeting the accuracy target alone never closes
   search. Freeze one effective configuration across all suites.
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
6. Record search closure with referenced candidate evidence: an accuracy boundary,
   latency plateau, concrete mechanism limit, or a reasonably explored no-feasible
   outcome. Lack of time/budget is unfinished, not a boundary. The report validates
   these references separately from measurement completeness and withholds unfinished
   rows from the final queue. Explain any selection over a faster eligible candidate
   under the existing timing-variability tie rule; never pick a favorable repeat.
   Freeze ONE primary configuration per backend/method scope, its selection evidence,
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

The timing target is the lightweight **four-prefix replay**, at most 512 controls.
The existing native reference bank supplies RGB/depth; no new simulator campaign
is necessary. All methods on a backend share the same frozen input/content hashes.
Average ALL measured control-step latencies, including queue-consumption steps;
divide the matched native mean by the method mean. This is representative-input
policy compute, not task-macro performance or closed-loop completion time. Do not
report repeat standard deviation from one pass, or mix historical protocols in the
primary table. The current primary comparison is H8; old H1 evidence is not eligible.

Use fresh model processes and fixed CPU threads. Concurrent timing processes need
disjoint CPU affinity sets as well as separate GPUs; remapping a GPU must also
revalidate its CPU assignment. Reject cohorts affected by resource contention.
A small batch on the same GPU may
share native pre/post references. Accept at most 3% native latency drift and no
observed thermal slowdown during measured replay intervals; loading, warmup and
trace-file decoding are outside those intervals. Inspect telemetry per run: a method's thermal failure excludes that
run, not its healthy neighbors, provided BOTH native references are valid. An
unstable reference invalidates the associated speedup until targeted remeasurement.
If only the post-reference is thermally rejected, remeasure that reference on the
same GPU with unchanged source/config/protocol, retaining the healthy pre-reference
and methods. Keep the rejected check and replacement reason in the audit; accept
the first valid replacement satisfying the drift bound, never the fastest repeat.
No power/clock/fan changes. Repeat only for an explained anomaly, never select the
fastest repeat. A >10% speedup difference from compatible old evidence triggers
diagnosis, not adjustment to match the old number. Prefer reaggregating the exact
same prefixes; cross-GPU absolute latencies are not directly comparable.

VOTE uses CogACT's native sixteen-predict/eight-execute overlap, normally two votes.
Its tie follows the original equation, selecting the older disagreeing prediction;
this sparse-committee adaptation is not the original dense multi-vote setting.
OFT's eight-step predictions do not overlap at H8, so that binding is unsupported.
No H1 baseline or extra model calls are substituted. AWQ precision/memory and offline
calibration costs remain explicit. ASR's
auxiliary refresh stays in total compute timing; real-robot overlap/serve latency
is a different, currently unmeasured claim.

**Measurement history:** The lightweight protocol replaces the old
full-trajectory repeated timing. Dev-200 evidence is reused only for the same policy,
checkpoint and audited implementation. The generated result ledger is authoritative
for current scope counts, selected configurations and search closure. Historical H1
rows are not H8 evidence; unresolved cores and auxiliaries are not default final-run
candidates. Final measurement still requires separate owner scheduling.

The completed protocol transition compared 37 compatible old/new configurations:
four used identical reaggregated prefixes, the others had explicitly different
reference episodes/lengths. No speedup change exceeded the 10% diagnostic trigger;
three new configurations had no compatible retained old timing. The ignored
comparison/retirement audits retain these distinctions after obsolete timings are
removed. Native input provenance, Dev outcomes and calibration remain available.

Create each compact workload with `scripts/timing_workload.py --reference-workload
"$REFERENCE_WORKLOAD" --output "$WORKLOAD"`, or supply complete native reference
shards using `--manifest` and `--run-dirs`. Both routes produce the SAME compact
protocol; the full reference format remains readable for historical audits only.
Pass it to `benchmark --workload-manifest "$WORKLOAD" --repeats 1 --warmup 10`; missing,
changed or extra traces are rejected. `scripts/readiness_report.py` consumes an
explicit selection, retained Dev summaries, timing records and accepted native
pre/post quality groups. It refuses missing evidence, unknown runtime changes,
configuration/checkpoint/cadence mismatches and multiple primary presets for one
scope. It rechecks the actual 200 outcomes and seeds, backend metadata, active
dependency/extension identities and coverage of the frozen reviewed campaign.
Its launch list is descriptive only; it never launches final evaluations.
Useful precision/accuracy alternatives belong in `auxiliary_configurations` and
undergo the same evidence checks, but never enter the default final launch list.
They must differ from the primary and belong to an already certified method scope.
`final_partition()` accepts only one complete final-2000 aggregate and derives the
untuned 1800 from those same episode outcomes, not an older Dev run.

The readiness source audit reconstructed runtime identities at `0cc594e`, `1d954ec`,
`557dac8` and `83cd166`. Their 307 backend/core/control/model-ops/method source files
are identical (SHA256 `c118623603d159cb90bbc445e91d2e7cc33672cfc07856f4afd552348b402ef4`).
Changed runtime hashes concern evaluation/profiling/CLI infrastructure; algorithm,
native backend and controller logic did not change. Cross-version Dev reuse requires
this explicit scoped audit plus identical effective policy configuration, checkpoint
and method/artifact metadata. It is not a general permission to ignore code identity.
The compact-timing revision `e6a9117` was checked against the same baseline: no
backend/core/control/model-ops/method Python sources changed. The new runtime
identity is `b30a41d426f30c6f8e7731d6ab35fe56ea52ac691f601faf83c2daff72e9c099`.
The subsequent native-four capture and measured-interval boundary changes through
`2a49223` likewise leave all policy sources unchanged; their runtime identity is
`fb5c9a52ab1f503f3ad5b8aca8a23b63f8fd395c73adbb95a624f6156b8bb63b`.

The caching batch through `d3c46d3` only adds isolated method files and
`ForwardPatches`; all existing native/backend/control/harness Python files are
unchanged from `b293e89`. Existing native Dev evidence therefore remains the
reference. New method Dev/timing pairs retain their actual runtime identities;
no measurement revision or dirty-source flag is relabeled as a later clean commit.

The H8 voting change at `e185205` changes only VOTE policy sources relative to
the preceding selection. All non-VOTE method, backend and shared control/model
sources remain identical; compatible prior Dev/timing evidence is retained for
those scopes. VOTE receives fresh native-H8 evidence, never relabeled H1 results.
Search-only configuration and reporting changes do not alter inference equations.

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
The primary ledger replaces historical candidate tables. Component READMEs retain
mechanism limitations; ignored comparison and retirement audits retain replacement
reasons without leaving obsolete results available for accidental selection.

A later batch may use a compact `review_scope_coverage` manifest containing only
the effective configurations of all reviewed scopes. Its digest pins coverage;
it contains no imputed outcomes. Supply it alongside the actual complete Dev
summaries when regenerating the primary ledger. Every measured row still requires
its own strictly validated 200 episode records and matched timing evidence.

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
