# Development contract

Read this file, [docs/development.md](docs/development.md), the selected backend
README and the relevant method README before editing. This is the single source
of development and Git rules; upstream submodules retain their own instructions.

## Scope and architecture

- Keep the benchmark independent of the survey manuscript. One method or native
  baseline per run; never compose independent papers to manufacture a gain.
- Admit a port only after identifying and testing its core mechanism. An independent
  core contribution is allowed with explicit exclusions in its ID, README and
  metadata. An unresolved core rule requires deferral, not a runnable proxy.
- One canonical directory per method under its principal category. Add backend
  bindings there, not copies of the algorithm. Cross-category labels are metadata.
- Methods never import other methods. Backends never import paper-specific code.
  Shared control/evaluation/profiling never branch on backend or method names.
- Core/discovery/harnesses work without Torch, Transformers or simulator packages.
  Optional model operations may depend on tensor runtimes but not on backends or
  papers. Incompatible backend dependencies belong in separate environments.
- State is per policy instance and resets between episodes/instruction changes.
  Hooks must be reversible and close idempotent. Quantization needs a fresh native
  model to return to baseline; relabeling a converted model does not undo it.
- Native mask visibility, token positions, action readout and preprocessing are
  backend contracts. Supporting one model never implies all methods are compatible.
  Declare supported variants/capabilities and reject unsupported pairs explicitly.
- Python packages/IDs use snake_case; distribution/branch names use kebab-case.
  Prefer descriptive modules over catch-all utils. Typed config owns validation;
  YAML exposes portable parameters, never arbitrary unchecked keys.

## Resources and evidence

- Pass every external resource location through CLI arguments. Never commit private
  filesystem paths, host/network identities, credentials or environment dumps.
- Keep machine-specific launchers, deployment/scheduling scripts and ad-hoc smoke
  drivers in ignored local runtime storage, not Git. Commit reusable harnesses,
  portable result tools and generic tests; never force-add private operational files.
- Original checkpoints/datasets are read-only. Derived calibration artifacts stay
  separate. No global site-package edits or global model-class monkey patches.
- Keep weights, datasets, traces, logs, environments, binaries and exported result
  JSON outside Git. Canonical result records live in ignored `outputs/result_records/`;
  raw runs stay under their output IDs. Do not recreate `docs/results/`.
- Commit the generated `docs/benchmark_results.md`, including negative outcomes,
  actual measurement revisions and paired baselines. Retain evidence needed by
  selected results and their selection/validity checks, not every experiment forever.
  An older smoke source must never be relabeled as the final timing source.
- Use the common harness and actual native speed reference. Include preprocessing,
  transfers, selection/cache/control work and postprocessing. Report precision,
  cadence, hardware, protocol and full versus partial mechanism scope.
- Four-episode smoke is not formal success rate. Preserve unsuccessful episodes and
  errors; do not retry/tune until success. Full success campaigns, training, broad
  sweeps and large downloads require explicit user scheduling/authority.
- Preserve third-party notices and pinned fetchable upstream revisions. Do not
  invent a license or describe unverified paper conventions as author-code parity.

## Selection and reporting contract

- One primary configuration per backend/method scope, shared across four suites.
  Use the fixed Dev-200 subset for selection, not the final 2,000 outcomes. Tune
  selectively when evidence warrants it; no mandatory number of candidate presets.
- Target at most 3 percentage points of Dev-200 success loss against the matched
  native baseline, then prefer reliably faster full-cost execution among tested
  eligible configurations. This is a selection goal, NOT an admission/reporting filter.
  Never disable the core or substitute a baseline no-op to obtain eligibility.
- Meeting the accuracy target is NOT search completion. Inspect every admitted
  scope for speedward headroom; test meaningful stronger configurations and record
  a measured accuracy boundary, latency plateau or concrete mechanism limit.
  Measurement-complete but search-incomplete rows cannot enter the final queue.
  Keep the current timing and Dev protocols unchanged; no best-of-repeat selection.
- Primary comparisons use native H8. CogACT VOTE is a disclosed two-vote overlapping-
  chunk adaptation; OFT VOTE is unsupported without overlapping native predictions.
  Historical H1 evidence must never be relabeled as an H8 result.
- A tested, core-faithful port may report low success, no speedup or a slowdown.
  Check implementation, scope and reasonable parameter choices first. Say what
  the tested configuration shows; do not infer that the published method fails.
  Unresolved core definitions remain provisional, not certified negative findings.
- Freeze configuration/checkpoint/runtime/protocol identities before the owner's
  scheduled final 2,000-episode run. Run it once; do not retune or select the best
  final repeat. Infrastructure resume must preserve episode identity and failures.
- Final-2000 is a separate approval gate after the owner reviews the frozen Dev
  and timing results. General permission to continue, use idle GPUs, or complete
  integration is NOT final-run authorization. Never auto-chain Dev into Final.
  This quantization repair batch is Dev/timing-only; prior prematurely launched
  quantization Final outcomes are withdrawn from publication, not tuning evidence.
- Keep one clean primary result row, with Dev selection evidence, final success
  and separately measured speed. Valuable auxiliary configurations are explicitly
  secondary, never additional default report rows. Missing measurements stay missing.
- Retire genuinely obsolete results after checking baseline, calibration, selection,
  validity and provenance dependencies. Remove unnecessary raw artifacts rather than
  creating archives/backups; keep compact retirement reasons and the evidence still
  needed to support live results. Never erase failures from a retained campaign.

## Git and workspace lifecycle

1. Use an existing authorized checkout. Never create additional clones, worktrees,
   staging workspaces, recovery copies, bundles or backups without explicit approval.
   This also applies to clean-install tests. Canonical experiment outputs are not backups.
2. When the owner explicitly requests collaborator setup, fork/clone as agreed.
   Use `origin` for the fork and `upstream` for the shared project; initialize only
   needed submodules. Creating a remote fork or pushing requires authorization.
3. Branch from current `main`: `feat/<component>-<task>`, `fix/<component>-<issue>`,
   `refactor/<component>-<task>` or `docs/<task>`. Keep one feature per commit;
   shared infrastructure is a separate earlier commit.
4. Commit format: `type(scope): imperative summary`, no trailing period.
   Types: feat, fix, refactor, test, docs, build, chore. Use the owning component
   as scope. Nontrivial bodies contain `Why:`, `Changes:`, `Validation:` and
   `Limitations:`; report checks actually run, never planned checks or private paths.
5. Upstream modifications require their own submodule branch/commit and an
   authorized fetchable source before updating the parent gitlink. No personal
   filesystem remotes or hidden dependencies on another checkout.
6. Validate, then merge into `main`, preferably fast-forward. Verify the branch
   tip is reachable from `main` and delete the completed task branch. Never
   force-delete unmerged work or rewrite/push shared history without permission.
7. Preserve unrelated user edits. No automatic pre-cleanup backups, Git clean,
   environment deletion, model/data removal or unrelated remote mutations.

## Validation and documentation

- Run relevant CPU tests, lint/format, import boundaries, private-path and repository
  hygiene checks. For algorithm/model changes also run mechanism/native-parity tests,
  real kernels, the agreed short LIBERO smoke and shared sequential timing.
- Test reset/detach, unsupported inputs, optional-dependency isolation and repeated
  construction. Weight conversion checks must cover reload and checkpoint immutability.
- Update config, compatibility metadata and the owning README with a feature.
  Use existing environments/checkouts; request authority for additional ones.
- Keep `docs/` limited to the development guide and result ledger. Update those or
  the relevant component README instead of adding per-turn/per-batch reports.
  Do not delete unresolved limitations merely to shorten documentation.
- Root README owns user entry points; this file owns workflow rules; the development
  guide owns technical interfaces. Avoid repeating the same normative content.

Current support is listed in README and validated by the lazy catalog. starVLA
remains source-only. FlashVLA action reuse and VLA-IAP's CogACT semantic construction
remain unresolved. Native tail refinement is supported only through ActionCache's
explicitly scoped frozen-bank port, not a trained few-step sampler. SQAP-VLA
is deferred pending activation-granularity clarification. DepthCache primary and
ASR deletion/KV reuse are explicitly scoped components, never full-paper reproductions;
do not silently promote excluded or unverified mechanisms to implemented status.
EcoVLA's dynamic channels/heads and EfficientVLA's static layer core require
training-only calibration artifacts. They do not compose with other methods.
A learned router/exit/expert cannot be replaced by untrained heuristics; the
training-dependent candidate audit in the development guide is not runtime support.
