# Development contract

Read this file, CONTRIBUTING.md, docs/architecture.md and the relevant backend and
method READMEs before changing code. Upstream submodules keep their own instructions.

## Branch and workspace lifecycle

- Develop on a short-lived task branch from current `main` in an existing authorized
  checkout. Keep one feature per commit and validate before merging.
- Merge completed work into `main`, preferably by fast-forward, preserving its
  commits. Confirm every branch tip is reachable from `main`, then delete the
  completed task branch. Do not retain merged branches or force-delete unmerged work.
- Do not create additional project clones, Git worktrees, staging workspaces,
  recovery copies or backups without the user's explicit request. This includes
  Git bundles, archive copies and automatic pre-cleanup backups.
- A clean-install test is not permission to create another checkout. Reuse an
  existing authorized checkout/environment or ask before creating one.
- Ordinary commits and canonical experiment outputs are not backups. Retain raw
  measurement evidence, logs, models, datasets and runtime environments unless
  their removal is separately authorized.

## Invariants

- Keep this repository independent of the survey manuscript.
- One method per run; do not compose independently published methods. An independently
  meaningful core contribution may be ported without the paper's other components,
  but its ID, README and runtime metadata must explicitly disclose that scope.
- One canonical method directory under its principal category. Cross-category labels
  are metadata, never duplicated implementations.
- Methods never import another method's implementation. Backend plugins never import
  paper-specific code. Shared harnesses never branch on backend or method names.
- Main-package installation, discovery and CPU contract tests must work without model
  plugins, PyTorch, Transformers or simulation dependencies installed.
- Backend dependencies are optional and may need separate environments. Never import
  incompatible upstream packages in one process or modify global site-packages.
- Model-specific computation lives in the backend or method implementation, not core.
- State is per policy instance and resets between episodes. Converted weights require
  a fresh baseline model; changing a method label cannot undo quantization.
- All external resource paths come from CLI arguments. No private paths, machine names,
  network addresses, credentials or environment dumps in source, commits or reports.
- Original checkpoints and datasets are read-only. Derived artifacts are separate.
- Preserve upstream revision/license information and explicitly report adaptations.
- Evaluation and profiling use the same policy protocol for every method. Include
  method selection, caching and control overhead in policy timing. Retain failures
  and negative speed results. Native inference is the speed reference.
- Full success-rate campaigns require explicit user scheduling. Development is limited
  to the agreed four-episode smoke set. No training or large downloads by implication.
- One feature per commit, with tests/config/docs. Shared infrastructure has separate
  commits. Do not describe partial mechanisms or smoke outcomes as paper reproductions.
- Every measured algorithm updates docs/benchmark_results.md through the audited
  result-recording workflow; preserve the paired baseline, source/config/raw hashes,
  negative outcomes and the distinction between smoke and formal success rate.
- No pushes, remote history replacement, new remote forks, or unrelated deletions
  without explicit authority.

## Before committing

Run relevant CPU tests, formatting, import-boundary and privacy checks. Model changes
also require real-model smoke checks; algorithm integrations additionally require the
shared timing harness. Record commands using placeholders, not personal paths.
Validate important boundaries without relying on untracked files, using an existing
authorized checkout. Additional validation checkouts require explicit permission.
Update documentation when behavior or the interface contract changes.

## Implemented scope

OpenVLA-OFT native baseline; nine token methods including EfficientVLA's token core,
ST-Merge and FlashVLA's ICS core; VLA-Cache and VLA-InfoEntropy; AWQ; SP-VLA visual
pruning plus scheduling; and VOTE's trajectory ensemble core. Independent paper-based
implementations disclose unverified conventions, not published-score reproductions.
Preserve these qualifications in reports. FlashVLA action reuse remains unresolved;
EfficientVLA excludes layer pruning and DiT caching; VOTE excludes its other paper
components. SP-VLA uses one canonical directory and offers scheduling-only as an
ablation preset, not a second method. CogACT has a native joint-suite baseline
with causal SDPA, a BF16 VLM and FP32 DiT-B; its existing OFT methods are not yet
ported. Preserve the single-camera input, suite statistics and sampler conventions
in its README. starVLA retains pinned sources for subsequent implementation; do
not advertise runtime support until its own model checks have passed. Do not
fabricate an empty plugin that only registers a name and cannot perform inference.
