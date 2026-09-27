# efficient-vla-benchmark

A modular benchmark for VLA acceleration on pinned native backends. One run uses
one method or the native baseline; algorithms share evaluation and timing harnesses.
This repository contains code, configuration and summarized results, not model
weights, datasets or raw experiment artifacts.

## Supported backends

| Backend / variant | Token reduction | Caching | Model routing | Action chunking | Quantization |
|---|---|---|---|---|---|
| OpenVLA-OFT / libero_joint | Fifteen scopes below | VLA-Cache, VLA-InfoEntropy, Temporal Redundancy visual core, ASR KV reuse | SP-VLA, EcoVLA | — | AWQ W4A16 |
| CogACT / libero_joint | Eleven scopes below | Above, EfficientVLA DiT cache, DP-Cache, C³ache conditioned residual, ActionCache, BAC | SP-VLA, EcoVLA | VOTE H8 two-vote adaptation | AWQ W4A16 |
| starVLA | Source pinned only; runtime not implemented | — | — | — | — |

Action generation additionally exposes KeyStone on CogACT: one shared VLM context,
batched independent native DDIM candidates and guarded cluster-medoid selection.
Its registration is not a measured acceleration claim; consult the result ledger
for completed measurements. Deterministic OFT L1 sampling is unsupported.

OFT token methods: SpecPrune-VLA, ADP, VLA-Pruner, SAFE-Pruner, TEAM-VLA, VLA-IAP,
EfficientVLA token core, ST-Merge and FlashVLA ICS, plus FastV, DivPrune, DART,
SparseVLM v1, DepthCache primary-view core and ASR deletion instrument.
CogACT supports the same token scopes except VLA-Pruner, TEAM-VLA, VLA-IAP and
EfficientVLA token core; its README explains those fidelity deferrals.
SP-VLA includes pruning and
scheduling in one method. Kernel precision options and scheduling-only ablations
are not additional algorithms.

These are common-checkpoint adaptations, not reproductions of published scores.
TEAM-VLA, VLA-IAP and ST-Merge retain unverified paper interpretation choices.
EfficientVLA exposes token pruning on OFT, a separately named DiT cache core
on CogACT, and calibrated static layer pruning on both. Each is an independent
scope, never the complete pipeline or a combination. Layer pruning belongs to
model compression, not dynamic routing. EcoVLA conditionally prunes coupled FFN
channels and whole attention heads with real sparse GEMMs, not whole layers.
FlashVLA excludes unresolved
action reuse; VOTE implements H8 overlapping-chunk voting, not its trained architecture.
Native CogACT provides two aligned votes after warmup; its equation-defined tie
selects the older prediction. OFT's eight-predict/eight-execute chunks do not
overlap, so VOTE is not registered there. No H1 exception enters the primary table.
DepthCache requires measured metric depth and excludes unresolved grasp/wrist
control. ASR includes every auxiliary dense refresh in reported total cost;
its primary-forward diagnostic is not end-to-end serving latency. SQAP-VLA remains
deferred: activation quantization granularity is contradictory and no verifiable
implementation is available. It is not counted as supported.
The new caches have distinct scopes: DP-Cache reuses native denoiser outputs;
C³ache retains a live conditioning prefix and reuses cross-chunk suffix residuals;
Temporal Redundancy reuses visual attention rows, excluding its trained action branch.
ActionCache retrieves independently collected, frozen intermediate actions and
refines them with the native DDIM tail; it is not a reproduction of flow-model scores.
No trained few-step sampler is advertised. Read each
[method README](src/efficient_vla_benchmark/methods/) before using its results.

Executable bindings do not by themselves certify fidelity. The result ledger
separates admitted method scopes from unresolved TEAM-VLA, VLA-IAP and ST-Merge
interpretations. Their diagnostic outcomes are not certified algorithm findings.

## Install and run

Install only the selected backend, following its pinned environment recipe:
[OpenVLA-OFT](backends/openvla_oft/INSTALL.md) or [CogACT](backends/cogact/INSTALL.md).
Their Transformers/runtime requirements differ; use separate environments.
For lightweight development without a model runtime:

```bash
python -m pip install -e '.[dev]'
evb list
evb validate-config --config configs/experiments/openvla_oft/specprune_vla.yaml
```

After backend installation, supply resource locations at launch:

```bash
evb evaluate --config configs/experiments/openvla_oft/baseline.yaml \
  --model-path "$MODEL_DIR" --libero-root third_party/benchmarks/libero \
  --output-dir "$SMOKE_DIR" --record-traces

python scripts/timing_workload.py --reference-workload "$REFERENCE_WORKLOAD" \
  --output "$WORKLOAD"

evb benchmark --config configs/experiments/openvla_oft/specprune_vla.yaml \
  --model-path "$MODEL_DIR" \
  --traces "$SPATIAL_TRACE" "$OBJECT_TRACE" "$GOAL_TRACE" "$LONG_TRACE" \
  --workload-manifest "$WORKLOAD" --repeats 1 --warmup 10 --output-dir "$TIMING_DIR"
```

The caller defines the resource variables. Change the experiment YAML to select a
supported backend/method pair and parameters; AWQ, ActionCache, EcoVLA and
EfficientVLA layer pruning also need
`--artifact-path`. BAC also requires frozen training-only branch statistics for its
ACS schedules and Bubbling Union protection. ActionCache requires its independently collected frozen bank;
see its [population instructions](src/efficient_vla_benchmark/methods/caching/action_cache/README.md).
Timing uses the four trace files selected by the frozen workload, not arbitrary
smoke observations: at most 128 continuous controls per suite, one measured pass.
Matched native references use the same inputs, GPU and execution cadence.
LIBERO defaults to four smoke episodes, not a formal success-rate campaign.
For depth-aware methods, record native baseline traces with `--depth` first;
their method presets enable depth during evaluation. RGB-only traces are rejected.
CUDA/EGL device selections must match. Closed-loop evaluation does not require
the offline training dataset. See the [development guide](docs/development.md)
for calibration, measurement boundaries and result recording.

## Repository layout

- `src/efficient_vla_benchmark/`: lightweight contracts and common harnesses;
  `methods/<category>/<method>/` owns each paper, with per-backend implementations.
- `backends/<backend>/`: optional integration package and pinned native `upstream/`.
- `configs/`: portable experiment presets; no machine-local resource locations.
- `third_party/`: pinned algorithm references, runtimes and LIBERO.
- `tests/`, `templates/`, `scripts/`: contracts, extension examples and maintenance tools.
- `docs/`: [development](docs/development.md) and [results](docs/benchmark_results.md).
- `outputs/`: ignored local experiments, calibration artifacts and result records.

## Development and results

Read [AGENTS.md](AGENTS.md) for the single development/Git contract.
[Development](docs/development.md) covers interfaces and extension workflows.
[Benchmark results](docs/benchmark_results.md) joins selected Dev-200 outcomes with
separately measured compact-replay latency; final 2,000-episode success remains unmeasured.
The [selection workflow](docs/development.md#selection-and-final-measurement-workflow)
targets one primary configuration per backend/method; poor faithful results remain
reportable. Useful raw evidence stays local under `outputs/result_records/`, outside Git;
unneeded results are retired after dependency checks, not archived indefinitely.
Use [primary_selection.yaml](configs/evaluation/primary_selection.yaml) to select
the currently frozen report presets. Generic method presets retain reference or
initial migration defaults; they are not necessarily the selected hyperparameters.
Do not equate replay speed, mechanism tests or four-episode smoke with formal
accuracy or closed-loop task completion time.

This is a private development repository without a selected first-party
open-source license. Preserve [LICENSE](LICENSE), [NOTICE](NOTICE) and upstream
licenses; publication and license selection require the owner's decision.
