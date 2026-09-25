# efficient-vla-benchmark

A modular benchmark for VLA acceleration on pinned native backends. One run uses
one method or the native baseline; algorithms share evaluation and timing harnesses.
This repository contains code, configuration and summarized results, not model
weights, datasets or raw experiment artifacts.

## Supported backends

| Backend / variant | Token reduction | Caching | Model routing | Action chunking | Quantization |
|---|---|---|---|---|---|
| OpenVLA-OFT / libero_joint | Fifteen scopes below | VLA-Cache, VLA-InfoEntropy | SP-VLA | VOTE ensemble core | AWQ W4A16 |
| CogACT / libero_joint | Eleven scopes below | VLA-Cache, VLA-InfoEntropy, EfficientVLA DiT cache core | SP-VLA | VOTE ensemble core | AWQ W4A16 |
| starVLA | Source pinned only; runtime not implemented | — | — | — | — |

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
EfficientVLA exposes token pruning on OFT and a separately named DiT cache core
on CogACT; neither includes layer pruning or the complete pipeline. FlashVLA excludes unresolved
action reuse; VOTE implements the independent ensemble, not its trained architecture.
DepthCache requires measured metric depth and excludes unresolved grasp/wrist
control. ASR includes every auxiliary dense refresh in reported total cost;
its primary-forward diagnostic is not end-to-end serving latency. SQAP-VLA remains
deferred: activation quantization granularity is contradictory and no verifiable
implementation is available. It is not counted as supported.
No action-generation sampler is currently advertised. Read each
[method README](src/efficient_vla_benchmark/methods/) before using its results.

CogACT ST-Merge's current preset has 0/4 development-smoke successes (native 3/4),
without runtime errors. It is a measured negative configuration, not accuracy-
validated support. Per-preset outcomes, including newer batches, are in the ledger. These checks
are not formal success-rate measurements.

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
supported backend/method pair and parameters; AWQ also needs `--artifact-path`.
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
Do not equate replay speed, mechanism tests or four-episode smoke with formal
accuracy or closed-loop task completion time.

This is a private development repository without a selected first-party
open-source license. Preserve [LICENSE](LICENSE), [NOTICE](NOTICE) and upstream
licenses; publication and license selection require the owner's decision.
