# efficient-vla-benchmark

A modular benchmark for VLA acceleration on pinned native backends. One run uses
one method or the native baseline; algorithms share evaluation and timing harnesses.
This repository contains code, configuration and summarized results, not model
weights, datasets or raw experiment artifacts.

## Supported backends

| Backend / variant | Token reduction | Caching | Model routing | Action chunking | Quantization |
|---|---|---|---|---|---|
| OpenVLA-OFT / libero_joint | Nine methods below | VLA-Cache, VLA-InfoEntropy | SP-VLA | VOTE ensemble core | AWQ W4A16 |
| CogACT / libero_joint | ADP, FlashVLA ICS, SAFE-Pruner, ST-Merge, SpecPrune-VLA | VLA-Cache, VLA-InfoEntropy, EfficientVLA DiT cache core | SP-VLA | VOTE ensemble core | AWQ W4A16 |
| starVLA | Source pinned only; runtime not implemented | — | — | — | — |

OFT token methods: SpecPrune-VLA, ADP, VLA-Pruner, SAFE-Pruner, TEAM-VLA, VLA-IAP,
EfficientVLA token core, ST-Merge and FlashVLA ICS. SP-VLA includes pruning and
scheduling in one method. Kernel precision options and scheduling-only ablations
are not additional algorithms.

These are common-checkpoint adaptations, not reproductions of published scores.
TEAM-VLA, VLA-IAP and ST-Merge retain unverified paper interpretation choices.
EfficientVLA exposes token pruning on OFT and a separately named DiT cache core
on CogACT; neither includes layer pruning or the complete pipeline. FlashVLA excludes unresolved
action reuse; VOTE implements the independent ensemble, not its trained architecture.
No action-generation sampler is currently advertised. Read each
[method README](src/efficient_vla_benchmark/methods/) before using its results.

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

evb benchmark --config configs/experiments/openvla_oft/specprune_vla.yaml \
  --model-path "$MODEL_DIR" \
  --traces "$SPATIAL_TRACE" "$OBJECT_TRACE" "$GOAL_TRACE" "$LONG_TRACE" \
  --output-dir "$TIMING_DIR"
```

The caller defines the resource variables. Change the experiment YAML to select a
supported backend/method pair and parameters; AWQ also needs `--artifact-path`.
LIBERO defaults to four smoke episodes, not a formal success-rate campaign.
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
[Benchmark results](docs/benchmark_results.md) retains measured regressions,
failed smoke episodes, configuration variants and provenance. Raw records remain
local under `outputs/result_records/`, outside the current tracked tree.
Do not equate replay speed, mechanism tests or four-episode smoke with formal
accuracy or closed-loop task completion time.

This is a private development repository without a selected first-party
open-source license. Preserve [LICENSE](LICENSE), [NOTICE](NOTICE) and upstream
licenses; publication and license selection require the owner's decision.
