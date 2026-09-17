# Algorithm measurement ledger

Generated from the versioned JSON evidence in `results/`. This is a development
measurement ledger, not a final paper comparison or a formal success-rate table.

Backend, hardware and protocol are recorded per row below. Policy timing
includes preprocessing, transfer, all method/control overhead and postprocessing,
but not model loading, trace I/O or simulation. Fixed replay is not closed-loop
task completion time. Different protocol cohorts must not be pooled.

Speedup = paired native baseline total replay time / method total replay time.
H is the executed prefix, distinct from the backend's full prediction length.
The present OFT VOTE preset uses its matched H=1 baseline; the other OFT presets
use native H=8. Do not compare their speedup
ratios as if query cadence were identical. Peak GiB is runtime allocated tensor
memory, not loading peak or total process memory. Smoke is successes/episodes,
not a success-rate estimate. Formal success rate is **not measured for any row**.

## 0. Native references

| Method / configuration | Backend | H | Queries / replacements | Query mean / p95 ms | ms/control step | Replay speedup | Peak GiB | Smoke |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [Native BF16 / SDPA](results/control-methods-final-benchmark-baseline-001.json) | openvla_oft | 8 | 216 / 0 | 197.86 / 199.86 | 25.10 | 1.000× | 14.98 | 4/4; errors 0 |
| [Native BF16 / SDPA, matched H=1](results/control-methods-final-benchmark-baseline_horizon1-001.json) | openvla_oft | 1 | 1704 / 0 | 199.83 / 201.42 | 199.83 | 1.000× | 14.98 | Not run |
| [Native BF16 / SDPA, end recheck](results/control-methods-final-benchmark-baseline_recheck-001.json) | openvla_oft | 8 | 216 / 0 | 198.91 / 201.15 | 25.23 | 0.995× | 14.98 | Not run |

## 5.1 Token Pruning and Context Compression

| Method / configuration | Backend | H | Queries / replacements | Query mean / p95 ms | ms/control step | Replay speedup | Peak GiB | Smoke |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [SpecPrune-VLA](results/control-methods-final-benchmark-specprune_vla-001.json) | openvla_oft | 8 | 237 / 0 | 146.91 / 163.52 | 20.51 | 1.224× | 14.60 | 3/4; errors 0 |

## 5.2 Caching and Computation Reuse

| Method / configuration | Backend | H | Queries / replacements | Query mean / p95 ms | ms/control step | Replay speedup | Peak GiB | Smoke |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [VLA-Cache, full statistics / eager](results/control-methods-final-benchmark-vla_cache-001.json) | openvla_oft | 8 | 216 / 0 | 207.64 / 219.38 | 26.34 | 0.953× | 15.03 | 4/4; errors 0 |

## 5.3 Layer Skipping, Early Exit, and Model Routing

| Method / configuration | Backend | H | Queries / replacements | Query mean / p95 ms | ms/control step | Replay speedup | Peak GiB | Smoke |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [SP-VLA scheduling core](results/control-methods-final-benchmark-sp_vla_scheduling-001.json) | openvla_oft | 8 | 201 / 153 | 197.51 / 199.26 | 23.34 | 1.076× | 14.98 | 4/4; errors 0 |

## 5.5 Action Chunking and Replanning

| Method / configuration | Backend | H | Queries / replacements | Query mean / p95 ms | ms/control step | Replay speedup | Peak GiB | Smoke |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [VOTE ensemble core](results/control-methods-final-benchmark-vote_ensemble-001.json) | openvla_oft | 1 | 1704 / 0 | 199.62 / 201.47 | 199.62 | 1.001× | 14.98 | 3/4; errors 0 |

## 6.1 Quantization and Low-Precision Inference

| Method / configuration | Backend | H | Queries / replacements | Query mean / p95 ms | ms/control step | Replay speedup | Peak GiB | Smoke |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [AWQ W4A16, BF16 kernel](results/control-methods-final-benchmark-awq-001.json) | openvla_oft | 8 | 216 / 0 | 204.95 / 206.69 | 26.00 | 0.965× | 6.12 | 4/4; errors 0 |
| [AWQ W4A16, explicit FP16 kernel](results/control-methods-final-benchmark-awq_fp16_kernel-001.json) | openvla_oft | 8 | 216 / 0 | 170.32 / 171.73 | 21.61 | 1.162× | 6.12 | 4/4; errors 0 |

## Traceability

Raw files remain under `outputs/<run ID>/`; the linked JSON preserves their
hashes, full metadata, summary, paired baseline and optional smoke records.
Model/trace/artifact identities, precision, kernel, dependencies, source
fingerprint, dirty flag and parameters are in those records. A smoke from an
earlier source revision is explicitly marked, never silently relabeled.

| Method | Preset | Timing commit | Baseline run |
|---|---|---|---|
| AWQ W4A16, BF16 kernel | [awq](../configs/experiments/openvla_oft/awq.yaml) | `be5be7d8e0e1` | `control-methods-final-benchmark-baseline-001` |
| AWQ W4A16, explicit FP16 kernel | [awq_fp16_kernel](../configs/experiments/openvla_oft/awq_fp16_kernel.yaml) | `be5be7d8e0e1` | `control-methods-final-benchmark-baseline-001` |
| Native BF16 / SDPA | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `be5be7d8e0e1` | `control-methods-final-benchmark-baseline-001` |
| Native BF16 / SDPA, matched H=1 | [baseline_horizon1](../configs/experiments/openvla_oft/baseline_horizon1.yaml) | `be5be7d8e0e1` | `control-methods-final-benchmark-baseline_horizon1-001` |
| Native BF16 / SDPA, end recheck | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `be5be7d8e0e1` | `control-methods-final-benchmark-baseline-001` |
| SP-VLA scheduling core | [sp_vla_scheduling](../configs/experiments/openvla_oft/sp_vla_scheduling.yaml) | `be5be7d8e0e1` | `control-methods-final-benchmark-baseline-001` |
| SpecPrune-VLA | [specprune_vla](../configs/experiments/openvla_oft/specprune_vla.yaml) | `be5be7d8e0e1` | `control-methods-final-benchmark-baseline-001` |
| VLA-Cache, full statistics / eager | [vla_cache](../configs/experiments/openvla_oft/vla_cache.yaml) | `be5be7d8e0e1` | `control-methods-final-benchmark-baseline-001` |
| VOTE ensemble core | [vote_ensemble](../configs/experiments/openvla_oft/vote_ensemble.yaml) | `be5be7d8e0e1` | `control-methods-final-benchmark-baseline_horizon1-001` |

## Timing protocols

| Method | Hardware | Measured frames | Repeats | Warmup queries |
|---|---|---:|---:|---:|
| AWQ W4A16, BF16 kernel | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| AWQ W4A16, explicit FP16 kernel | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| Native BF16 / SDPA | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| Native BF16 / SDPA, matched H=1 | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| Native BF16 / SDPA, end recheck | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| SP-VLA scheduling core | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| SpecPrune-VLA | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| VLA-Cache, full statistics / eager | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| VOTE ensemble core | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |

## Interpretation boundaries

- SP-VLA means scheduling core only: no token pruning. Its fitted actions can
  reduce query count, but success loss still requires formal evaluation.
- VOTE means trajectory ensemble only: no ACT-token/head retraining and no
  claim to reproduce the full paper's acceleration. H=1 costs substantially
  more than native H=8 even if the voting overhead itself is small.
- AWQ's FP16-kernel row is mixed precision, not the default BF16 kernel;
  boundary casts are timed. Accuracy equivalence is not established.
- Cache regressions, failed smoke episodes and all precision variants remain
  visible. Historical runs in other validation reports are separate cohorts.
