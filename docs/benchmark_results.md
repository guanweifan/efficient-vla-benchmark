# Algorithm measurement ledger

Generated from the versioned JSON evidence in `results/`. This is a development
measurement ledger, not a final paper comparison or a formal success-rate table.
Superseded rows preserve earlier, corrected implementations for traceability;
do not use them as results of the current implementation. Their raw records
are unchanged. Mechanism scope and limits are detailed in the method READMEs.

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
| [Native H=8 (token batch 2)](results/batch2-final-benchmark-baseline-001.json) | openvla_oft | 8 | 216 / 0 | 197.31 / 199.10 | 25.03 | 1.000× | 14.98 | Not run |
| [Native H=8 end recheck (token batch 2)](results/batch2-final-benchmark-baseline_recheck-001.json) | openvla_oft | 8 | 216 / 0 | 198.63 / 200.64 | 25.20 | 0.993× | 14.98 | Not run |
| [CogACT native DDIM / primary runtime](results/cogact-final-benchmark-baseline-001.json) | cogact | 8 | 321 / 0 | 183.69 / 184.81 | 23.39 | 1.000× | 14.73 | 3/4; errors 0 |
| [CogACT native DDIM / secondary runtime](results/cogact-secondary-benchmark-baseline-001.json) | cogact | 8 | 321 / 0 | 194.39 / 204.01 | 24.75 | 1.000× | 14.73 | 3/4; errors 0 |
| [CogACT native DDIM / secondary recheck (thermal throttling)](results/cogact-secondary-benchmark-recheck-001.json) | cogact | 8 | 321 / 0 | 195.57 / 209.94 | 24.90 | 1.000× | 14.73 | 3/4; errors 0 |
| [CogACT native H=8 (token batch 1)](results/cogact-token-batch1-benchmark-baseline-001.json) | cogact | 8 | 321 / 0 | 183.86 / 184.84 | 23.41 | 1.000× | 14.73 | 3/4; errors 0 |
| [CogACT native H=8 end recheck (token batch 1)](results/cogact-token-batch1-benchmark-baseline_recheck-001.json) | cogact | 8 | 321 / 0 | 184.36 / 185.67 | 23.47 | 0.997× | 14.73 | Not run |
| [Native BF16 / SDPA](results/control-methods-final-benchmark-baseline-001.json) | openvla_oft | 8 | 216 / 0 | 197.86 / 199.86 | 25.10 | 1.000× | 14.98 | 4/4; errors 0 |
| [Native BF16 / SDPA, matched H=1](results/control-methods-final-benchmark-baseline_horizon1-001.json) | openvla_oft | 1 | 1704 / 0 | 199.83 / 201.42 | 199.83 | 1.000× | 14.98 | Not run |
| [Native BF16 / SDPA, end recheck](results/control-methods-final-benchmark-baseline_recheck-001.json) | openvla_oft | 8 | 216 / 0 | 198.91 / 201.15 | 25.23 | 0.995× | 14.98 | Not run |
| [Native H=8 (fidelity audit)](results/fidelity-final-benchmark-baseline-001.json) | openvla_oft | 8 | 216 / 0 | 196.74 / 198.44 | 24.96 | 1.000× | 14.98 | Not run |
| [Native H=8 end recheck (fidelity audit)](results/fidelity-final-benchmark-baseline_recheck-001.json) | openvla_oft | 8 | 216 / 0 | 197.84 / 199.24 | 25.10 | 0.994× | 14.98 | Not run |
| [Native H=8 (token batch 1)](results/token-batch-final-benchmark-baseline-001.json) | openvla_oft | 8 | 216 / 0 | 200.57 / 203.81 | 25.44 | 1.000× | 14.98 | Not run |
| [Native H=8 end recheck (token batch 1)](results/token-batch-final-benchmark-baseline_recheck-001.json) | openvla_oft | 8 | 216 / 0 | 198.77 / 200.57 | 25.21 | 1.009× | 14.98 | Not run |

## 5.1 Token Pruning and Context Compression

| Method / configuration | Backend | H | Queries / replacements | Query mean / p95 ms | ms/control step | Replay speedup | Peak GiB | Smoke |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [Superseded — EfficientVLA token core (paper-based)](results/batch2-final-benchmark-efficient_vla-001.json) | openvla_oft | 8 | 216 / 0 | 170.08 / 171.33 | 21.58 | 1.160× | 14.62 | 4/4; errors 0 |
| [FlashVLA ICS core (paper-based)](results/batch2-final-benchmark-flash_vla-001.json) | openvla_oft | 8 | 216 / 0 | 238.64 / 241.15 | 30.27 | 0.827× | 14.64 | 4/4; errors 0 |
| [Superseded — ST-Merge spatial VLA (paper-based)](results/batch2-final-benchmark-st_merge-001.json) | openvla_oft | 8 | 216 / 0 | 171.42 / 172.94 | 21.75 | 1.151× | 14.62 | 4/4; errors 0 |
| [ADP / CogACT (observed EEF)](results/cogact-token-batch1-benchmark-adp-001.json) | cogact | 8 | 321 / 0 | 168.27 / 184.16 | 21.44 | 1.092× | 14.73 | 4/4; errors 0 |
| [FlashVLA ICS / CogACT (complete SVD)](results/cogact-token-batch1-benchmark-flash_vla-001.json) | cogact | 8 | 321 / 0 | 157.56 / 166.44 | 20.07 | 1.167× | 14.66 | 4/4; errors 0 |
| [SpecPrune-VLA](results/control-methods-final-benchmark-specprune_vla-001.json) | openvla_oft | 8 | 237 / 0 | 146.91 / 163.52 | 20.51 | 1.224× | 14.60 | 3/4; errors 0 |
| [ADP / control recheck](results/fidelity-final-benchmark-adp-001.json) | openvla_oft | 8 | 216 / 0 | 193.63 / 198.37 | 24.58 | 1.015× | 14.68 | 4/4; errors 0 |
| [EfficientVLA token core / corrected Q-K direction](results/fidelity-final-benchmark-efficient_vla-001.json) | openvla_oft | 8 | 216 / 0 | 170.19 / 172.02 | 21.59 | 1.156× | 14.63 | 4/4; errors 0 |
| [FlashVLA ICS / automatic complete SVD](results/fidelity-final-benchmark-flash_vla-001.json) | openvla_oft | 8 | 216 / 0 | 208.73 / 210.37 | 26.48 | 0.943× | 14.64 | 4/4; errors 0 |
| [FlashVLA ICS / gesvd reference](results/fidelity-final-benchmark-flash_vla_gesvd-001.json) | openvla_oft | 8 | 216 / 0 | 237.86 / 240.28 | 30.17 | 0.827× | 14.64 | Not run |
| [SAFE-Pruner / control recheck](results/fidelity-final-benchmark-safe_pruner-001.json) | openvla_oft | 8 | 216 / 0 | 146.76 / 198.59 | 18.62 | 1.340× | 14.68 | 4/4; errors 0 |
| [SpecPrune-VLA / control recheck](results/fidelity-final-benchmark-specprune_vla-001.json) | openvla_oft | 8 | 237 / 0 | 146.13 / 162.17 | 20.42 | 1.222× | 14.60 | 3/4; errors 0 |
| [ST-Merge spatial VLA / key dot product](results/fidelity-final-benchmark-st_merge-001.json) | openvla_oft | 8 | 216 / 0 | 171.36 / 172.73 | 21.74 | 1.148× | 14.62 | 3/4; errors 0 |
| [TEAM-VLA / audited paper-based port](results/fidelity-final-benchmark-team_vla-001.json) | openvla_oft | 8 | 216 / 0 | 139.58 / 148.03 | 17.71 | 1.409× | 14.61 | 1/4; errors 0 |
| [VLA-IAP / audited paper-based port](results/fidelity-final-benchmark-vla_iap-001.json) | openvla_oft | 8 | 216 / 0 | 193.23 / 200.12 | 24.51 | 1.018× | 14.69 | 3/4; errors 0 |
| [VLA-Pruner / control recheck](results/fidelity-final-benchmark-vla_pruner-001.json) | openvla_oft | 8 | 216 / 0 | 163.52 / 198.05 | 20.75 | 1.203× | 14.68 | 4/4; errors 0 |
| [ADP](results/token-batch-final-benchmark-adp-001.json) | openvla_oft | 8 | 216 / 0 | 195.37 / 200.98 | 24.80 | 1.026× | 14.68 | 4/4; errors 0 |
| [SAFE-Pruner](results/token-batch-final-benchmark-safe_pruner-001.json) | openvla_oft | 8 | 216 / 0 | 148.28 / 200.16 | 18.82 | 1.352× | 14.68 | 4/4; errors 0 |
| [SpecPrune-VLA (token batch 1)](results/token-batch-final-benchmark-specprune_vla-001.json) | openvla_oft | 8 | 237 / 0 | 147.06 / 163.44 | 20.53 | 1.239× | 14.60 | 3/4; errors 0 |
| [TEAM-VLA (paper-based)](results/token-batch-final-benchmark-team_vla-001.json) | openvla_oft | 8 | 216 / 0 | 139.47 / 148.07 | 17.70 | 1.438× | 14.61 | 1/4; errors 0 |
| [VLA-IAP (paper-based)](results/token-batch-final-benchmark-vla_iap-001.json) | openvla_oft | 8 | 216 / 0 | 196.31 / 204.09 | 24.90 | 1.022× | 14.69 | 3/4; errors 0 |
| [VLA-Pruner](results/token-batch-final-benchmark-vla_pruner-001.json) | openvla_oft | 8 | 216 / 0 | 165.05 / 199.81 | 20.94 | 1.215× | 14.68 | 4/4; errors 0 |

## 5.2 Caching and Computation Reuse

| Method / configuration | Backend | H | Queries / replacements | Query mean / p95 ms | ms/control step | Replay speedup | Peak GiB | Smoke |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [VLA-Cache full statistics / eager (token batch 2)](results/batch2-final-benchmark-vla_cache-001.json) | openvla_oft | 8 | 216 / 0 | 207.43 / 217.93 | 26.31 | 0.951× | 15.03 | Not run |
| [VLA-InfoEntropy KV reuse (paper-based)](results/batch2-final-benchmark-vla_infoentropy-001.json) | openvla_oft | 8 | 216 / 0 | 213.14 / 222.39 | 27.04 | 0.926× | 15.03 | 4/4; errors 0 |
| [VLA-Cache, full statistics / eager](results/control-methods-final-benchmark-vla_cache-001.json) | openvla_oft | 8 | 216 / 0 | 207.64 / 219.38 | 26.34 | 0.953× | 15.03 | 4/4; errors 0 |
| [VLA-Cache full statistics / control recheck](results/fidelity-final-benchmark-vla_cache-001.json) | openvla_oft | 8 | 216 / 0 | 206.24 / 216.85 | 26.16 | 0.954× | 15.03 | 4/4; errors 0 |
| [VLA-InfoEntropy / audited entropy and KV reuse](results/fidelity-final-benchmark-vla_infoentropy-001.json) | openvla_oft | 8 | 216 / 0 | 213.47 / 222.80 | 27.08 | 0.922× | 15.03 | 4/4; errors 0 |

## 5.3 Layer Skipping, Early Exit, and Model Routing

| Method / configuration | Backend | H | Queries / replacements | Query mean / p95 ms | ms/control step | Replay speedup | Peak GiB | Smoke |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [SP-VLA pruning + scheduling](results/batch2-final-benchmark-sp_vla-001.json) | openvla_oft | 8 | 201 / 153 | 189.63 / 201.81 | 22.41 | 1.117× | 14.68 | 4/4; errors 0 |
| [SP-VLA scheduling-only ablation (token batch 2)](results/batch2-final-benchmark-sp_vla_scheduling-001.json) | openvla_oft | 8 | 201 / 153 | 197.93 / 200.72 | 23.39 | 1.070× | 14.98 | Not run |
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

## Observed visual-token retention

Query-weighted measured counts, including warmup/refresh queries; these
are not configured keep ratios or FLOPs. Final counts do not describe
earlier-layer cost. Per-layer sequence lengths remain in raw diagnostics.

| Method | Final mean / min / max | Early mean | Mode counts (true/observed) |
|---|---:|---:|---|
| Superseded — EfficientVLA token core (paper-based) | 256.0 / 256 / 256 | — | — |
| FlashVLA ICS core (paper-based) | 320.0 / 320 / 320 | — | — |
| SP-VLA pruning + scheduling | 418.9 / 158 / 512 | — | — |
| Superseded — ST-Merge spatial VLA (paper-based) | 256.0 / 256 / 256 | — | — |
| ADP / CogACT (observed EEF) | 223.7 / 192 / 256 | — | pruned 162/321 |
| FlashVLA ICS / CogACT (complete SVD) | 160.0 / 160 / 160 | — | — |
| ADP / control recheck | 458.7 / 384 / 512 | — | pruned 90/216 |
| EfficientVLA token core / corrected Q-K direction | 256.0 / 256 / 256 | — | — |
| FlashVLA ICS / automatic complete SVD | 320.0 / 320 / 320 | — | — |
| FlashVLA ICS / gesvd reference | 320.0 / 320 / 320 | — | — |
| SAFE-Pruner / control recheck | 170.7 / 128 / 512 | — | keyframe 24/216 |
| SpecPrune-VLA / control recheck | 68.5 / 60 / 108 | — | — |
| ST-Merge spatial VLA / key dot product | 256.0 / 256 / 256 | — | — |
| TEAM-VLA / audited paper-based port | 80.0 / 80 / 80 | 269.1 | — |
| VLA-IAP / audited paper-based port | 438.0 / 174 / 512 | — | warmup 24/216 |
| VLA-Pruner / control recheck | 192.0 / 128 / 512 | — | — |
| ADP | 458.7 / 384 / 512 | — | pruned 90/216 |
| SAFE-Pruner | 170.7 / 128 / 512 | — | keyframe 24/216 |
| SpecPrune-VLA (token batch 1) | 68.5 / 60 / 108 | — | — |
| TEAM-VLA (paper-based) | 80.0 / 80 / 80 | 269.1 | — |
| VLA-IAP (paper-based) | 438.0 / 174 / 512 | — | warmup 24/216 |
| VLA-Pruner | 192.0 / 128 / 512 | — | — |

## Traceability

Raw files remain under `outputs/<run ID>/`; the linked JSON preserves their
hashes, full metadata, summary, paired baseline and optional smoke records.
Model/trace/artifact identities, precision, kernel, dependencies, source
fingerprint, dirty flag and parameters are in those records. A smoke from an
earlier source revision is explicitly marked, never silently relabeled.

| Method | Preset | Timing commit | Baseline run |
|---|---|---|---|
| Native H=8 (token batch 2) | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `01bf64629fd2` | `batch2-final-benchmark-baseline-001` |
| Native H=8 end recheck (token batch 2) | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `01bf64629fd2` | `batch2-final-benchmark-baseline-001` |
| Superseded — EfficientVLA token core (paper-based) | [efficient_vla](../configs/experiments/openvla_oft/efficient_vla.yaml) | `01bf64629fd2` | `batch2-final-benchmark-baseline-001` |
| FlashVLA ICS core (paper-based) | [flash_vla](../configs/experiments/openvla_oft/flash_vla.yaml) | `01bf64629fd2` | `batch2-final-benchmark-baseline-001` |
| SP-VLA pruning + scheduling | [sp_vla](../configs/experiments/openvla_oft/sp_vla.yaml) | `01bf64629fd2` | `batch2-final-benchmark-baseline-001` |
| SP-VLA scheduling-only ablation (token batch 2) | [sp_vla_scheduling](../configs/experiments/openvla_oft/sp_vla_scheduling.yaml) | `01bf64629fd2` | `batch2-final-benchmark-baseline-001` |
| Superseded — ST-Merge spatial VLA (paper-based) | [st_merge](../configs/experiments/openvla_oft/st_merge.yaml) | `01bf64629fd2` | `batch2-final-benchmark-baseline-001` |
| VLA-Cache full statistics / eager (token batch 2) | [vla_cache](../configs/experiments/openvla_oft/vla_cache.yaml) | `01bf64629fd2` | `batch2-final-benchmark-baseline-001` |
| VLA-InfoEntropy KV reuse (paper-based) | [vla_infoentropy](../configs/experiments/openvla_oft/vla_infoentropy.yaml) | `01bf64629fd2` | `batch2-final-benchmark-baseline-001` |
| CogACT native DDIM / primary runtime | [baseline](../configs/experiments/cogact/baseline.yaml) | `a984ef33c463` | `cogact-final-benchmark-baseline-001` |
| CogACT native DDIM / secondary runtime | [baseline](../configs/experiments/cogact/baseline.yaml) | `a984ef33c463` | `cogact-secondary-benchmark-baseline-001` |
| CogACT native DDIM / secondary recheck (thermal throttling) | [baseline](../configs/experiments/cogact/baseline.yaml) | `a984ef33c463` | `cogact-secondary-benchmark-recheck-001` |
| ADP / CogACT (observed EEF) | [adp](../configs/experiments/cogact/adp.yaml) | `befad9866515` (dirty) | `cogact-token-batch1-benchmark-baseline-001` |
| CogACT native H=8 (token batch 1) | [baseline](../configs/experiments/cogact/baseline.yaml) | `befad9866515` (dirty) | `cogact-token-batch1-benchmark-baseline-001` |
| CogACT native H=8 end recheck (token batch 1) | [baseline](../configs/experiments/cogact/baseline.yaml) | `1d9d98ec3070` (dirty) | `cogact-token-batch1-benchmark-baseline-001` |
| FlashVLA ICS / CogACT (complete SVD) | [flash_vla](../configs/experiments/cogact/flash_vla.yaml) | `b03667ab6362` (dirty) | `cogact-token-batch1-benchmark-baseline-001` |
| AWQ W4A16, BF16 kernel | [awq](../configs/experiments/openvla_oft/awq.yaml) | `be5be7d8e0e1` | `control-methods-final-benchmark-baseline-001` |
| AWQ W4A16, explicit FP16 kernel | [awq_fp16_kernel](../configs/experiments/openvla_oft/awq_fp16_kernel.yaml) | `be5be7d8e0e1` | `control-methods-final-benchmark-baseline-001` |
| Native BF16 / SDPA | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `be5be7d8e0e1` | `control-methods-final-benchmark-baseline-001` |
| Native BF16 / SDPA, matched H=1 | [baseline_horizon1](../configs/experiments/openvla_oft/baseline_horizon1.yaml) | `be5be7d8e0e1` | `control-methods-final-benchmark-baseline_horizon1-001` |
| Native BF16 / SDPA, end recheck | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `be5be7d8e0e1` | `control-methods-final-benchmark-baseline-001` |
| SP-VLA scheduling core | [sp_vla_scheduling](../configs/experiments/openvla_oft/sp_vla_scheduling.yaml) | `be5be7d8e0e1` | `control-methods-final-benchmark-baseline-001` |
| SpecPrune-VLA | [specprune_vla](../configs/experiments/openvla_oft/specprune_vla.yaml) | `be5be7d8e0e1` | `control-methods-final-benchmark-baseline-001` |
| VLA-Cache, full statistics / eager | [vla_cache](../configs/experiments/openvla_oft/vla_cache.yaml) | `be5be7d8e0e1` | `control-methods-final-benchmark-baseline-001` |
| VOTE ensemble core | [vote_ensemble](../configs/experiments/openvla_oft/vote_ensemble.yaml) | `be5be7d8e0e1` | `control-methods-final-benchmark-baseline_horizon1-001` |
| ADP / control recheck | [adp](../configs/experiments/openvla_oft/adp.yaml) | `b2d91319e440` | `fidelity-final-benchmark-baseline-001` |
| Native H=8 (fidelity audit) | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `b2d91319e440` | `fidelity-final-benchmark-baseline-001` |
| Native H=8 end recheck (fidelity audit) | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `b2d91319e440` | `fidelity-final-benchmark-baseline-001` |
| EfficientVLA token core / corrected Q-K direction | [efficient_vla](../configs/experiments/openvla_oft/efficient_vla.yaml) | `b2d91319e440` | `fidelity-final-benchmark-baseline-001` |
| FlashVLA ICS / automatic complete SVD | [flash_vla](../configs/experiments/openvla_oft/flash_vla.yaml) | `b2d91319e440` | `fidelity-final-benchmark-baseline-001` |
| FlashVLA ICS / gesvd reference | [flash_vla_gesvd](../configs/experiments/openvla_oft/flash_vla_gesvd.yaml) | `b2d91319e440` | `fidelity-final-benchmark-baseline-001` |
| SAFE-Pruner / control recheck | [safe_pruner](../configs/experiments/openvla_oft/safe_pruner.yaml) | `b2d91319e440` | `fidelity-final-benchmark-baseline-001` |
| SpecPrune-VLA / control recheck | [specprune_vla](../configs/experiments/openvla_oft/specprune_vla.yaml) | `b2d91319e440` | `fidelity-final-benchmark-baseline-001` |
| ST-Merge spatial VLA / key dot product | [st_merge](../configs/experiments/openvla_oft/st_merge.yaml) | `b2d91319e440` | `fidelity-final-benchmark-baseline-001` |
| TEAM-VLA / audited paper-based port | [team_vla](../configs/experiments/openvla_oft/team_vla.yaml) | `b2d91319e440` | `fidelity-final-benchmark-baseline-001` |
| VLA-Cache full statistics / control recheck | [vla_cache](../configs/experiments/openvla_oft/vla_cache.yaml) | `b2d91319e440` | `fidelity-final-benchmark-baseline-001` |
| VLA-IAP / audited paper-based port | [vla_iap](../configs/experiments/openvla_oft/vla_iap.yaml) | `b2d91319e440` | `fidelity-final-benchmark-baseline-001` |
| VLA-InfoEntropy / audited entropy and KV reuse | [vla_infoentropy](../configs/experiments/openvla_oft/vla_infoentropy.yaml) | `b2d91319e440` | `fidelity-final-benchmark-baseline-001` |
| VLA-Pruner / control recheck | [vla_pruner](../configs/experiments/openvla_oft/vla_pruner.yaml) | `b2d91319e440` | `fidelity-final-benchmark-baseline-001` |
| ADP | [adp](../configs/experiments/openvla_oft/adp.yaml) | `ff880edb449d` | `token-batch-final-benchmark-baseline-001` |
| Native H=8 (token batch 1) | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `ff880edb449d` | `token-batch-final-benchmark-baseline-001` |
| Native H=8 end recheck (token batch 1) | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `ff880edb449d` | `token-batch-final-benchmark-baseline-001` |
| SAFE-Pruner | [safe_pruner](../configs/experiments/openvla_oft/safe_pruner.yaml) | `ff880edb449d` | `token-batch-final-benchmark-baseline-001` |
| SpecPrune-VLA (token batch 1) | [specprune_vla](../configs/experiments/openvla_oft/specprune_vla.yaml) | `ff880edb449d` | `token-batch-final-benchmark-baseline-001` |
| TEAM-VLA (paper-based) | [team_vla](../configs/experiments/openvla_oft/team_vla.yaml) | `ff880edb449d` | `token-batch-final-benchmark-baseline-001` |
| VLA-IAP (paper-based) | [vla_iap](../configs/experiments/openvla_oft/vla_iap.yaml) | `ff880edb449d` | `token-batch-final-benchmark-baseline-001` |
| VLA-Pruner | [vla_pruner](../configs/experiments/openvla_oft/vla_pruner.yaml) | `ff880edb449d` | `token-batch-final-benchmark-baseline-001` |

## Timing protocols

| Method | Hardware | Measured frames | Repeats | Warmup queries |
|---|---|---:|---:|---:|
| Native H=8 (token batch 2) | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| Native H=8 end recheck (token batch 2) | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| Superseded — EfficientVLA token core (paper-based) | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| FlashVLA ICS core (paper-based) | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| SP-VLA pruning + scheduling | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| SP-VLA scheduling-only ablation (token batch 2) | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| Superseded — ST-Merge spatial VLA (paper-based) | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| VLA-Cache full statistics / eager (token batch 2) | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| VLA-InfoEntropy KV reuse (paper-based) | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| CogACT native DDIM / primary runtime | NVIDIA GeForce RTX 3090 | 2523 | 3 | 10 |
| CogACT native DDIM / secondary runtime | NVIDIA GeForce RTX 3090 | 2523 | 3 | 10 |
| CogACT native DDIM / secondary recheck (thermal throttling) | NVIDIA GeForce RTX 3090 | 2523 | 3 | 10 |
| ADP / CogACT (observed EEF) | NVIDIA GeForce RTX 3090 | 2523 | 3 | 10 |
| CogACT native H=8 (token batch 1) | NVIDIA GeForce RTX 3090 | 2523 | 3 | 10 |
| CogACT native H=8 end recheck (token batch 1) | NVIDIA GeForce RTX 3090 | 2523 | 3 | 10 |
| FlashVLA ICS / CogACT (complete SVD) | NVIDIA GeForce RTX 3090 | 2523 | 3 | 10 |
| AWQ W4A16, BF16 kernel | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| AWQ W4A16, explicit FP16 kernel | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| Native BF16 / SDPA | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| Native BF16 / SDPA, matched H=1 | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| Native BF16 / SDPA, end recheck | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| SP-VLA scheduling core | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| SpecPrune-VLA | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| VLA-Cache, full statistics / eager | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| VOTE ensemble core | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| ADP / control recheck | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| Native H=8 (fidelity audit) | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| Native H=8 end recheck (fidelity audit) | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| EfficientVLA token core / corrected Q-K direction | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| FlashVLA ICS / automatic complete SVD | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| FlashVLA ICS / gesvd reference | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| SAFE-Pruner / control recheck | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| SpecPrune-VLA / control recheck | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| ST-Merge spatial VLA / key dot product | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| TEAM-VLA / audited paper-based port | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| VLA-Cache full statistics / control recheck | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| VLA-IAP / audited paper-based port | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| VLA-InfoEntropy / audited entropy and KV reuse | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| VLA-Pruner / control recheck | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| ADP | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| Native H=8 (token batch 1) | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| Native H=8 end recheck (token batch 1) | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| SAFE-Pruner | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| SpecPrune-VLA (token batch 1) | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| TEAM-VLA (paper-based) | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| VLA-IAP (paper-based) | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |
| VLA-Pruner | NVIDIA GeForce RTX 3090 | 1704 | 3 | 10 |

## Interpretation boundaries

- SP-VLA scheduling-only rows exclude pruning; pruning + scheduling rows
  include both mechanisms in one implementation. Fitted actions change query
  counts; compare full replay cost and evaluate success loss separately.
- VOTE means trajectory ensemble only: no ACT-token/head retraining and no
  claim to reproduce the full paper's acceleration. H=1 costs substantially
  more than native H=8 even if the voting overhead itself is small.
- AWQ's FP16-kernel row is mixed precision, not the default BF16 kernel;
  boundary casts are timed. Accuracy equivalence is not established.
- Cache regressions, failed smoke episodes and all precision variants remain
  visible. Historical runs in other validation reports are separate cohorts.
- TEAM-VLA and VLA-IAP are independent paper-based implementations with
  explicit unresolved author-code conventions; consult their method READMEs.
  Their rows are not verified reproductions of the published implementations.
- EfficientVLA token core excludes layer pruning and DiT caching; FlashVLA
  ICS core excludes unresolved action reuse. ST-Merge is a paper-based spatial
  VLA port with shared two-encoder grouping and a timed dense projector bridge.
- VLA-InfoEntropy skips visual QUERY rows while retaining visual KV context.
  It is not context pruning; its query counts do not belong in the retention
  table. Paper-based methods disclose reconstruction choices in their READMEs.
