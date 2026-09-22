# Algorithm measurement ledger

Generated from local, ignored JSON records in `outputs/result_records/`.
Only this summary is tracked now; raw runs and the live record archive stay local.
This is a development ledger, not a final paper comparison or formal success rate.
Superseded rows preserve earlier, corrected implementations for traceability;
do not use them as results of the current implementation. Their raw records
are unchanged. Valid earlier cohorts remain separate, not selected by best speed.
Mechanism scope and limits are detailed in the method READMEs.

Backend, hardware and protocol are recorded per row below. Policy timing
includes preprocessing, transfer, all method/control overhead and postprocessing,
but not model loading, trace I/O or simulation. Fixed replay is not closed-loop
task completion time. Different protocol cohorts must not be pooled.

Speedup = paired native baseline total replay time / method total replay time.
H is the executed prefix, distinct from the backend's full prediction length.
VOTE presets use a matched H=1 baseline on each backend; the other current presets
use native H=8. Do not compare their speedup
ratios as if query cadence were identical. Peak GiB is runtime allocated tensor
memory, not loading peak or total process memory. Smoke is successes/episodes,
not a success-rate estimate. Formal success rate is **not measured for any row**.

## 0. Native references

| Record / configuration | Backend | H | Queries / replacements | Query mean / p95 ms | ms/control step | Replay speedup | Peak GiB | Smoke |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [R01](#r01) Native H=8 (token batch 2) | openvla_oft | 8 | 216 / 0 | 197.31 / 199.10 | 25.03 | 1.000× | 14.98 | Not run |
| [R02](#r02) Native H=8 end recheck (token batch 2) | openvla_oft | 8 | 216 / 0 | 198.63 / 200.64 | 25.20 | 0.993× | 14.98 | Not run |
| [R12](#r12) CogACT native H8 / category examples | cogact | 8 | 321 / 0 | 185.37 / 186.99 | 23.60 | 1.000× | 14.73 | 3/4; errors 0 |
| [R13](#r13) CogACT native H1 / matched VOTE reference | cogact | 1 | 2523 / 0 | 185.82 / 186.85 | 185.82 | 1.000× | 14.73 | 4/4; errors 0 |
| [R14](#r14) CogACT native H8 / end-of-cohort recheck | cogact | 8 | 321 / 0 | 184.56 / 185.72 | 23.50 | 1.004× | 14.73 | Not run |
| [R19](#r19) CogACT native DDIM / primary runtime | cogact | 8 | 321 / 0 | 183.69 / 184.81 | 23.39 | 1.000× | 14.73 | 3/4; errors 0 |
| [R20](#r20) CogACT native DDIM / secondary runtime | cogact | 8 | 321 / 0 | 194.39 / 204.01 | 24.75 | 1.000× | 14.73 | 3/4; errors 0 |
| [R21](#r21) CogACT native DDIM / secondary recheck (thermal throttling) | cogact | 8 | 321 / 0 | 195.57 / 209.94 | 24.90 | 1.000× | 14.73 | 3/4; errors 0 |
| [R23](#r23) CogACT native H=8 (token batch 1) | cogact | 8 | 321 / 0 | 183.86 / 184.84 | 23.41 | 1.000× | 14.73 | 3/4; errors 0 |
| [R24](#r24) CogACT native H=8 end recheck (token batch 1) | cogact | 8 | 321 / 0 | 184.36 / 185.67 | 23.47 | 0.997× | 14.73 | Not run |
| [R28](#r28) Native BF16 / SDPA | openvla_oft | 8 | 216 / 0 | 197.86 / 199.86 | 25.10 | 1.000× | 14.98 | 4/4; errors 0 |
| [R29](#r29) Native BF16 / SDPA, matched H=1 | openvla_oft | 1 | 1704 / 0 | 199.83 / 201.42 | 199.83 | 1.000× | 14.98 | Not run |
| [R30](#r30) Native BF16 / SDPA, end recheck | openvla_oft | 8 | 216 / 0 | 198.91 / 201.15 | 25.23 | 0.995× | 14.98 | Not run |
| [R36](#r36) Native H=8 (fidelity audit) | openvla_oft | 8 | 216 / 0 | 196.74 / 198.44 | 24.96 | 1.000× | 14.98 | Not run |
| [R37](#r37) Native H=8 end recheck (fidelity audit) | openvla_oft | 8 | 216 / 0 | 197.84 / 199.24 | 25.10 | 0.994× | 14.98 | Not run |
| [R50](#r50) Native H=8 (token batch 1) | openvla_oft | 8 | 216 / 0 | 200.57 / 203.81 | 25.44 | 1.000× | 14.98 | Not run |
| [R51](#r51) Native H=8 end recheck (token batch 1) | openvla_oft | 8 | 216 / 0 | 198.77 / 200.57 | 25.21 | 1.009× | 14.98 | Not run |

## 5.1 Token Pruning and Context Compression

| Record / configuration | Backend | H | Queries / replacements | Query mean / p95 ms | ms/control step | Replay speedup | Peak GiB | Smoke |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [R03](#r03) Superseded — EfficientVLA token core (paper-based) | openvla_oft | 8 | 216 / 0 | 170.08 / 171.33 | 21.58 | 1.160× | 14.62 | 4/4; errors 0 |
| [R04](#r04) FlashVLA ICS core (paper-based) | openvla_oft | 8 | 216 / 0 | 238.64 / 241.15 | 30.27 | 0.827× | 14.64 | 4/4; errors 0 |
| [R07](#r07) Superseded — ST-Merge spatial VLA (paper-based) | openvla_oft | 8 | 216 / 0 | 171.42 / 172.94 | 21.75 | 1.151× | 14.62 | 4/4; errors 0 |
| [R22](#r22) ADP / CogACT (observed EEF) | cogact | 8 | 321 / 0 | 168.27 / 184.16 | 21.44 | 1.092× | 14.73 | 4/4; errors 0 |
| [R25](#r25) FlashVLA ICS / CogACT (complete SVD) | cogact | 8 | 321 / 0 | 157.56 / 166.44 | 20.07 | 1.167× | 14.66 | 4/4; errors 0 |
| [R32](#r32) SpecPrune-VLA | openvla_oft | 8 | 237 / 0 | 146.91 / 163.52 | 20.51 | 1.224× | 14.60 | 3/4; errors 0 |
| [R35](#r35) ADP / control recheck | openvla_oft | 8 | 216 / 0 | 193.63 / 198.37 | 24.58 | 1.015× | 14.68 | 4/4; errors 0 |
| [R38](#r38) EfficientVLA token core / corrected Q-K direction | openvla_oft | 8 | 216 / 0 | 170.19 / 172.02 | 21.59 | 1.156× | 14.63 | 4/4; errors 0 |
| [R39](#r39) FlashVLA ICS / automatic complete SVD | openvla_oft | 8 | 216 / 0 | 208.73 / 210.37 | 26.48 | 0.943× | 14.64 | 4/4; errors 0 |
| [R40](#r40) FlashVLA ICS / gesvd reference | openvla_oft | 8 | 216 / 0 | 237.86 / 240.28 | 30.17 | 0.827× | 14.64 | Not run |
| [R41](#r41) SAFE-Pruner / control recheck | openvla_oft | 8 | 216 / 0 | 146.76 / 198.59 | 18.62 | 1.340× | 14.68 | 4/4; errors 0 |
| [R42](#r42) SpecPrune-VLA / control recheck | openvla_oft | 8 | 237 / 0 | 146.13 / 162.17 | 20.42 | 1.222× | 14.60 | 3/4; errors 0 |
| [R43](#r43) ST-Merge spatial VLA / key dot product | openvla_oft | 8 | 216 / 0 | 171.36 / 172.73 | 21.74 | 1.148× | 14.62 | 3/4; errors 0 |
| [R44](#r44) TEAM-VLA / audited paper-based port | openvla_oft | 8 | 216 / 0 | 139.58 / 148.03 | 17.71 | 1.409× | 14.61 | 1/4; errors 0 |
| [R46](#r46) VLA-IAP / audited paper-based port | openvla_oft | 8 | 216 / 0 | 193.23 / 200.12 | 24.51 | 1.018× | 14.69 | 3/4; errors 0 |
| [R48](#r48) VLA-Pruner / control recheck | openvla_oft | 8 | 216 / 0 | 163.52 / 198.05 | 20.75 | 1.203× | 14.68 | 4/4; errors 0 |
| [R49](#r49) ADP | openvla_oft | 8 | 216 / 0 | 195.37 / 200.98 | 24.80 | 1.026× | 14.68 | 4/4; errors 0 |
| [R52](#r52) SAFE-Pruner | openvla_oft | 8 | 216 / 0 | 148.28 / 200.16 | 18.82 | 1.352× | 14.68 | 4/4; errors 0 |
| [R53](#r53) SpecPrune-VLA (token batch 1) | openvla_oft | 8 | 237 / 0 | 147.06 / 163.44 | 20.53 | 1.239× | 14.60 | 3/4; errors 0 |
| [R54](#r54) TEAM-VLA (paper-based) | openvla_oft | 8 | 216 / 0 | 139.47 / 148.07 | 17.70 | 1.438× | 14.61 | 1/4; errors 0 |
| [R55](#r55) VLA-IAP (paper-based) | openvla_oft | 8 | 216 / 0 | 196.31 / 204.09 | 24.90 | 1.022× | 14.69 | 3/4; errors 0 |
| [R56](#r56) VLA-Pruner | openvla_oft | 8 | 216 / 0 | 165.05 / 199.81 | 20.94 | 1.215× | 14.68 | 4/4; errors 0 |

## 5.2 Caching and Computation Reuse

| Record / configuration | Backend | H | Queries / replacements | Query mean / p95 ms | ms/control step | Replay speedup | Peak GiB | Smoke |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [R08](#r08) VLA-Cache full statistics / eager (token batch 2) | openvla_oft | 8 | 216 / 0 | 207.43 / 217.93 | 26.31 | 0.951× | 15.03 | Not run |
| [R09](#r09) VLA-InfoEntropy KV reuse (paper-based) | openvla_oft | 8 | 216 / 0 | 213.14 / 222.39 | 27.04 | 0.926× | 15.03 | 4/4; errors 0 |
| [R16](#r16) VLA-Cache full statistics eager / CogACT | cogact | 8 | 321 / 0 | 158.58 / 167.35 | 20.20 | 1.169× | 14.75 | 3/4; errors 0 |
| [R17](#r17) VLA-Cache full statistics SDPA / CogACT | cogact | 8 | 321 / 0 | 159.88 / 168.24 | 20.36 | 1.159× | 14.75 | Not run |
| [R33](#r33) VLA-Cache, full statistics / eager | openvla_oft | 8 | 216 / 0 | 207.64 / 219.38 | 26.34 | 0.953× | 15.03 | 4/4; errors 0 |
| [R45](#r45) VLA-Cache full statistics / control recheck | openvla_oft | 8 | 216 / 0 | 206.24 / 216.85 | 26.16 | 0.954× | 15.03 | 4/4; errors 0 |
| [R47](#r47) VLA-InfoEntropy / audited entropy and KV reuse | openvla_oft | 8 | 216 / 0 | 213.47 / 222.80 | 27.08 | 0.922× | 15.03 | 4/4; errors 0 |

## 5.3 Layer Skipping, Early Exit, and Model Routing

| Record / configuration | Backend | H | Queries / replacements | Query mean / p95 ms | ms/control step | Replay speedup | Peak GiB | Smoke |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [R05](#r05) SP-VLA pruning + scheduling | openvla_oft | 8 | 201 / 153 | 189.63 / 201.81 | 22.41 | 1.117× | 14.68 | 4/4; errors 0 |
| [R06](#r06) SP-VLA scheduling-only ablation (token batch 2) | openvla_oft | 8 | 201 / 153 | 197.93 / 200.72 | 23.39 | 1.070× | 14.98 | Not run |
| [R15](#r15) SP-VLA pruning + scheduling / CogACT | cogact | 8 | 306 / 141 | 171.98 / 189.20 | 20.89 | 1.130× | 14.73 | 4/4; errors 0 |
| [R31](#r31) SP-VLA scheduling core | openvla_oft | 8 | 201 / 153 | 197.51 / 199.26 | 23.34 | 1.076× | 14.98 | 4/4; errors 0 |

## 5.5 Action Chunking and Replanning

| Record / configuration | Backend | H | Queries / replacements | Query mean / p95 ms | ms/control step | Replay speedup | Peak GiB | Smoke |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [R18](#r18) VOTE ensemble core / CogACT | cogact | 1 | 2523 / 0 | 186.53 / 187.57 | 186.53 | 0.996× | 14.73 | 3/4; errors 0 |
| [R34](#r34) VOTE ensemble core | openvla_oft | 1 | 1704 / 0 | 199.62 / 201.47 | 199.62 | 1.001× | 14.98 | 3/4; errors 0 |

## 6.1 Quantization and Low-Precision Inference

| Record / configuration | Backend | H | Queries / replacements | Query mean / p95 ms | ms/control step | Replay speedup | Peak GiB | Smoke |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [R10](#r10) AWQ W4A16 BF16 kernel / CogACT | cogact | 8 | 321 / 0 | 155.51 / 156.42 | 19.81 | 1.192× | 5.90 | 3/4; errors 0 |
| [R11](#r11) AWQ W4A16 explicit FP16 kernel / CogACT | cogact | 8 | 321 / 0 | 135.92 / 136.99 | 17.31 | 1.363× | 5.90 | 4/4; errors 0 |
| [R26](#r26) AWQ W4A16, BF16 kernel | openvla_oft | 8 | 216 / 0 | 204.95 / 206.69 | 26.00 | 0.965× | 6.12 | 4/4; errors 0 |
| [R27](#r27) AWQ W4A16, explicit FP16 kernel | openvla_oft | 8 | 216 / 0 | 170.32 / 171.73 | 21.61 | 1.162× | 6.12 | 4/4; errors 0 |

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
| SP-VLA pruning + scheduling / CogACT | 219.0 / 108 / 256 | — | — |
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

## Computed visual queries and retained KV context

Query-weighted last-layer counts; skipping query/MLP rows does not remove KV context.

| Method | Computed queries: mean / min / max | Retained KV: mean / min / max |
|---|---:|---:|
| VLA-Cache full statistics eager / CogACT | 162.4 / 145 / 256 | 256.0 / 256 / 256 |
| VLA-Cache full statistics SDPA / CogACT | 162.4 / 145 / 256 | 256.0 / 256 / 256 |

## Measurement cohorts

Each row uses its own paired baseline, never another cohort's denominator.
F = measured control frames; R = repeats; W = warmup queries.

| Cohort | Backend | Hardware | F / R / W | Source fingerprint | Checkpoint |
|---|---|---|---:|---|---|
| C01 | openvla_oft | NVIDIA GeForce RTX 3090 | 1704 / 3 / 10 | `8d179bafa867453407239838c7e82460837d3032bcedf549cc181385c3fd858c` | `b127ee4bbf88c6e61364da6af2df22efe10363140092c52d54d34f8dbd918fe1` |
| C02 | cogact | NVIDIA GeForce RTX 3090 | 2523 / 3 / 10 | `b4cbf022759a0ec4de476f24ee77efa21f1df093d410d467bbc98e763131734f` | `c8aa6908079a1e83fde009bfe5d50bc755aab2e83f160ea4e372f4a8ddc7fc9c` |
| C03 | cogact | NVIDIA GeForce RTX 3090 | 2523 / 3 / 10 | `d9505c5563884804dca293748aef8871017ace1e534cc9378cb9d59e1ec3e8ee` | `c8aa6908079a1e83fde009bfe5d50bc755aab2e83f160ea4e372f4a8ddc7fc9c` |
| C04 | cogact | NVIDIA GeForce RTX 3090 | 2523 / 3 / 10 | `dc290916d485bb0f7d3607956ee7075aa0f500d8fad11db0501315d03cd618c8` | `c8aa6908079a1e83fde009bfe5d50bc755aab2e83f160ea4e372f4a8ddc7fc9c` |
| C05 | cogact | NVIDIA GeForce RTX 3090 | 2523 / 3 / 10 | `dc290916d485bb0f7d3607956ee7075aa0f500d8fad11db0501315d03cd618c8` | `c8aa6908079a1e83fde009bfe5d50bc755aab2e83f160ea4e372f4a8ddc7fc9c` |
| C06 | cogact | NVIDIA GeForce RTX 3090 | 2523 / 3 / 10 | `dc290916d485bb0f7d3607956ee7075aa0f500d8fad11db0501315d03cd618c8` | `c8aa6908079a1e83fde009bfe5d50bc755aab2e83f160ea4e372f4a8ddc7fc9c` |
| C07 | openvla_oft | NVIDIA GeForce RTX 3090 | 1704 / 3 / 10 | `86810c762e0810442533cb422acdea45a9557d40415d30d49b88f7135c23c3e7` | `b127ee4bbf88c6e61364da6af2df22efe10363140092c52d54d34f8dbd918fe1` |
| C08 | openvla_oft | NVIDIA GeForce RTX 3090 | 1704 / 3 / 10 | `1ca39ee30a256e6b873adad4b37f284cb2e2cc93cb0ed1b36ff7136f60677cf8` | `b127ee4bbf88c6e61364da6af2df22efe10363140092c52d54d34f8dbd918fe1` |
| C09 | openvla_oft | NVIDIA GeForce RTX 3090 | 1704 / 3 / 10 | `f1ac75f63d548db2c37f959f18d83bc5667500cba54122c1979468e88f307ab0` | `b127ee4bbf88c6e61364da6af2df22efe10363140092c52d54d34f8dbd918fe1` |

<details>
<summary>Run identities, presets and local-record checksums</summary>

Raw runs: `outputs/<run ID>/`; exported records: `outputs/result_records/<run ID>.json`.
The SHA256 identifies the full canonical JSON, which retains raw-file hashes,
parameters, baseline, runtime/precision and per-episode smoke outcomes.
The preset links are current; use the recorded commit for measured source.
Smoke-source labels: same / different / unknown / not run. Earlier smoke is
never promoted to a final-source or formal evaluation.

| Record | Run ID | Cohort | Preset | Timing commit | Baseline | Smoke source | Record SHA256 |
|---|---|---|---|---|---|---|---|
| <a id="r01"></a>R01 | `batch2-final-benchmark-baseline-001` | C01 | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `01bf64629fd2db27edef978ffa76d9ec6ed0ef9d` | R01 | not run | `1a4a0a8b39ef12b830893c1531bfe016ebbcd342fd7fed50ef16be87be4eab67` |
| <a id="r02"></a>R02 | `batch2-final-benchmark-baseline_recheck-001` | C01 | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `01bf64629fd2db27edef978ffa76d9ec6ed0ef9d` | R01 | not run | `cb9f68ed3f840ea9843f56c02c027d61b83d404459b0039d016c57515f7a40ee` |
| <a id="r03"></a>R03 | `batch2-final-benchmark-efficient_vla-001` | C01 | [efficient_vla](../configs/experiments/openvla_oft/efficient_vla.yaml) | `01bf64629fd2db27edef978ffa76d9ec6ed0ef9d` | R01 | different | `903a047607388c797f10dd279ec97cb61cf2c2aa7fad2ecd74df5027c1a436fb` |
| <a id="r04"></a>R04 | `batch2-final-benchmark-flash_vla-001` | C01 | [flash_vla](../configs/experiments/openvla_oft/flash_vla.yaml) | `01bf64629fd2db27edef978ffa76d9ec6ed0ef9d` | R01 | different | `fe5edc1c157bb9e3274011309dcb858323dee6db4f4d423e8a00ca7ce1557934` |
| <a id="r05"></a>R05 | `batch2-final-benchmark-sp_vla-001` | C01 | [sp_vla](../configs/experiments/openvla_oft/sp_vla.yaml) | `01bf64629fd2db27edef978ffa76d9ec6ed0ef9d` | R01 | different | `a1495e644469ba669444d321923077cc4f38ca62a9561abf9c706a1b773deed6` |
| <a id="r06"></a>R06 | `batch2-final-benchmark-sp_vla_scheduling-001` | C01 | [sp_vla_scheduling](../configs/experiments/openvla_oft/sp_vla_scheduling.yaml) | `01bf64629fd2db27edef978ffa76d9ec6ed0ef9d` | R01 | not run | `b86562025ba0d8f6cdcdbf65edecece0dba652dbf810215228cde8abe7acc7e0` |
| <a id="r07"></a>R07 | `batch2-final-benchmark-st_merge-001` | C01 | [st_merge](../configs/experiments/openvla_oft/st_merge.yaml) | `01bf64629fd2db27edef978ffa76d9ec6ed0ef9d` | R01 | different | `7b11ea6a370d84f3b48e720f50be4dd8f997ce54bd498f00025ca4e82108bd87` |
| <a id="r08"></a>R08 | `batch2-final-benchmark-vla_cache-001` | C01 | [vla_cache](../configs/experiments/openvla_oft/vla_cache.yaml) | `01bf64629fd2db27edef978ffa76d9ec6ed0ef9d` | R01 | not run | `9a3d77d7648bf8abad73efc28a1a9472b4baf9dc399d2245cf3dc6481b1c6e2b` |
| <a id="r09"></a>R09 | `batch2-final-benchmark-vla_infoentropy-001` | C01 | [vla_infoentropy](../configs/experiments/openvla_oft/vla_infoentropy.yaml) | `01bf64629fd2db27edef978ffa76d9ec6ed0ef9d` | R01 | different | `9059b709cd8c5ea40bbee8ee3467cfd19e10dae004058a225ddd1496e55343e8` |
| <a id="r10"></a>R10 | `cogact-category-final-awq-001` | C02 | [awq](../configs/experiments/cogact/awq.yaml) | `ba55950deca55b2f56c946948d315fd2d84031ba` | R12 | different | `9634222837c9f4a5b56e735822cc2310f29d1d6bfdfb64ca39c891c6e00f5d65` |
| <a id="r11"></a>R11 | `cogact-category-final-awq_fp16_kernel-001` | C02 | [awq_fp16_kernel](../configs/experiments/cogact/awq_fp16_kernel.yaml) | `ba55950deca55b2f56c946948d315fd2d84031ba` | R12 | different | `09ef55635c0eaf4754fd2b5dfd4164bb3ac45b14cb9218a4e93ba8e70eae22f8` |
| <a id="r12"></a>R12 | `cogact-category-final-baseline-001` | C02 | [baseline](../configs/experiments/cogact/baseline.yaml) | `ba55950deca55b2f56c946948d315fd2d84031ba` | R12 | different | `f472eb7876f494ce3c0c597d848b6677db7c9d0266c8b5a45e80a74fe36fb268` |
| <a id="r13"></a>R13 | `cogact-category-final-baseline_horizon1-001` | C02 | [baseline_horizon1](../configs/experiments/cogact/baseline_horizon1.yaml) | `ba55950deca55b2f56c946948d315fd2d84031ba` | R13 | different | `4b2f45f7ac3d8b0ff318d0409c4335f1454fb971bf63220a528f747e45bfbe92` |
| <a id="r14"></a>R14 | `cogact-category-final-baseline_recheck-001` | C02 | [baseline](../configs/experiments/cogact/baseline.yaml) | `ba55950deca55b2f56c946948d315fd2d84031ba` | R12 | not run | `5fdb0c317370235bb24211600bc5372414828d0746500fce0184d5aaf9cfccd9` |
| <a id="r15"></a>R15 | `cogact-category-final-sp_vla-001` | C02 | [sp_vla](../configs/experiments/cogact/sp_vla.yaml) | `ba55950deca55b2f56c946948d315fd2d84031ba` | R12 | different | `e7b15f41e8ce983a77416c0e5e4b94e671b6a03bb18390e77c6a77c6feeedf10` |
| <a id="r16"></a>R16 | `cogact-category-final-vla_cache-001` | C02 | [vla_cache](../configs/experiments/cogact/vla_cache.yaml) | `ba55950deca55b2f56c946948d315fd2d84031ba` | R12 | different | `cdb080cf51322720c5cd5848e2fd53f9a95990d3b787ce0829215ccb736868a6` |
| <a id="r17"></a>R17 | `cogact-category-final-vla_cache_sdpa-001` | C02 | [vla_cache_sdpa](../configs/experiments/cogact/vla_cache_sdpa.yaml) | `ba55950deca55b2f56c946948d315fd2d84031ba` | R12 | not run | `50ba76327548d5e6d8eaeb783ba260fbd16f3f36a2ff91316cc30b311e74800e` |
| <a id="r18"></a>R18 | `cogact-category-final-vote_ensemble-001` | C02 | [vote_ensemble](../configs/experiments/cogact/vote_ensemble.yaml) | `ba55950deca55b2f56c946948d315fd2d84031ba` | R13 | different | `36344f4cd135853b4a2fa7a42afe1298f3ff7deb3831ecdbfe2cd596ab94d384` |
| <a id="r19"></a>R19 | `cogact-final-benchmark-baseline-001` | C03 | [baseline](../configs/experiments/cogact/baseline.yaml) | `a984ef33c463dd2af38be19cb4dee4f8e73a4f56` | R19 | same | `d95cfef815a1d179a566d5bd06f1d67043ab5d39af8267e56137f28bc57f8213` |
| <a id="r20"></a>R20 | `cogact-secondary-benchmark-baseline-001` | C03 | [baseline](../configs/experiments/cogact/baseline.yaml) | `a984ef33c463dd2af38be19cb4dee4f8e73a4f56` | R20 | same | `19e73138b950fe69654c83d6169a8525d7c86247b505976fcc90c8fdc2c70095` |
| <a id="r21"></a>R21 | `cogact-secondary-benchmark-recheck-001` | C03 | [baseline](../configs/experiments/cogact/baseline.yaml) | `a984ef33c463dd2af38be19cb4dee4f8e73a4f56` | R21 | same | `8f4b6c4d743d29ae60261d15c7524cb3c48e254e8880e7a357521661d8fe31f8` |
| <a id="r22"></a>R22 | `cogact-token-batch1-benchmark-adp-001` | C04 | [adp](../configs/experiments/cogact/adp.yaml) | `befad9866515af21713435bdc69f5ddde8b0592d` (dirty) | R23 | same | `0d193f6a157cb83fb6d652d88367bf8a7d9a7dce2b96993ba134f324b67071df` |
| <a id="r23"></a>R23 | `cogact-token-batch1-benchmark-baseline-001` | C04 | [baseline](../configs/experiments/cogact/baseline.yaml) | `befad9866515af21713435bdc69f5ddde8b0592d` (dirty) | R23 | same | `406891d6c9c3be93424bdb1bff40be6faad3737617e926510fc27ed18f9acc4a` |
| <a id="r24"></a>R24 | `cogact-token-batch1-benchmark-baseline_recheck-001` | C05 | [baseline](../configs/experiments/cogact/baseline.yaml) | `1d9d98ec3070e1d216738b2978bb7ab825ce015b` (dirty) | R23 | not run | `92ca052fe9e981caaebfca1124d0ccab13a2b4722a429d89e3bc7afd8ef7c896` |
| <a id="r25"></a>R25 | `cogact-token-batch1-benchmark-flash_vla-001` | C06 | [flash_vla](../configs/experiments/cogact/flash_vla.yaml) | `b03667ab6362a42ac5215678b8d8829197dcc7a6` (dirty) | R23 | same | `9173edd82490a071f88397e3200389e8b3435650f06f8209ee77e5e606a00b48` |
| <a id="r26"></a>R26 | `control-methods-final-benchmark-awq-001` | C07 | [awq](../configs/experiments/openvla_oft/awq.yaml) | `be5be7d8e0e170e1f5956632ce537a3fa93cb653` | R28 | different | `7c45d89c736771315f940f7f90e5167f96df563b15676415340a70bab1a9b82d` |
| <a id="r27"></a>R27 | `control-methods-final-benchmark-awq_fp16_kernel-001` | C07 | [awq_fp16_kernel](../configs/experiments/openvla_oft/awq_fp16_kernel.yaml) | `be5be7d8e0e170e1f5956632ce537a3fa93cb653` | R28 | different | `82803a005adb4c6288906f1d683b6483fa8e446fb842a352400c6ca213fadb7e` |
| <a id="r28"></a>R28 | `control-methods-final-benchmark-baseline-001` | C07 | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `be5be7d8e0e170e1f5956632ce537a3fa93cb653` | R28 | different | `5cb6a9ec850c9f870dd24b7d9d8f46651871d7b403ee604cc9dc2a69a0e40b68` |
| <a id="r29"></a>R29 | `control-methods-final-benchmark-baseline_horizon1-001` | C07 | [baseline_horizon1](../configs/experiments/openvla_oft/baseline_horizon1.yaml) | `be5be7d8e0e170e1f5956632ce537a3fa93cb653` | R29 | not run | `f060e73d87b99b1ba3513e629711e774dad111286f407714c057af702f89a9d1` |
| <a id="r30"></a>R30 | `control-methods-final-benchmark-baseline_recheck-001` | C07 | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `be5be7d8e0e170e1f5956632ce537a3fa93cb653` | R28 | not run | `59d8e88fa26a72997a2cc1f68e1ad47a2f09c7c457be8d38eab276d1c40abbcd` |
| <a id="r31"></a>R31 | `control-methods-final-benchmark-sp_vla_scheduling-001` | C07 | [sp_vla_scheduling](../configs/experiments/openvla_oft/sp_vla_scheduling.yaml) | `be5be7d8e0e170e1f5956632ce537a3fa93cb653` | R28 | different | `a0273879758dae45be9f1842c1075c84ed1f2b62af0e85b77b26688c0cc813d1` |
| <a id="r32"></a>R32 | `control-methods-final-benchmark-specprune_vla-001` | C07 | [specprune_vla](../configs/experiments/openvla_oft/specprune_vla.yaml) | `be5be7d8e0e170e1f5956632ce537a3fa93cb653` | R28 | different | `d0cf29382c0b1da4527e97b3774cfeed605d59dbde35233d3250eccbc9b81fba` |
| <a id="r33"></a>R33 | `control-methods-final-benchmark-vla_cache-001` | C07 | [vla_cache](../configs/experiments/openvla_oft/vla_cache.yaml) | `be5be7d8e0e170e1f5956632ce537a3fa93cb653` | R28 | different | `8b7c57663e402e87953bc434a0f0fc3d21ac0234d8909bddd825e499ee9a24b1` |
| <a id="r34"></a>R34 | `control-methods-final-benchmark-vote_ensemble-001` | C07 | [vote_ensemble](../configs/experiments/openvla_oft/vote_ensemble.yaml) | `be5be7d8e0e170e1f5956632ce537a3fa93cb653` | R29 | different | `3784f976868209483ed74c9ec4bd1f4324dff8b2dbf44c0bbe4d13104c763f9b` |
| <a id="r35"></a>R35 | `fidelity-final-benchmark-adp-001` | C08 | [adp](../configs/experiments/openvla_oft/adp.yaml) | `b2d91319e440d6c3aeaf1ddc2c04a2c1f10c5362` | R36 | different | `fc1d8ece0b442050e715565e1f932e9d4ae170b43b2b010cf72c981db5d6b313` |
| <a id="r36"></a>R36 | `fidelity-final-benchmark-baseline-001` | C08 | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `b2d91319e440d6c3aeaf1ddc2c04a2c1f10c5362` | R36 | not run | `9e8d1cad9518377fd4b21b84b03ccbb70ebd8ecd7e0addea5ae2c700a5f402a3` |
| <a id="r37"></a>R37 | `fidelity-final-benchmark-baseline_recheck-001` | C08 | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `b2d91319e440d6c3aeaf1ddc2c04a2c1f10c5362` | R36 | not run | `0e78732459c98afe86a499171b2e162874b15b54533e8d260df8be4450c71f23` |
| <a id="r38"></a>R38 | `fidelity-final-benchmark-efficient_vla-001` | C08 | [efficient_vla](../configs/experiments/openvla_oft/efficient_vla.yaml) | `b2d91319e440d6c3aeaf1ddc2c04a2c1f10c5362` | R36 | different | `49639a395fd6bec2352dbb1608df5fb6accc9f9d003f86f878ea3f68544d33c4` |
| <a id="r39"></a>R39 | `fidelity-final-benchmark-flash_vla-001` | C08 | [flash_vla](../configs/experiments/openvla_oft/flash_vla.yaml) | `b2d91319e440d6c3aeaf1ddc2c04a2c1f10c5362` | R36 | different | `321ef6b360eb7fce75f5da6d783d9fea4b2081c808db185d0df4f2d115a92776` |
| <a id="r40"></a>R40 | `fidelity-final-benchmark-flash_vla_gesvd-001` | C08 | [flash_vla_gesvd](../configs/experiments/openvla_oft/flash_vla_gesvd.yaml) | `b2d91319e440d6c3aeaf1ddc2c04a2c1f10c5362` | R36 | not run | `c6817118ab0cc9e2d5f0fe7fb016f44ca9c8120026c506e816f73703622b9a15` |
| <a id="r41"></a>R41 | `fidelity-final-benchmark-safe_pruner-001` | C08 | [safe_pruner](../configs/experiments/openvla_oft/safe_pruner.yaml) | `b2d91319e440d6c3aeaf1ddc2c04a2c1f10c5362` | R36 | different | `ee1d25419d4ed8ef486fc26b10adf0b7bdf11530ea65b38d189311d17f7baf44` |
| <a id="r42"></a>R42 | `fidelity-final-benchmark-specprune_vla-001` | C08 | [specprune_vla](../configs/experiments/openvla_oft/specprune_vla.yaml) | `b2d91319e440d6c3aeaf1ddc2c04a2c1f10c5362` | R36 | different | `7b6b8ba2d827fa403a1bcbee9108cb9a8d14a4e50e6ff2ae80499a0d247bfc05` |
| <a id="r43"></a>R43 | `fidelity-final-benchmark-st_merge-001` | C08 | [st_merge](../configs/experiments/openvla_oft/st_merge.yaml) | `b2d91319e440d6c3aeaf1ddc2c04a2c1f10c5362` | R36 | different | `da7829e5a8c2121e36035a1b9e29067c62507861d180a2a3eae635c580ac0009` |
| <a id="r44"></a>R44 | `fidelity-final-benchmark-team_vla-001` | C08 | [team_vla](../configs/experiments/openvla_oft/team_vla.yaml) | `b2d91319e440d6c3aeaf1ddc2c04a2c1f10c5362` | R36 | different | `f4dbc822013399631c0353d0a007604afe9101bacbf2f2560968e220d8014882` |
| <a id="r45"></a>R45 | `fidelity-final-benchmark-vla_cache-001` | C08 | [vla_cache](../configs/experiments/openvla_oft/vla_cache.yaml) | `b2d91319e440d6c3aeaf1ddc2c04a2c1f10c5362` | R36 | different | `7e24802f0945e179c6b8fc5965c6d5ed7e961c50e1399961a31fa5066a49c943` |
| <a id="r46"></a>R46 | `fidelity-final-benchmark-vla_iap-001` | C08 | [vla_iap](../configs/experiments/openvla_oft/vla_iap.yaml) | `b2d91319e440d6c3aeaf1ddc2c04a2c1f10c5362` | R36 | different | `666948372c4174d51e555e819f4e1ef23a658088e44d4b7ef2f3522535a14e40` |
| <a id="r47"></a>R47 | `fidelity-final-benchmark-vla_infoentropy-001` | C08 | [vla_infoentropy](../configs/experiments/openvla_oft/vla_infoentropy.yaml) | `b2d91319e440d6c3aeaf1ddc2c04a2c1f10c5362` | R36 | different | `8d7db1528cb12540d7f9bbbe30851bdf49699a4815a1621c938dbfa43fac1536` |
| <a id="r48"></a>R48 | `fidelity-final-benchmark-vla_pruner-001` | C08 | [vla_pruner](../configs/experiments/openvla_oft/vla_pruner.yaml) | `b2d91319e440d6c3aeaf1ddc2c04a2c1f10c5362` | R36 | different | `b219ae2d9a9917df9e242566c12d711efa0e15ff8027e57ec9e5547956d37b08` |
| <a id="r49"></a>R49 | `token-batch-final-benchmark-adp-001` | C09 | [adp](../configs/experiments/openvla_oft/adp.yaml) | `ff880edb449d3840d2a7420e1363d155ac9dfe95` | R50 | different | `3692e25e645d7b0a7a511f97cbcc5e618124737c2cbc1841d88a8b5560cd166a` |
| <a id="r50"></a>R50 | `token-batch-final-benchmark-baseline-001` | C09 | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `ff880edb449d3840d2a7420e1363d155ac9dfe95` | R50 | not run | `64a8c6d3d1239176e992d8baab97bb17e54512746d3ecaf22a069f1cb4c37b40` |
| <a id="r51"></a>R51 | `token-batch-final-benchmark-baseline_recheck-001` | C09 | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `ff880edb449d3840d2a7420e1363d155ac9dfe95` | R50 | not run | `94e5197bd45df492e23c8692ca77d74cc67b3d53795bc905a554c081c84264f9` |
| <a id="r52"></a>R52 | `token-batch-final-benchmark-safe_pruner-001` | C09 | [safe_pruner](../configs/experiments/openvla_oft/safe_pruner.yaml) | `ff880edb449d3840d2a7420e1363d155ac9dfe95` | R50 | different | `6057ae01a3230a08bd6bc39d984771bc016c678321296eab9061ad1e5acc2cd7` |
| <a id="r53"></a>R53 | `token-batch-final-benchmark-specprune_vla-001` | C09 | [specprune_vla](../configs/experiments/openvla_oft/specprune_vla.yaml) | `ff880edb449d3840d2a7420e1363d155ac9dfe95` | R50 | different | `c98ebecd8039fcd1ffd691677410e4bb02708f0cf5e0473962eab52c80bcbaf2` |
| <a id="r54"></a>R54 | `token-batch-final-benchmark-team_vla-001` | C09 | [team_vla](../configs/experiments/openvla_oft/team_vla.yaml) | `ff880edb449d3840d2a7420e1363d155ac9dfe95` | R50 | different | `abfd5e1af5ccbbedce776eddc3825ca0bdeec7bbea54338cc4bf1597f36dc7b5` |
| <a id="r55"></a>R55 | `token-batch-final-benchmark-vla_iap-001` | C09 | [vla_iap](../configs/experiments/openvla_oft/vla_iap.yaml) | `ff880edb449d3840d2a7420e1363d155ac9dfe95` | R50 | different | `293184fa17e39c25574f6d91ec5a7b4118623d07239ca2f1703fe7a43fd1b0df` |
| <a id="r56"></a>R56 | `token-batch-final-benchmark-vla_pruner-001` | C09 | [vla_pruner](../configs/experiments/openvla_oft/vla_pruner.yaml) | `ff880edb449d3840d2a7420e1363d155ac9dfe95` | R50 | different | `df68768c2b672d1b1bbfefded760cb6f4c42d1d9aa67961fddb5062dce0972ce` |

</details>

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
  visible. Historical repeats and superseded rows are not new algorithms.
- TEAM-VLA and VLA-IAP are independent paper-based implementations with
  explicit unresolved author-code conventions; consult their method READMEs.
  Their rows are not verified reproductions of the published implementations.
- EfficientVLA token core excludes layer pruning and DiT caching; FlashVLA
  ICS core excludes unresolved action reuse. ST-Merge is a paper-based spatial
  VLA port with shared two-encoder grouping and a timed dense projector bridge.
- VLA-InfoEntropy skips visual QUERY rows while retaining visual KV context.
  It is not context pruning; its query counts do not belong in the retention
  table. Paper-based methods disclose reconstruction choices in their READMEs.
- CogACT native parity does not verify the unidentified training fork's
  evaluation conventions. Its original four-episode Long failure is retained.
- Secondary CogACT native rechecks had observed thermal throttling; do not
  use them as stable references. The category-example cohort retained seven
  sparse thermal-flag samples; its late native drift was 0.435%. VOTE's tiny
  difference is not significant evidence of acceleration or regression.
- No native sampling accelerator is implemented. Do not relabel fewer DDIM
  steps as a faithful paper method. Full success campaigns remain unmeasured.
