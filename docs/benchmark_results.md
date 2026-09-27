# Benchmark results

One selected configuration per backend/method scope. Dev-200 is tuning evidence,
not final accuracy. Timing uses frozen four-suite prefixes (at most 128 controls each), ten warmup queries, one pass and the mean over all measured controls.
Native pre/post checks must pass. Final success
is unmeasured until the owner schedules the frozen 2,000-episode campaign.
Low success and slowdowns remain reportable. All primary comparisons use H8.
Complete measurements do not certify parameter-search completion; unfinished
searches are excluded from the final launch list even when Dev meets the target.
Selection targets at most 3 percentage points of Dev loss versus the matched
native; this is not an outcome filter. Δ pp uses that same-cadence reference.
Speedup >1 means faster and <1 means slower. Near-1 differences from one pass
do not establish a reliable gain; no repeat standard deviation is claimed.
Each speedup uses its own same-GPU native pair; the displayed native row is
not a universal denominator. Absolute latency across GPU pairs is descriptive.
Measured GPU models: NVIDIA GeForce RTX 3090.

Selection digest: `99e58902a150c48761ba3c833d97dd9a15aebb89fca869457bbe6549a6acf5d0`.

## cogact

| Section | Preset | H | Dev SR (n/200) | Δ pp | ms/control | Speedup | Peak GiB | Final SR (n/2000) | Selection / scope |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Reference | `baseline` | 8 | 91.5% (183) | +0.0 | 23.399 | 1.000× | 14.73 | — | Native H8 reference. Native model. Search: native_reference. |
| 3.3 | `efficient_vla_layer_prune04` | 8 | 93.0% (186) | +1.5 | 21.797 | 1.071× | 13.19 | — | layer_prune04 gives 186/200, fastest valid timing among tested configurations meeting >= 177/200. Measured below-target boundary: layer_prune08=114/200, layer_prune06=156/200, layer_prune05=165/200. EfficientVLA calibrated static non-contiguous layer core, not dynamic routing or the full pipeline. Search: accuracy_boundary. |
| 5.1 | `actuation_slack_refresh_keep25` | 8 | 89.0% (178) | -2.5 | 41.122 | 0.569× | 14.74 | — | Keep25% gives178/200; stronger keep17.5% gives169/200. Retain the faithful full-cost slowdown. Deletion core with full auxiliary refresh cost. Search: accuracy_boundary. |
| 5.1 | `adp_keep125` | 8 | 91.0% (182) | -0.5 | 18.845 | 1.228× | 14.73 | — | Keep12.5% gives182/200 and the fastest eligible timing;6.25% falls to170/200. Preserve the reference motion gate and dense-refresh cap. Adaptive pruning. Search: accuracy_boundary. |
| 5.1 | `dart_keep25` | 8 | 89.5% (179) | -2.0 | 15.292 | 1.529× | 14.57 | — | Keep25% gives179/200; stronger18.75% and12.5% fall to164/200 and141/200. Pivot-based token selection. Search: accuracy_boundary. |
| 5.1 | `depth_cache_primary_floor75` | 8 | 88.5% (177) | -3.0 | 19.903 | 1.159× | 14.73 | — | Minimum merge0.75 gives177/200 and1.159x;0.85 also gives177/200 with only0.73% more measured speed. Retain the less aggressive0.75 setting as a practical one-pass timing tie. Primary-view depth merging only; unknown wrist/carrying rules excluded. Search: latency_plateau. |
| 5.1 | `divprune_keep375` | 8 | 92.5% (185) | +1.0 | 15.937 | 1.449× | 14.61 | — | Keep37.5% gives185/200 and improves paired speed; keep25% falls to175/200. Diversity-based token selection. Search: accuracy_boundary. |
| 5.1 | `fastv_keep25` | 8 | 90.0% (180) | -1.5 | 15.104 | 1.548× | 14.57 | — | Keep25% gives180/200 and the fastest eligible measured timing;18.75% gives178/200 without further acceleration, while12.5% falls to173/200. Attention-based visual token pruning. Search: accuracy_boundary. |
| 5.1 | `flash_vla_keep232` | 8 | 89.5% (179) | -2.0 | 23.173 | 1.017× | 14.71 | — | Keep232 recovers179/200 versus176/200 at224 and is faster than240; near-unity speed is not a reliable gain claim. Active ICS core only; unconfirmed action reuse excluded. Search: accuracy_boundary. |
| 5.1 | `safe_pruner` | 8 | 88.5% (177) | -3.0 | 15.805 | 1.460× | 14.74 | — | Keep25% meets177/200; stronger keep18.75% and12.5% give174/200 and163/200. Task-aware token pruning. Search: accuracy_boundary. |
| 5.1 | `sparsevlm_rank625` | 8 | 91.5% (183) | +0.0 | 16.106 | 1.434× | 14.56 | — | Rank scale0.625 gives183/200 and1.434x; stronger0.75 gives164/200 for little additional speed. Keep task-related selection and recycling active. Task-related selection and token recycling. Search: accuracy_boundary. |
| 5.1 | `specprune_vla_static112` | 8 | 90.0% (180) | -1.5 | 20.290 | 1.139× | 14.63 | — | Static deletion112 gives180/200 and the fastest tested eligible timing; deletion128 gives174/200. Static/dynamic pruning with bounded static deletion. Search: accuracy_boundary. |
| 5.2 | `action_cache_threshold80` | 8 | 89.0% (178) | -2.5 | 19.958 | 1.180× | 14.75 | — | Threshold0.80 with two native tail steps gives178/200 and1.180x. Threshold0.75 gives176/200; one tail step at0.80 gives171/200. Preserve the frozen independent bank and full retrieval/refinement cost. DDIM refinement adaptation; 1412 pairs from 77/80 independent successful episodes. Search: accuracy_boundary. |
| 5.2 | `actuation_slack_refresh_kv_keep50` | 8 | 89.5% (179) | -2.0 | 46.395 | 0.499× | 14.74 | — | asr_kv_keep50 gives 179/200, fastest valid timing among tested configurations meeting >= 177/200. Measured below-target boundary: asr_kv_keep10=37/200, asr_kv_keep25=126/200, asr_kv_keep375=176/200. Retain the full-cost slowdown; auxiliary work is not free actuation overlap. ASR immutable dense-base KV reuse with unconditional same-frame full auxiliary refresh; no composition. Search: accuracy_boundary. |
| 5.2 | `bac_updates4` | 8 | 88.5% (177) | -3.0 | 21.983 | 1.061× | 14.73 | — | Four ACS updates gives177/200. Stronger three/two updates give169/152; six gives175 and is slower. Preserve BUA and global schedule allocation. ACS plus BUA; native CFG/DDIM residuals and frozen training-only statistics. Search: accuracy_boundary. |
| 5.2 | `c3ache_conditioned_residual_prefix6` | 8 | 91.5% (183) | +0.0 | 22.971 | 1.012× | 14.73 | — | Prefix4 is not faster than prefix6 and lowers success; caching four steps falls to162/200. Retain183/200 with no reliable speedup claim. Cross-chunk residual core; two cached steps, refresh interval two; not full Fast-WAM. Search: accuracy_boundary. |
| 5.2 | `dp_cache_interval6` | 8 | 91.0% (182) | -0.5 | 21.215 | 1.110× | 14.73 | — | Interval6 fills the profiled middle window2..7 and gives182/200 with1.110x. Extending to2..8 or1..8 gives134/200 or130/200. Keep all native DDIM transitions and both CFG branches. Native denoiser output reuse only; no VAE/compiler pipeline. Search: accuracy_boundary. |
| 5.2 | `efficient_vla_diffusion_cache_interval4` | 8 | 93.0% (186) | +1.5 | 21.208 | 1.093× | 14.73 | — | Interval4 gives186/200; interval5 falls to165/200. Choose the eligible stronger cache with its own paired timing. Within-sampling DiT feature cache; native sampler unchanged. Search: accuracy_boundary. |
| 5.2 | `evo_budget168` | 8 | 89.5% (179) | -2.0 | 22.656 | 1.030× | 14.73 | — | Measured candidates: evo_budget168=179/200, evo=165/200, evo_budget72=146/200. Choose the fastest eligible candidate. Layout search was frozen before Dev:640 independent episodes,24 candidates; none met the independent confirmation target. No convergence or global optimality claim. Globally budgeted residual cache with frozen independent rollout-searched layouts. Search: accuracy_boundary. |
| 5.2 | `temporal_redundancy_visual_update75` | 8 | 90.0% (180) | -1.5 | 23.954 | 0.970× | 14.85 | — | Update75% gives180/200; update62.5% falls to174/200. No tested reliable acceleration at the accuracy target. Visual attention core; trained action branch excluded. Search: accuracy_boundary. |
| 5.2 | `v_aefusion_s1` | 8 | 91.5% (183) | +0.0 | 24.227 | 0.973× | 14.74 | — | S1 gives183/200 and the fastest full-cost timing among all five active overlap windows; S5=177 and S9=14. All measured settings are slower than native. Stale-cognition early native head steps overlapped on a CUDA stream, mandatory fresh tail; no assumed actuation slack. Search: latency_plateau. |
| 5.2 | `vla_cache_budget256_protect75` | 8 | 89.5% (179) | -2.0 | 18.407 | 1.272× | 14.75 | — | Budget256 with75 protected tokens gives179/200 and1.272x;50 protected tokens falls to161/200. Preserve task protection, complete entropy statistics, and native KV visibility. Query-row reuse with retained visual KV. Search: accuracy_boundary. |
| 5.2 | `vla_infoentropy` | 8 | 87.5% (175) | -4.0 | 20.307 | 1.140× | 14.75 | — | Native-output SDPA still gives173/200; stricter similarity thresholds give175/200 and a smaller budget gives174/200. Keep the reference-close175/200 full-entropy port as an explicitly below-target result. Entropy-guided query-row reuse; below Dev loss target. Search: no_feasible_configuration. |
| 5.3 | `eco_vla_keep60` | 8 | 92.0% (184) | +0.5 | 20.089 | 1.162× | 15.77 | — | eco_keep60 gives 184/200, fastest valid timing among tested configurations meeting >= 177/200. Measured below-target boundary: eco_keep50=167/200, eco_keep55=175/200. EcoVLA dynamic FFN-channel and whole-head core; declared trigger/grouping conventions and frozen training prior. Search: accuracy_boundary. |
| 5.3 | `sp_vla_z025_min_actions3` | 8 | 89.0% (178) | -2.5 | 17.825 | 1.292× | 14.73 | — | Minimum three native actions in the unchanged six-command history, with pruning Z threshold 0.25, gives 178/200. Requiring only two native actions gives 176/200. Keep pruning and scheduling together. Pruning plus executed-action scheduling. Search: accuracy_boundary. |
| 5.4 | `keystone_k3` | 8 | 92.0% (184) | +0.5 | 23.715 | 0.996× | 14.73 | — | K3 gives184/200 and is fastest among measured active candidate counts; K4 gives178/200, while K8/K16 add latency. No measured end-to-end acceleration. Batched native-DDIM candidates and guarded geometric selection, not a trained sampler. Search: mechanism_limit. |
| 5.4 | `probeflow_eps0015` | 8 | 90.5% (181) | -1.0 | 21.510 | 1.083× | 14.73 | — | Measured candidates: probeflow=182/200, probeflow_eps0015=181/200, probeflow_eps03=176/200, probeflow_eps01=167/200. Choose the fastest eligible candidate. Adaptive lookahead core through the tested native DDIM coordinate bridge; not FM-trained. Search: accuracy_boundary. |
| 5.5 | `vote_ensemble_h8` | 8 | 87.0% (174) | -4.5 | 23.720 | 0.996× | 14.73 | — | H8 thresholds0.25/0.50/0.75 give174/174/172 of200, all below177; retain reference0.50 as an explicit negative two-vote adaptation. H8 two-vote overlapping-chunk ensemble core; no single-ACT architecture or extra queries. Search: no_feasible_configuration. |
| 6.1 | `awq_fp16_kernel` | 8 | 91.0% (182) | -0.5 | 17.005 | 1.356× | 5.90 | — | FP16 is the faster eligible option among both measured supported kernel precisions. Explicit FP16 cast boundary; training-only calibration. Search: mechanism_limit. |

## openvla_oft

| Section | Preset | H | Dev SR (n/200) | Δ pp | ms/control | Speedup | Peak GiB | Final SR (n/2000) | Selection / scope |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Reference | `baseline` | 8 | 99.0% (198) | +0.0 | 24.793 | 1.000× | 14.98 | — | Native H8 reference. Native model. Search: native_reference. |
| 3.3 | `efficient_vla_layer_prune03` | 8 | 96.5% (193) | -2.5 | 22.970 | 1.072× | 13.80 | — | layer_prune03 gives 193/200, fastest valid timing among tested configurations meeting >= 192/200. Measured below-target boundary: layer_prune04=181/200, layer_prune08=135/200. EfficientVLA calibrated static non-contiguous layer core, not dynamic routing or the full pipeline. Search: accuracy_boundary. |
| 5.1 | `actuation_slack_refresh_deletion` | 8 | 97.5% (195) | -1.5 | 44.000 | 0.563× | 14.69 | — | Halving retention to5% lowers success without improving full-cost timing; dense auxiliary refresh remains included. Deletion core; not real-robot overlap. Search: latency_plateau. |
| 5.1 | `adp_keep1875` | 8 | 97.5% (195) | -1.5 | 20.140 | 1.260× | 14.68 | — | Keep18.75% gives195/200 and the fastest eligible timing; keep12.5% falls to189/200 without further measured speed benefit. Adaptive pruning. Search: accuracy_boundary. |
| 5.1 | `dart_keep1875` | 8 | 97.5% (195) | -1.5 | 16.131 | 1.535× | 14.60 | — | Keep18.75% is the fastest tested eligible point (195/200); keep12.5% gives191/200. Pivot-based token selection, later pruning layer. Search: accuracy_boundary. |
| 5.1 | `depth_cache_primary` | 8 | 98.0% (196) | -1.0 | 25.041 | 0.990× | 14.68 | — | Ceiling0.90,progressive window2,and minimum merge0.25 all retain196/200 but do not improve full-cost timing. Keep reference-close primary-view core; wrist and ambiguous carrying logic remain excluded. Primary-view depth merging only; unknown wrist/carrying rules excluded. Search: latency_plateau. |
| 5.1 | `divprune` | 8 | 97.0% (194) | -2.0 | 23.605 | 1.039× | 14.62 | — | Keep50% remains the only eligible tested point; keep43.75%,37.5%,25% give191,187,188/200. Diversity-based token selection. Search: accuracy_boundary. |
| 5.1 | `efficient_vla_keep96` | 8 | 98.5% (197) | -0.5 | 21.083 | 1.180× | 14.61 | — | Keep96/view gives197/200; stronger80 and64 give191/200 and182/200. Preserve all selection components. Declared OFT token-reduction core. Search: accuracy_boundary. |
| 5.1 | `fastv_keep3125` | 8 | 96.5% (193) | -2.5 | 17.404 | 1.432× | 14.60 | — | Keep31.25% is the fastest tested eligible point (193/200); keep25% falls to183/200. Attention-based visual token pruning. Search: accuracy_boundary. |
| 5.1 | `flash_vla_keep192` | 8 | 98.0% (196) | -1.0 | 26.203 | 0.937× | 14.66 | — | Keep176 retains 196/200 but is not faster; retain keep192 and report the full-cost slowdown. ICS core only; action reuse is unconfirmed and excluded. Search: latency_plateau. |
| 5.1 | `safe_pruner_keep1875` | 8 | 96.0% (192) | -3.0 | 16.233 | 1.528× | 14.68 | — | Keep18.75% meets192/200; stronger keep12.5% falls to189/200. Task-aware token pruning. Search: accuracy_boundary. |
| 5.1 | `sparsevlm_rank50` | 8 | 96.0% (192) | -3.0 | 17.275 | 1.422× | 14.60 | — | Rank scale0.50 gives192/200; stronger0.625 and0.75 give190/200 and168/200. Recycling remains active. Task-related selection and token recycling. Search: accuracy_boundary. |
| 5.1 | `specprune_vla_dynamic80` | 8 | 97.5% (195) | -1.5 | 18.730 | 1.307× | 14.59 | — | Dynamic keep80% gives195/200; keep60% gives190/200. Score-first migration scope remains explicit. Score-before-first-prune variant, not author-default scheduling. Search: accuracy_boundary. |
| 5.1 | `vla_pruner_keep125` | 8 | 96.5% (193) | -2.5 | 17.668 | 1.393× | 14.68 | — | Keep12.5% gives193/200; stronger9.375% and6.25% give181/200 and171/200. Preserve dual-score selection. Declared progressive visual pruning core. Search: accuracy_boundary. |
| 5.2 | `actuation_slack_refresh_kv_keep175` | 8 | 97.0% (194) | -2.0 | 46.145 | 0.533× | 14.98 | — | asr_kv_keep175 gives 194/200, fastest valid timing among tested configurations meeting >= 192/200. Measured below-target boundary: asr_kv_keep10=185/200, asr_kv_keep05=168/200. Retain the full-cost slowdown; auxiliary work is not free actuation overlap. ASR immutable dense-base KV reuse with unconditional same-frame full auxiliary refresh; no composition. Search: accuracy_boundary. |
| 5.2 | `temporal_redundancy_visual` | 8 | 96.0% (192) | -3.0 | 25.941 | 0.947× | 15.19 | — | Update50% gives192/200; update25% falls to182/200 without a timing benefit. Report the slowdown. Visual attention core; trained action branch excluded. Search: accuracy_boundary. |
| 5.2 | `vla_cache_budget256_protect50` | 8 | 97.0% (194) | -2.0 | 24.345 | 1.043× | 15.03 | — | Budget256 with50 protected tokens gives194/200 and1.043x full-cost speed;25 protected tokens falls to190/200. Task protection and complete entropy statistics remain active. Query-row reuse with retained visual KV. Search: accuracy_boundary. |
| 5.2 | `vla_infoentropy_budget224` | 8 | 96.0% (192) | -3.0 | 25.058 | 0.985× | 15.03 | — | Budget224 meets192/200 and reduces full-cost overhead;256 falls to189/200 and reducing entropy protection quotas gives179/200. No reliable acceleration over native is claimed. Entropy-guided query-row reuse with retained KV. Search: accuracy_boundary. |
| 5.3 | `eco_vla_keep625` | 8 | 96.5% (193) | -2.5 | 23.660 | 1.037× | 17.04 | — | eco_keep625 gives 193/200, fastest valid timing among tested configurations meeting >= 192/200. Measured below-target boundary: eco_keep60=186/200, eco_keep50=190/200. EcoVLA dynamic FFN-channel and whole-head core; declared trigger/grouping conventions and frozen training prior. Search: accuracy_boundary. |
| 5.3 | `sp_vla_z025_min_actions3` | 8 | 97.5% (195) | -1.5 | 20.921 | 1.184× | 14.68 | — | Minimum three native actions in the unchanged six-command history, with pruning Z threshold 0.25, gives 195/200. Requiring only two native actions gives 191/200. Keep pruning and scheduling together. Pruning plus executed-action scheduling. Search: accuracy_boundary. |
| 6.1 | `awq_fp16_kernel` | 8 | 97.5% (195) | -1.5 | 21.359 | 1.161× | 6.12 | — | FP16 is the faster eligible option among both measured supported kernel precisions. Explicit FP16 cast boundary; training-only calibration. Search: mechanism_limit. |

## Useful auxiliary configurations

Secondary evidence only; these do not enter the default final-evaluation queue.

| Backend / preset | Dev SR (n/200) | ms/control | Speedup | Reason / scope |
|---|---:|---:|---:|---|
| openvla_oft / `awq` | 99.0% (198) | 25.626 | 0.967× | Retained accuracy/precision alternative; not a second primary. BF16-kernel W4A16. |
| cogact / `awq` | 91.5% (183) | 19.262 | 1.197× | Retained accuracy/precision alternative; not a second primary. BF16-kernel W4A16. |
| cogact / `actuation_slack_refresh_keep50` | 91.0% (182) | 45.692 | 0.512× | 182/200 at higher full refresh cost than keep25. Deletion core; full auxiliary cost. |

## Not certified for final evaluation

Diagnostic measurements below do NOT certify the unresolved core or authorize final runs.

| Backend / method | Dev SR (n/200) | ms/control | Speedup | Reason |
|---|---:|---:|---:|---|
| cogact / `st_merge` | 81.0% (162) | 19.809 | 1.177× | Source/target direction and one-to-many merge rule remain unresolved. |
| openvla_oft / `st_merge` | 98.5% (197) | 23.994 | 1.033× | Source/target direction and one-to-many merge rule remain unresolved. |
| openvla_oft / `team_vla` | 83.5% (167) | 19.226 | 1.279× | Soft-weight direction and merge normalization lack an executable reference. |
| openvla_oft / `vla_iap` | 94.5% (189) | 24.090 | 1.021× | Semantic encoder/alignment and centering construction remain unverified. |

## Evidence index

Raw measurements and resource locations are ignored local evidence, not public files.
Use the selection manifest and generated readiness JSON to resolve exact records.
Recorded revisions predating history cleanup can be resolved through the
[history map](../scripts/history_commit_map.json); recorded source IDs are not rewritten.

| Backend / preset | Dev experiment | Timing record | Dev / timing commit | Record digest |
|---|---|---|---|---|
| cogact / `baseline` | `a293c4ce24a5b840f961549f41769a1f59f1952c990b14d0ebef644ce8bda358` | `compact-41-4-cogact-1--native-pre` | `e86cbe2edc52d3bc69122a66dc727a8aa1109d93` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `c3be4cb17862b40f8519c7c2a4a52805b268f4e547991a93fae666ee733fb862` |
| cogact / `efficient_vla_layer_prune04` | `2c4f114dbefc70fd9001735c927d158cfb17bd28fee9d21786a3676ebed679fd` | `routing-layers-v1-124-7-cogact--layer_prune04` | `0281aa8f0f59a2ef08380f36ea95362b27748d90` / `0281aa8f0f59a2ef08380f36ea95362b27748d90` | `46a1ecab3fb1b69ae5b614c6cd22c0afc3d625fa338d906ee0c9bcb2a2fcbd21` |
| cogact / `actuation_slack_refresh_keep25` | `1745acc6a866125188a9997d38d3a193bf3f2238f71a72e8522ed6a2933d2115` | `compact-41-4-cogact-1--actuation_slack_refresh_keep25` | `1dc47be1a1e7d1cd8623a50e329d3a72e5daa8ef` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `ba45d59001f53064891ce78afc87457c6fcfb42d74aae8f49bd7b9ed91127717` |
| cogact / `adp_keep125` | `c903c16ec47f230039b17243336af1865e56397c01e2dab8090c51685f02efe1` | `search-r16-41-7-cogact--adp_keep125` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `523c898b888194f2503aa3d509b855a544a1c3d428dc003e0b64f13d84e0b222` |
| cogact / `dart_keep25` | `4d0507e9f3f98e0b60899e6bb2a488f0ffbf66ec8b09cad2cde88a13c1f7d9c5` | `search-r1-45-5-cogact-0--dart_keep25` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `04571245becc1aaf59e7862c21dfd61687b3e70af4d7def0eda78c69bcccba8e` |
| cogact / `depth_cache_primary_floor75` | `7d96c64920f5c4d64b45d896c858a33e184162a558c5beaa4f093c393720d34a` | `search-r27-41-7-cogact--depth_cache_primary_floor75` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `c60bc8918786bb101d5da7dfb2270c9f7c94e9e7b3e0bef92a4a6ad38c93c9f7` |
| cogact / `divprune_keep375` | `aa02db4d4648252bb86bbb9ce105d0cf3e3ee94946947a46d744454db64ec2bc` | `search-r10-recheck-41-7-cogact--divprune_keep375` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `527d0f3634ed62094523ca80d6aa4479ebcbee75a5fb619640f8f5b6fed76754` |
| cogact / `fastv_keep25` | `a581c15da97c796e59b7c3a2ceb059e77d7abc62ff64e4b11386f5ac9ed724bf` | `search-r1-45-5-cogact-0--fastv_keep25` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `ecef890ff4e4d70a080d005228ee81213473005d2814f362e2a61e29059e1b15` |
| cogact / `flash_vla_keep232` | `78e61c2a5e331e0593f782b04a4ccec96cf5e2a17892ffc603699eabb99d3a44` | `search-r17-125-5-cogact--flash_vla_keep232` | `96c738e00f6342e85790130ea2e660eec6d30dee` / `faabaa5959b49364a808c7327559087f3cf1c892` | `cc458a77aefc8f5d3149096540bdccdee1d5d239f87d5245b7cd72d9f45a9388` |
| cogact / `safe_pruner` | `bdc7820db41f431415a887ed411fead64ed546d22e056708ae5ef56344353f26` | `compact-43-0-cogact-token-retry--safe_pruner` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `2a49223fe7b558d8ca59b6ceab6557f06fc354b9` | `3ba866bc10cd8608a9e9300225f35f6f3a43c3d9c1397d8f0897837bde572a1d` |
| cogact / `sparsevlm_rank625` | `a0d03b1c34f2357464e19f61ff9e670df3480f699f5e4978512b777c19bea8ba` | `search-r23-41-7-cogact--sparsevlm_rank625` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `264dd9f9bd35e2ca092b01a36ba0a8953608dba6a183919cf5bb85952d35aaa5` |
| cogact / `specprune_vla_static112` | `d1a060bd5b0c4e31b979a8f9e8d98c7a9ae441766e68e7ca093b06fcbda92f0e` | `search-r1-recheck2-41-7-cogact--specprune_vla_static112` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `b9d53df2dc6d5ca8f0c09c47384cd94f376351308573344ce8c9a53c7fe2029a` |
| cogact / `action_cache_threshold80` | `07d39fad8017a5d0d5e41055d16e57df95a07b79ae182ff50b46b5188ba61b1a` | `search-r1112-124-6-cogact--action_cache_threshold80` | `faabaa5959b49364a808c7327559087f3cf1c892` / `96c738e00f6342e85790130ea2e660eec6d30dee` | `88c1bdc966c9cf2b464b5fb3ff19d3dd563b43307a139f155bea7b90bf2c1f4b` |
| cogact / `actuation_slack_refresh_kv_keep50` | `a335934e5a3e5f57336b356f9da9fd8d227ab7e9155a6fc2b2f13d17f5a74f10` | `routing-asr-v2-45-6-cogact--asr_kv_keep50` | `86343119fa1e50550eb90a994b9e4a6cebd779b3` / `86343119fa1e50550eb90a994b9e4a6cebd779b3` | `98252b428fb9dc09532504ff45493504345b8a1eab3a4bbe7a0ae9800440dd64` |
| cogact / `bac_updates4` | `ead4b40ef77bafdf2f26a73525b15a847f5459460cbf7dd48a5eceb065cfc8ee` | `bac-v1-41-7-recheck4-cogact--bac_updates4` | `d338c60a1fe7d206157b70cb47227c35c664f2ac` / `d338c60a1fe7d206157b70cb47227c35c664f2ac` | `007e78c7dcac729f7653dccccd38164545312bd8c4192240e6e4ae925548bc0a` |
| cogact / `c3ache_conditioned_residual_prefix6` | `c692772b868aa42a4d774070b8028db7bf93ab031712cb7cb5d7b65d9cd03d6e` | `cache-v2-45-6-cogact--c3ache_conditioned_residual_prefix6` | `d3c46d34696eb47f753dba3306526bad5ad461c7` / `d3c46d34696eb47f753dba3306526bad5ad461c7` | `539f67019c803f6333f2f804c123a83bd0f31e373402b7c5fc321eab70d1662a` |
| cogact / `dp_cache_interval6` | `2243cf3b022aafef382570f434f2e1fc298613e6434860a84873d27be7ca3d16` | `search-r1112-124-6-cogact--dp_cache_interval6` | `faabaa5959b49364a808c7327559087f3cf1c892` / `96c738e00f6342e85790130ea2e660eec6d30dee` | `6cc089a6b63d02a883e2f84613bb972e5b3daceebb692db6142c4778323e59a5` |
| cogact / `efficient_vla_diffusion_cache_interval4` | `d1b43467534ec99b6bf7a6d3eb7523fd184068eb8834ff9d9d5e4a2d6d8d1f31` | `search-r1-recheck-41-7-cogact-5--efficient_vla_diffusion_cache_interval4` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `513e6fb4b5cd17be3ac8adabfe3493923cfcd6da35be5c5ba6ce3774e013a3c2` |
| cogact / `evo_budget168` | `d653e21d06192fafc907bcf141442ab7c5e7d0d604c36c323215b48c87412cc2` | `evo-v1-124-7-cogact--evo_budget168` | `195d27fa6e65889e964e9a23483d43319b57e50c` / `195d27fa6e65889e964e9a23483d43319b57e50c` | `fae1a7db1d8fb71f002d70731365848c236d3202250687bc7ab2dd56fada4bca` |
| cogact / `temporal_redundancy_visual_update75` | `fd8de3e4c0776b169a8a488af5f09a840a22c597869235a171387ed9ab2adca5` | `cache-v3-45-6-cogact--temporal_redundancy_visual_update75` | `d3c46d34696eb47f753dba3306526bad5ad461c7` / `d3c46d34696eb47f753dba3306526bad5ad461c7` | `35d15909a0b0e2be29c00576968e94183efe13ea35c330a6d70b828780b89793` |
| cogact / `v_aefusion_s1` | `1195ea8e4181a5a4bf42f16aa8077a9344695797ff52f31f050cab5d4de04539` | `v-aefusion-v1-125-7-drift-recheck-cogact--v_aefusion_s1` | `f21d38df23182cf63c82159c4e1b7834d9259752` / `f21d38df23182cf63c82159c4e1b7834d9259752` | `1cf44684809e41e1c1ade1316d2f5720095b7c47c926f7528191922f92c08c47` |
| cogact / `vla_cache_budget256_protect75` | `9927e2ef608b3b72e97501649002c6a57b707b0566404abae53f571a1422666f` | `search-r25-125-5-cogact--vla_cache_budget256_protect75` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `9a790606de30f1470c6aa17f60f15bb3253c85fe273d641090a7df9f82b65292` |
| cogact / `vla_infoentropy` | `3e7f5140c644c221a9a02a45731cf1e949650a3564053c9d662f85e37b11cabb` | `compact-43-2-cogact-inputs-retry--vla_infoentropy` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `2a49223fe7b558d8ca59b6ceab6557f06fc354b9` | `3330159196088131057ec15ea17cdc1667f6f9668632ed2266ff747cae62d186` |
| cogact / `eco_vla_keep60` | `9b3df9db8fb40e0e1a2088fc5ec34728f4cef5d25c363dff38c9bc22ecf8a308` | `routing-eco-dyn1-45-7-cogact--eco_keep60` | `0281aa8f0f59a2ef08380f36ea95362b27748d90` / `a7c6b45a939067ff2dd25b438068333659391ed2` | `4a8fd532058d373edabd54bf8fa874bfca0d9ead3c2433f15df23d7a03cc7794` |
| cogact / `sp_vla_z025_min_actions3` | `e5ff0e4f5f44dd1112561436b4d6c982a84db488c0c2af8cdb0c3228fdc657a8` | `search-r20-41-7-cogact--sp_vla_z025_history3` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `1f5fb02f8a6417f3a9c787db7838353c79a99c4b2111031977193c1ca41774f9` |
| cogact / `keystone_k3` | `f13cedd6f644a8494c160c48269b65b6ea280edf87ec9a40223127add688bd2a` | `keystone-v1-125-6-cogact--keystone_k3` | `93f4848e25b37cd2928241719cd8557f4e652bf8` / `93f4848e25b37cd2928241719cd8557f4e652bf8` | `ac0630d0ddc3f1f1ff62e255ed92748eacbf230a7ec28842e29181187b9d7196` |
| cogact / `probeflow_eps0015` | `b71a6562b1dcf2888da0ea30fe58e81609ab848e884364482b68948389d09403` | `probeflow-eps0015-v1-41-7-cogact--probeflow_eps0015` | `b7db84fde6b0dcbd96eac64c37b43b7376929991` / `b7db84fde6b0dcbd96eac64c37b43b7376929991` | `43f7ec53f3b6aebe8fcf70d6d266fd38511d1a846f32c040c5e8898ef32ba25a` |
| cogact / `vote_ensemble_h8` | `9ebbbb6d52e6b5e4131af9d6bc14a493375a387d02bb298bceedb70680ac660c` | `search-vote-124-6-cogact--vote_ensemble_h8` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `8f775c6f832ad620cd65450fa517502a6cd0c29bc684c904f4710b310e2a0baa` |
| cogact / `awq_fp16_kernel` | `be03b017c6a40862b3ebc5afa78a0ce2cf31111d49e8763c1fe5d0650391c117` | `compact-42-0-cogact-quant--awq_fp16_kernel` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `465838d6af15428eace0f168b48392ce6886137b` | `cb173d2eb42c17ac8ae3398af5f884071e082d7e3ed7db8ac15db2f40bfa4a61` |
| openvla_oft / `baseline` | `83a2952de24bf599cafcbe504ce1b8e0c5f91824411122370deec9846dbaa71e` | `compact-124-7-openvla_oft-quant--native-pre` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `2a49223fe7b558d8ca59b6ceab6557f06fc354b9` | `77575dfbcb36ed385377a886f994a3d21e6cb2eb1de5d620f896aea2f9f493cd` |
| openvla_oft / `efficient_vla_layer_prune03` | `4d5841c74fa9f501a353b8e29e29afb085d26b3b1af36d17906060b696e5c792` | `routing-layers-v3-45-5-openvla_oft--layer_prune03` | `0281aa8f0f59a2ef08380f36ea95362b27748d90` / `a7c6b45a939067ff2dd25b438068333659391ed2` | `2de43370c80511d8e1c3f3be88da768559dfa66708f3ef0c17638717831fcde6` |
| openvla_oft / `actuation_slack_refresh_deletion` | `2e2b75988b34636d5fa39e24e573aa4a159c179d48954e47203a23897921b39c` | `compact-45-5-openvla_oft-recheck--actuation_slack_refresh_deletion` | `f63c95a6f22a8fd2c1fcdc9fed2441ecb376379d` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `515f46e72adb7e0bd549ce44c0e3716db4d7b3573c3fdf3ad56c826d9df12c2d` |
| openvla_oft / `adp_keep1875` | `58bdb42316badeba12a20a4e3522c93743998b542c79bde2983afc2f661b1e64` | `search-recovery-125-6-openvla_oft-0--adp_keep1875` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `ced38686f07822d35bfb5812d002d8793d6a0dd20e0f4559297f674b220b4990` |
| openvla_oft / `dart_keep1875` | `66630bb5525cf103faf07c776372466697caa7b3d157a532be1923dda958a0ab` | `search-r78-124-6-openvla_oft--dart_keep1875` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `81bfc1a595ebaba3c4bc9b478f5db1734e7eb0195e082ac346c6d4ba43e3027c` |
| openvla_oft / `depth_cache_primary` | `23ee03e39b8e019f7c9674baabb00b045f8f303a25ea442dd81e7a27516cb331` | `compact-45-5-openvla_oft-recheck--depth_cache_primary` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `498b1401c4d447a11878f85934fb8fac87172fb74647f47716cf0afae77c7fc8` |
| openvla_oft / `divprune` | `4482ae63177ce1099a9ba5360b83ab01317cb8d5519527dbd8aecd75cb083aea` | `compact-45-6-openvla_oft-1--divprune` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `2985f18b349a71a33698d80f742a62ab0fe8ff8250dc1e1ae0b1008dae27f93c` |
| openvla_oft / `efficient_vla_keep96` | `3695d50ab9056a7d393c7c8d6715f33a53c29ae90306facde37abedda3e5ff38` | `search-r10-124-6-openvla_oft--efficient_vla_keep96` | `96c738e00f6342e85790130ea2e660eec6d30dee` / `faabaa5959b49364a808c7327559087f3cf1c892` | `727a146c27ad5151ef3a033b967b953692982dde0bd971acdf52dffd3e9b623b` |
| openvla_oft / `fastv_keep3125` | `0070caa61e8205fba10bab5c41cfc52a5789f0750f949eaa45166da55407fc23` | `search-r456-124-6-openvla_oft-0--fastv_keep3125` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `b69a94d8578b4fab49d9dd0127410f3014f59fbf173d0cebe0a215a359341e6e` |
| openvla_oft / `flash_vla_keep192` | `143755ed1aee192a41d545ed818b591dcfb00fc7d0f0c93802b56b060fb7f84e` | `compact-45-6-openvla_oft-2--flash_vla_keep192` | `1d954ec8f661f0f700810985e2e3d8b8ae5205aa` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `50392782613813f5ba30917361231127bbaf4cda0de1347d6322856fbd2853a6` |
| openvla_oft / `safe_pruner_keep1875` | `c6224353d91f72c020e1c61e427e9cec0964ba782497b05e1fe7db110a06ab07` | `search-r23-124-6-openvla_oft-0--safe_pruner_keep1875` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `ac964d850c6460e87b66c5c538681330e9653528fbff8cd2f16d075d81b2ab74` |
| openvla_oft / `sparsevlm_rank50` | `f20516d6ae83064a018e357944d7325885f1af3fbba0a8788ea76117ee0b48b1` | `search-r1-45-6-openvla_oft-1--sparsevlm_rank50` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `fb2a0ffe4c534d33840febac6dd9e1b1d16eda4487ac325cc39228ada5f7f891` |
| openvla_oft / `specprune_vla_dynamic80` | `3f7f62a0d04cbdcedc73cbab0a908e099270f7205d965fe8d0a7e4558eadb206` | `search-r1-45-6-openvla_oft-4--specprune_vla_dynamic80` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `ac88a1a6b771d873bd5c35f046d8dffc0b7cd3ef6b2f418f471ffa1ce0d3e4d9` |
| openvla_oft / `vla_pruner_keep125` | `67e27456b57078bbcc5d7a69c66702def13efe6de9858c124b549353013a056f` | `search-r1-45-5-openvla_oft-3--vla_pruner_keep125` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `20eb17c08bc7c89f797f7164cb7f7d1ed5e3a74bea773ada8cef4ee64a21a542` |
| openvla_oft / `actuation_slack_refresh_kv_keep175` | `05a0210e3be30b69e8bfdf6a755eb172425012bd4ed16f6ede86b35f62e27ddf` | `routing-asr-v3-45-5-openvla_oft--asr_kv_keep175` | `86343119fa1e50550eb90a994b9e4a6cebd779b3` / `86343119fa1e50550eb90a994b9e4a6cebd779b3` | `99ccf2f4666d0668321c8343bd81c5cf3a04ceac869e12cf5684e6ca157024da` |
| openvla_oft / `temporal_redundancy_visual` | `a0480729b2ee658d04b2ce8dc01bd888cadca71e070a130a2cb0143cf2efd768` | `cache-v1-45-7-oft-recheck--temporal_redundancy_visual` | `92b64fc3044235db11f39fe1e0f9c26df4ac1781` / `92b64fc3044235db11f39fe1e0f9c26df4ac1781` | `2ccdeeaaf24de6c4d950289190a72706b8eadb5bdeaea93e0df62f9ec9c71e15` |
| openvla_oft / `vla_cache_budget256_protect50` | `561783bb97b9dc7ae23abc9105ba90b1f11eb6468d675580da273605d05e9a77` | `search-recovery-125-6-openvla_oft-0--vla_cache_budget256_protect50` | `96c738e00f6342e85790130ea2e660eec6d30dee` / `faabaa5959b49364a808c7327559087f3cf1c892` | `d65232be7392f26af0b9ef1d1d0eadddaee7b903ad5c7da580edec30905e9c0e` |
| openvla_oft / `vla_infoentropy_budget224` | `84aa0e0fe7b5e53c2b930e0a2b6cfdefaecd2fe9632e8e5b60d683594fb9d11f` | `search-r1-45-7-openvla_oft-5--vla_infoentropy_budget224` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `e861dba523cc3c90a881ec09c1a03cccaf5a16b9d442ec1357d1ccd8ee898e4e` |
| openvla_oft / `eco_vla_keep625` | `420ed8a10ec07dafaa45251f29e974609a77d157b8a62e3fd5c2212a0924d73a` | `routing-eco-dyn3-45-5-openvla_oft--eco_keep625` | `0281aa8f0f59a2ef08380f36ea95362b27748d90` / `a7c6b45a939067ff2dd25b438068333659391ed2` | `ec654378d1c2e997a3b0fe1c03a643623644a7fb818c14d65535b2dd3055fb02` |
| openvla_oft / `sp_vla_z025_min_actions3` | `1d80b73d202ee632e4220288cfd3bec1fc5f2a62bb5baa5325fe800195a10572` | `search-r78-124-6-openvla_oft--sp_vla_z025_history3` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `bd1c4113caccb6ebfa41678e0b431f392f067adb6bc589dc2db7166a1de11913` |
| openvla_oft / `awq_fp16_kernel` | `566ed69c9ce2db872ba40344f9123dac0ba1db2f05ef3c240c7f156406fe6d66` | `compact-124-7-openvla_oft-quant--awq_fp16_kernel` | `e86cbe2edc52d3bc69122a66dc727a8aa1109d93` / `2a49223fe7b558d8ca59b6ceab6557f06fc354b9` | `b1a7536157e29af90a22cc43bb47a37850b6d3ea9e3d8b499df6089f0311cab1` |
| openvla_oft / `awq` | `5d64703c2acfcfe4a33ed22f76215b16457c1d61cc0be948e134f97124f509cf` | `compact-124-7-openvla_oft-quant--awq` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `2a49223fe7b558d8ca59b6ceab6557f06fc354b9` | `8720ab41aa47c710bc8c10a98479089045df80598caa44b3cf79b9645c6b8a8d` |
| cogact / `awq` | `a223cf526c859a8a6071ba21da16d2be890c981c5048f52ab863fda22eace8dc` | `compact-42-0-cogact-quant--awq` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `465838d6af15428eace0f168b48392ce6886137b` | `2b53ecaeb6d71307c5630e29c56920594ad7c3f6663d15eb0660d780aebc8f2f` |
| cogact / `actuation_slack_refresh_keep50` | `3e9ab07826e30785e609207a39c4c964aadc37185ff424e01b3643cfb527bd3c` | `compact-41-4-cogact-1--actuation_slack_refresh_keep50` | `1dc47be1a1e7d1cd8623a50e329d3a72e5daa8ef` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `c688748fc315d380b58c73462d81222c806f6ed88f9cf027c16ad63c3ca73762` |
| cogact / `st_merge_keep192` | `2715131409b114d95dc8dc1349d3ea4da8b4baa5ab6f0e78127aff0d3dc431eb` | `compact-43-2-cogact-3--st_merge_keep192` | `1dc47be1a1e7d1cd8623a50e329d3a72e5daa8ef` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `c67baa5ef382b3aa4fbdce5a312500f136981ba4179eb5163cb4f29398bb594f` |
| openvla_oft / `st_merge_keep192` | `d6e8965949d65ac92a7079a13bbe478bc63fd38ef9d9657f513d8051b75f9a46` | `compact-45-5-openvla_oft-recheck--st_merge_keep192` | `1dc47be1a1e7d1cd8623a50e329d3a72e5daa8ef` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `945d70c1affb3d3c243e7eed3a688112f005fbf591f8e87be257572bd16dc62a` |
| openvla_oft / `team_vla_merge160` | `02fc7c9f038ee76b07c85196460993479de227eba74c4343cca09310552f327c` | `compact-45-7-openvla_oft-3--team_vla_merge160` | `1dc47be1a1e7d1cd8623a50e329d3a72e5daa8ef` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `5f45fec789ab8a8b09723be9fa5fcb66ebf42d78ee443fa28c56b7be88d7a7bd` |
| openvla_oft / `vla_iap` | `0a6a2c9c71aefa83b1f9ce71ac8d7813d3bb4ed9c4494769d977fdf123756ae0` | `compact-45-7-openvla_oft-3--vla_iap` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `c51e783e5187bfc02f9c44dd8c91c8309d4a26383cb18a190dfa3dd0378a862b` |

## Independent action codec reconstruction

Frozen corpus `01a8ae346abe98d43ddc307e2ab3db08c99a0229b58074e7b199ec2cd4500762`; 40-frame windows, one per episode, disjoint calibration/evaluation episodes; 4 evaluated suites.
This is in-domain demonstration reconstruction, not policy success or VLA speedup.
Latency includes representation transforms and CPU/device transfers; native CPU and GPU implementations are labeled separately. Nominal bits are not entropy-coded size.
BEAST basis variants are retained rate–distortion points, not extra policy defaults.
X-Tokenizer uses a disclosed command-integrated pose representation and the initial observed state at decode; its conditional result is not an unconditional codec score.
Pretraining overlap is not excluded; native block sizes/receptive fields differ.

| Codec / preset | Device | Windows | Raw MSE | Raw MAE | Gripper sign | Tokens/frame | Bits/frame | Encode ms/window | Decode ms/window |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fast_plus | AMD EPYC 7532 32-Core Processor | 128 | 0.0004145 | 0.01077 | 1.0000 | 1.661 | 18.266 | 0.375 | 0.185 |
| beast / degree=4, gripper_zero_order=False, num_basis=5 | AMD EPYC 7532 32-Core Processor | 128 | 0.0110235 | 0.04374 | 0.9955 | 0.875 | 7.000 | 5.145 | 2.510 |
| beast / degree=4, gripper_zero_order=False, num_basis=10 | AMD EPYC 7532 32-Core Processor | 128 | 0.0043754 | 0.02078 | 0.9990 | 1.750 | 14.000 | 15.913 | 7.933 |
| beast / degree=4, gripper_zero_order=False, num_basis=20 | AMD EPYC 7532 32-Core Processor | 128 | 0.0018109 | 0.00951 | 0.9994 | 3.500 | 28.000 | 38.072 | 18.877 |
| vq_vla | NVIDIA GeForce RTX 3090 | 128 | 0.0026424 | 0.02868 | 0.9996 | 0.800 | 6.400 | 101.025 | 84.979 |
| actioncodec | NVIDIA GeForce RTX 3090 | 128 | 0.0010591 | 0.01865 | 1.0000 | 0.800 | 8.800 | 17.214 | 17.007 |
| x_tokenizer | NVIDIA GeForce RTX 3090 | 128 | 0.4746804 | 0.33120 | 0.4721 | 1.000 | 11.000 | 7.747 | 9.015 |

| Codec run | Measurement commit | Summary SHA256 |
|---|---|---|
| `codec-fast-plus-v2` | `e70caf8f8b9e7bc6d405012a433b3a49e2ae2c30` | `46540bca1f06d946a1c06ccf7fbeddf6b287eba6c6f5c0a30fc7432d1020d145` |
| `codec-beast-basis5-cpu-v1` | `605176e8ebd79f155d88cf2c82899ae60adb16f4` | `9c0a0e9f21ae289414ea26f8a0f0d1984ec6b907eee99302866d3b7724e52d4b` |
| `codec-beast-cpu-v1` | `605176e8ebd79f155d88cf2c82899ae60adb16f4` | `bb837a9657aa0e23d9c43153683338fabebd23398ef163657ee09ab69d45b429` |
| `codec-beast-basis20-cpu-v1` | `605176e8ebd79f155d88cf2c82899ae60adb16f4` | `e45c502dcd85c6f6de87a66a11cc960c53982024538d5aeab29fd5a06b2c19d3` |
| `codec-vq-vla-v2` | `e70caf8f8b9e7bc6d405012a433b3a49e2ae2c30` | `f7aed049337f6fd3af8f8a19c3c8f07c37f942b661a258c42127d25a93efb4ab` |
| `codec-actioncodec-v2` | `e70caf8f8b9e7bc6d405012a433b3a49e2ae2c30` | `4f550c96c00a8cdb15d7ccb91d36101706646ced4dc726fd20b82818ea3f5f4a` |
| `codec-x-tokenizer-v2` | `e70caf8f8b9e7bc6d405012a433b3a49e2ae2c30` | `48b67ed993ca192227e3a4fa247437d069a7cdbe4a8d02dfdfbc34ed5d822cec` |

<details>
<summary>Codec device checks (auxiliary, not default rows)</summary>

The small spline workload is faster on native CPU; both devices use four OpenMP threads and the same corpus. Retain the CUDA comparison, not a second primary.

| Run | Device | Raw MSE | Encode / decode ms per window | Summary SHA256 |
|---|---|---:|---:|---|
| `codec-beast-basis5-v2` | NVIDIA GeForce RTX 3090 | 0.0110235 | 13.278 / 6.622 | `436c77b75ccdf01c1824675dce9e767066ab145d398e8fc7c23e26a97f2b46b8` |
| `codec-beast-v2` | NVIDIA GeForce RTX 3090 | 0.0043754 | 45.569 / 22.683 | `b12c622bb6c26123c969eaaec6b41187f1ef829ee51cff00eb2d925e2325b575` |
| `codec-beast-basis20-v2` | NVIDIA GeForce RTX 3090 | 0.0018109 | 110.388 / 54.901 | `0785ce3eaa2bdc7a03aeb3bccb4cacd95ffb4f6da40eb880e63b507ad5bf4ad6` |

</details>
