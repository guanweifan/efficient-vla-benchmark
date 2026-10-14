# Benchmark results

One selected configuration per backend/method scope. Dev-200 is tuning evidence,
not final accuracy. Timing uses frozen four-suite prefixes (at most 128 controls each), ten warmup queries, one pass and the mean over all measured controls.
Native pre/post checks must pass. Final success is measured for 74/74 primary configurations, using one frozen 2,000-episode run per measured configuration.
Low success and slowdowns remain reportable. All native references use H8.
Adaptive rows execute real method-selected prefixes; H8 is their reference/default,
not their fixed execution length. The shown upper bound is native prediction length.
Complete measurements do not certify parameter-search completion; unfinished
searches are excluded from the final launch list even when Dev meets the target.
Selection targets at most 3 percentage points of Dev loss versus the matched
native; this is not an outcome filter. Δ pp uses that same native-H8 reference.
Speedup >1 means faster and <1 means slower. Near-1 differences from one pass
do not establish a reliable gain; no repeat standard deviation is claimed.
Each speedup uses its own same-GPU native pair; the displayed native row is
not a universal denominator. Absolute latency across GPU pairs is descriptive.
Measured GPU models: NVIDIA GeForce RTX 3090.

Selection digest: `727548c9b2fa51174d86634775b929b2ce2247a5cf10da045dbd5e92421e64e7`.

## cogact

| Section | Preset | Execution H | Dev SR (n/200) | Δ pp | ms/control | Speedup | Peak GiB | Final SR (n/2000) | Selection / scope |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Reference | `baseline` | 8 | 91.5% (183) | +0.0 | 23.399 | 1.000× | 14.73 | 90.60% (1812) | Native H8 reference. Native model. Search: native_reference. |
| 3.3 | `quantvla` | 8 | 90.5% (181) | -1.0 | 44.352 | 0.531× | 6.20 | 88.35% (1767) | Block64/16 give 181/178 of 200. Their valid latencies differ by 0.81%, a one-pass tie; retain the reference block64 with higher Dev success. Neither accelerates. Calibrated DuQuant plus ATM/OHB; physical W4/A8 storage with tile-local floating GEMM, not integer MMA. Search: latency_plateau. |
| 3.3 | `efficient_vla_layer_prune04` | 8 | 93.0% (186) | +1.5 | 21.797 | 1.071× | 13.19 | 90.15% (1803) | layer_prune04 gives 186/200, fastest valid timing among tested configurations meeting >= 177/200. Measured below-target boundary: layer_prune08=114/200, layer_prune06=156/200, layer_prune05=165/200. EfficientVLA calibrated static non-contiguous layer core, not dynamic routing or the full pipeline. Search: accuracy_boundary. |
| 5.1 | `a_mmr_current_frame_keep4375` | 8 | 89.5% (179) | -2.0 | 17.543 | 1.306× | 14.62 | 89.25% (1785) | Keep43.75% gives 179/200; keep37.5% falls to 175. Keep50% gives 177 with only about 2% higher paired speed, treated as a practical tie. Independent current-frame A-MMR core; native causal anchors and FP32 selection. Search: accuracy_boundary. |
| 5.1 | `actuation_slack_refresh_keep25` | 8 | 89.0% (178) | -2.5 | 41.122 | 0.569× | 14.74 | 87.90% (1758) | Keep25% gives178/200; stronger keep17.5% gives169/200. Retain the faithful full-cost slowdown. Deletion core with full auxiliary refresh cost. Search: accuracy_boundary. |
| 5.1 | `adp_keep125` | 8 | 91.0% (182) | -0.5 | 18.845 | 1.228× | 14.73 | 89.50% (1790) | Keep12.5% gives182/200 and the fastest eligible timing;6.25% falls to170/200. Preserve the reference motion gate and dense-refresh cap. Adaptive pruning. Search: accuracy_boundary. |
| 5.1 | `dart_keep25` | 8 | 89.5% (179) | -2.0 | 15.292 | 1.529× | 14.57 | 84.75% (1695) | Keep25% gives179/200; stronger18.75% and12.5% fall to164/200 and141/200. Pivot-based token selection. Search: accuracy_boundary. |
| 5.1 | `depth_cache_primary_floor75` | 8 | 88.5% (177) | -3.0 | 19.903 | 1.159× | 14.73 | 89.90% (1798) | Minimum merge0.75 gives177/200 and1.159x;0.85 also gives177/200 with only0.73% more measured speed. Retain the less aggressive0.75 setting as a practical one-pass timing tie. Primary-view depth merging only; unknown wrist/carrying rules excluded. Search: latency_plateau. |
| 5.1 | `divprune_keep375` | 8 | 92.5% (185) | +1.0 | 15.937 | 1.449× | 14.61 | 91.15% (1823) | Keep37.5% gives185/200 and improves paired speed; keep25% falls to175/200. Diversity-based token selection. Search: accuracy_boundary. |
| 5.1 | `eta_vla_ilsa_keep35` | 8 | 90.0% (180) | -1.5 | 16.190 | 1.456× | 14.60 | 89.80% (1796) | Keep35% gives 180/200; keep20% gives 179 with under 1% speed difference, while keep10% falls to 165. ETA-VLA ILSA core only; native cognition/DDIM preserved, TFM excluded. Search: accuracy_boundary. |
| 5.1 | `fastv_keep25` | 8 | 90.0% (180) | -1.5 | 15.104 | 1.548× | 14.57 | 89.45% (1789) | Keep25% gives180/200 and the fastest eligible measured timing;18.75% gives178/200 without further acceleration, while12.5% falls to173/200. Attention-based visual token pruning. Search: accuracy_boundary. |
| 5.1 | `flash_vla_keep232` | 8 | 89.5% (179) | -2.0 | 23.173 | 1.017× | 14.71 | 87.45% (1749) | Keep232 recovers179/200 versus176/200 at224 and is faster than240; near-unity speed is not a reliable gain claim. Active ICS core only; unconfirmed action reuse excluded. Search: accuracy_boundary. |
| 5.1 | `safe_pruner` | 8 | 88.5% (177) | -3.0 | 15.805 | 1.460× | 14.74 | 88.60% (1772) | Keep25% meets177/200; stronger keep18.75% and12.5% give174/200 and163/200. Task-aware token pruning. Search: accuracy_boundary. |
| 5.1 | `sparsevlm_v2_128` | 8 | 90.5% (181) | -1.0 | 15.916 | 1.453× | 14.56 | 87.05% (1741) | Profile 128 gives 181/200, the fastest measured eligible setting; profile 96 falls to 154/200. Profile 192 and its scale 1.5 variant give 184/182 but are slower. Official SparseVLM+ V2.0 selection on native causal cognition; scaled 256-patch grid, unchanged DDIM and H8. Search: accuracy_boundary. |
| 5.1 | `specprune_vla_static112` | 8 | 90.0% (180) | -1.5 | 20.290 | 1.139× | 14.63 | 89.60% (1792) | Static deletion112 gives180/200 and the fastest tested eligible timing; deletion128 gives174/200. Static/dynamic pruning with bounded static deletion. Search: accuracy_boundary. |
| 5.1 | `ties_keep575_h48` | 8 | 88.5% (177) | -3.0 | 18.594 | 1.263× | 14.65 | 89.15% (1783) | Keep57.5%/refresh48 gives 177/200. Keep50% at refresh48 falls to 171; refresh96 at keep57.5% falls to 174. Keep65% variants are slower. Select the fastest eligible tested setting, not a claimed global optimum. OpenReview revision core, native K=30/L=16; Dev-selected refresh48 executed controls versus reference H=20. All rank-probe/selection work is charged. Search: accuracy_boundary. |
| 5.1 | `tome` | 8 | 90.5% (181) | -1.0 | 27.692 | 0.828× | 14.73 | 90.15% (1803) | r=4 gives 181/200; r=8/9 gives 178/177 without a reliable speed gain, and r=10 falls to 166. Report the slowdown. Classic ToMe comparison on native paired vision branches; unchanged cognition and DDIM. Search: accuracy_boundary. |
| 5.1 | `visionzip` | 8 | 88.5% (177) | -3.0 | 15.804 | 1.453× | 14.63 | 90.90% (1818) | Keep50% gives 177/200 and is faster than keep43.75%; a same-cohort diagnostic confirms the nonmonotonic latency. Keep40.625% falls to 174. VisionZip dominant/contextual token core; fewer retained tokens need not imply faster kernels. Search: accuracy_boundary. |
| 5.2 | `action_cache_threshold80` | 8 | 89.0% (178) | -2.5 | 19.354 | 1.175× | 14.75 | 88.75% (1775) | Two-tail-step threshold0.8 gives 178/200. Clean zero-step thresholds0.8/0.85 give 163/176; threshold0.9 recovers 179 but is slower. Use the same frozen schema-2 bank. Historical schema-1 positive-tail boundaries (identical keys/states) are 176/200 at threshold0.75 and 171/200 at one step; these are not relabeled as schema-2 runs. Native DDIM refinement/clean-endpoint transfer; 1412 pairs from 77/80 independent episodes, no online population. Search: accuracy_boundary. |
| 5.2 | `actuation_slack_refresh_kv_keep50` | 8 | 89.5% (179) | -2.0 | 46.395 | 0.499× | 14.74 | 89.70% (1794) | asr_kv_keep50 gives 179/200, fastest valid timing among tested configurations meeting >= 177/200. Measured below-target boundary: asr_kv_keep10=37/200, asr_kv_keep25=126/200, asr_kv_keep375=176/200. Retain the full-cost slowdown; auxiliary work is not free actuation overlap. ASR immutable dense-base KV reuse with unconditional same-frame full auxiliary refresh; no composition. Search: accuracy_boundary. |
| 5.2 | `bac_updates4` | 8 | 88.5% (177) | -3.0 | 21.983 | 1.061× | 14.73 | 87.70% (1754) | Four ACS updates gives177/200. Stronger three/two updates give169/152; six gives175 and is slower. Preserve BUA and global schedule allocation. ACS plus BUA; native CFG/DDIM residuals and frozen training-only statistics. Search: accuracy_boundary. |
| 5.2 | `c3ache_conditioned_residual_prefix6` | 8 | 91.5% (183) | +0.0 | 22.971 | 1.012× | 14.73 | 90.90% (1818) | Prefix4 is not faster than prefix6 and lowers success; caching four steps falls to162/200. Retain183/200 with no reliable speedup claim. Cross-chunk residual core; two cached steps, refresh interval two; not full Fast-WAM. Search: accuracy_boundary. |
| 5.2 | `dp_cache_interval6` | 8 | 91.0% (182) | -0.5 | 21.215 | 1.110× | 14.73 | 90.05% (1801) | Interval6 fills the profiled middle window2..7 and gives182/200 with1.110x. Extending to2..8 or1..8 gives134/200 or130/200. Keep all native DDIM transitions and both CFG branches. Native denoiser output reuse only; no VAE/compiler pipeline. Search: accuracy_boundary. |
| 5.2 | `efficient_vla_diffusion_cache_interval4` | 8 | 93.0% (186) | +1.5 | 21.208 | 1.093× | 14.73 | 89.25% (1785) | Interval4 gives186/200; interval5 falls to165/200. Choose the eligible stronger cache with its own paired timing. Within-sampling DiT feature cache; native sampler unchanged. Search: accuracy_boundary. |
| 5.2 | `evo_budget168` | 8 | 89.5% (179) | -2.0 | 22.656 | 1.030× | 14.73 | 87.65% (1753) | Measured candidates: evo_budget168=179/200, evo=165/200, evo_budget72=146/200. Choose the fastest eligible candidate. Layout search was frozen before Dev:640 independent episodes,24 candidates; none met the independent confirmation target. No convergence or global optimality claim. Globally budgeted residual cache with frozen independent rollout-searched layouts. Search: accuracy_boundary. |
| 5.2 | `teacache_threshold05` | 8 | 92.0% (184) | +0.5 | 21.455 | 1.093× | 14.73 | 90.95% (1819) | Threshold0.5 gives 184/200. Threshold0.75 gives 181 with under 2% speed improvement; 0.875 and 1.0 fall to 172/173. Classic diffusion-cache comparison; frozen independent polynomial calibration and full native sampler cost. Search: accuracy_boundary. |
| 5.2 | `temporal_redundancy_visual_update75` | 8 | 90.0% (180) | -1.5 | 23.954 | 0.970× | 14.85 | 90.30% (1806) | Update75% gives180/200; update62.5% falls to174/200. No tested reliable acceleration at the accuracy target. Visual attention core; trained action branch excluded. Search: accuracy_boundary. |
| 5.2 | `v_aefusion_s1` | 8 | 91.5% (183) | +0.0 | 24.227 | 0.973× | 14.74 | 90.55% (1811) | S1 gives183/200 and the fastest full-cost timing among all five active overlap windows; S5=177 and S9=14. All measured settings are slower than native. Stale-cognition early native head steps overlapped on a CUDA stream, mandatory fresh tail; no assumed actuation slack. Search: latency_plateau. |
| 5.2 | `vla_cache_budget256_protect75` | 8 | 89.5% (179) | -2.0 | 18.407 | 1.272× | 14.75 | 88.05% (1761) | Budget256 with75 protected tokens gives179/200 and1.272x;50 protected tokens falls to161/200. Preserve task protection, complete entropy statistics, and native KV visibility. Query-row reuse with retained visual KV. Search: accuracy_boundary. |
| 5.2 | `vla_infoentropy_protect140` | 8 | 88.5% (177) | -3.0 | 21.170 | 1.092× | 14.75 | 86.40% (1728) | Static budget 150 with 56/84 protection quotas gives 177/200, the fastest measured eligible setting. Default 100-token protection at budgets 100/150/224 gives 170/173/135. Budget 75 gives 184 with lower paired speedup; retain it only as search evidence. The selected result is exactly at the 3pp Dev loss boundary. Paper-based 140-quota protection variant; complete statistics, causal full-KV reuse, native SDPA, unchanged DDIM and H8. Search: accuracy_boundary. |
| 5.3 | `eco_vla_keep60` | 8 | 92.0% (184) | +0.5 | 20.089 | 1.162× | 15.77 | 89.90% (1798) | eco_keep60 gives 184/200, fastest valid timing among tested configurations meeting >= 177/200. Measured below-target boundary: eco_keep50=167/200, eco_keep55=175/200. EcoVLA dynamic FFN-channel and whole-head core; declared trigger/grouping conventions and frozen training prior. Search: accuracy_boundary. |
| 5.3 | `sp_vla_z025_min_actions3` | 8 | 89.0% (178) | -2.5 | 17.825 | 1.292× | 14.73 | 90.95% (1819) | Minimum three native actions in the unchanged six-command history, with pruning Z threshold 0.25, gives 178/200. Requiring only two native actions gives 176/200. Keep pruning and scheduling together. Pruning plus executed-action scheduling. Search: accuracy_boundary. |
| 5.4 | `deis_steps4_log_snr` | 8 | 90.5% (181) | -1.0 | 19.396 | 1.210× | 14.73 | 87.45% (1749) | Four log-SNR evaluations give 181/200 and 1.2104x. Three give 174/200 below 177; five gives 184/200 but is slower. Preserve active high-order updates and the native finite-noise start. Native CogACT log-rho high-order sampler with stable clean-endpoint limit; no training. Search: accuracy_boundary. |
| 5.4 | `dpm_solver_pp_steps5_log_snr` | 8 | 93.0% (186) | +1.5 | 19.486 | 1.201× | 14.73 | 90.50% (1810) | Five log-SNR evaluations give 186/200. Four give 178/200 and only approximately 0.7% higher paired speedup; three fall to 143/200. Prefer the higher-success near-tied five-step configuration. Pinned high-order sampler with native finite-noise initialization, CFG and H8; not image-paper score reproduction. Search: accuracy_boundary. |
| 5.4 | `falcon` | 8 | 91.5% (183) | +0.0 | 23.523 | 0.978× | 14.73 | 88.55% (1771) | Distance0.01/noise-floor10 gives 183/200 but no acceleration. Distance0.0125/0.015 gives 168/156; floor0 at distance0.01 gives 174. Historical latent selection and native partial-DDIM denoising core; full-cost slowdown retained, not a claim about published checkpoint performance. Search: accuracy_boundary. |
| 5.4 | `keystone_k3` | 8 | 92.0% (184) | +0.5 | 23.715 | 0.996× | 14.73 | 91.00% (1820) | K3 gives184/200 and is fastest among measured active candidate counts; K4 gives178/200, while K8/K16 add latency. No measured end-to-end acceleration. Batched native-DDIM candidates and guarded geometric selection, not a trained sampler. Search: mechanism_limit. |
| 5.4 | `probeflow_eps0015` | 8 | 90.5% (181) | -1.0 | 21.510 | 1.083× | 14.73 | 91.20% (1824) | Measured candidates: probeflow=182/200, probeflow_eps0015=181/200, probeflow_eps03=176/200, probeflow_eps01=167/200. Choose the fastest eligible candidate. Adaptive lookahead core through the tested native DDIM coordinate bridge; not FM-trained. Search: accuracy_boundary. |
| 5.4 | `unipc_steps5_log_snr` | 8 | 91.5% (183) | +0.0 | 20.275 | 1.162× | 14.73 | 90.50% (1810) | Five log-SNR evaluations give 183/200; four give 171/200. Uniform five steps and its order-three variant give 168/200; six uniform steps are slower. Native CogACT predictor-corrector sampler; unchanged checkpoint, CFG and H8. Search: accuracy_boundary. |
| 5.5 | `a3_official_k4_steps5` | adaptive ≤16 | 89.5% (179) | -2.0 | 15.852 | 1.479× | 14.73 | 89.60% (1792) | K4 with five shared draft/verifier DDIM steps gives 179/200 (-2 pp) and 1.4791x, fastest tested eligible setting. Four steps gives 176/200; three/two give 165/105. Six steps gives 175/200; smaller K, relaxed tolerance and verification blocking do not recover the four-step target. Both verification branches stay active. Near-full H16 execution; no fixed-H16 control claim. A3 official-code version; GR00T global confidence and dual verification through native conditional DDIM; excludes paper clustering/medoid, strict-only prefix and W4A4 verifier Search: accuracy_boundary. |
| 5.5 | `aac_k4_m10` | adaptive ≤16 | 89.5% (179) | -2.0 | 14.073 | 1.664× | 14.73 | 90.20% (1804) | K4 with movement threshold 10 gives 179/200 and 37 replay queries. Threshold 8 gives 181/200 with 44 queries; threshold 12 gives 176/200, below the 177/200 target. Choose the faster eligible midpoint with all candidate and selector costs. Shared-cognition stochastic candidates, Gaussian/Bernoulli entropy and movement floor; all candidate and selector costs included. Entropy determines two measured query horizons at the selected setting. Search: accuracy_boundary. |
| 5.5 | `autohorizon_zero_plateau` | adaptive ≤16 | 91.0% (182) | -0.5 | 13.632 | 1.716× | 14.73 | 90.75% (1815) | Hold threshold 1e-6 gives 182/200. Hold 0.03 and 0.01 both give 181/200; all three give 36 replay queries and approximately 1.71x speed. The near-zero positive setting and 0.01 have exactly the same replay query/horizon sequence. Author bidirectional attention selector with native DiT action attention; unchanged predictions and adaptive execution. Search: latency_plateau. |
| 5.5 | `dvac_alpha16` | adaptive ≤16 | 89.5% (179) | -2.0 | 13.080 | 1.795× | 14.73 | 90.85% (1817) | Alpha 16 gives 179/200 and the fastest tested eligible execution. Its 35 replay queries reach the minimum allowed by first-query H8 and subsequent native H16 capacity. Native DDIM clean-estimate variance and causal rolling threshold; real adaptive prefixes, native H8 comparator. Search: mechanism_limit. |
| 5.5 | `pace_p95` | adaptive ≤16 | 90.0% (180) | -1.5 | 12.062 | 1.946× | 14.73 | 89.95% (1799) | Percentile 95 gives 180/200 and 32 replay queries, the minimum for four fixed prefixes under native H16. Only one Dev episode shortens execution at this aggressive setting. Demonstration-calibrated smoothed Cartesian-speed core; excludes joint-space and bimanual variants. Near-full-chunk operation; no measured comparison against a fixed-H16 control. Search: mechanism_limit. |
| 5.5 | `vote_ensemble_h8` | 8 | 87.0% (174) | -4.5 | 23.720 | 0.996× | 14.73 | 89.35% (1787) | H8 thresholds0.25/0.50/0.75 give174/174/172 of200, all below177; retain reference0.50 as an explicit negative two-vote adaptation. H8 two-vote overlapping-chunk ensemble core; no single-ACT architecture or extra queries. Search: no_feasible_configuration. |
| 6.1 | `awq_fp16_kernel` | 8 | 91.0% (182) | -0.5 | 17.005 | 1.356× | 5.90 | 90.20% (1804) | FP16 is the faster eligible option among both measured supported kernel precisions. Explicit FP16 cast boundary; training-only calibration. Search: mechanism_limit. |

## openvla_oft

| Section | Preset | Execution H | Dev SR (n/200) | Δ pp | ms/control | Speedup | Peak GiB | Final SR (n/2000) | Selection / scope |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Reference | `baseline` | 8 | 99.0% (198) | +0.0 | 24.793 | 1.000× | 14.98 | 96.60% (1932) | Native H8 reference. Native model. Search: native_reference. |
| 3.3 | `actquant` | 8 | 99.0% (198) | +0.0 | 23.451 | 1.056× | — | 97.15% (1943) | Projection budgets 4.5/3.5/2.5 give 198/195/185 of 200. The first two latencies tie at 23.451/23.483 ms/control; 2.5 is slower at 26.044. Select 4.5 for higher Dev success, not the larger ratio from a slower native pair. Author GGUF runtime with training-only HSIC/AMF, 200 balanced H8 chunks; projection-bit budget only, internal Q8_1 activations. Cross-runtime comparison, not quantization-only speedup; Torch GPU memory unavailable. Search: latency_plateau. |
| 3.3 | `qvla_b15p5` | 8 | 99.5% (199) | +0.5 | 30.717 | 0.805× | 14.48 | 96.35% (1927) | Channel budgets 4/8/12 give 0/0/3 of 200; 15.5 recovers 199/200. Valid 12/15.5 timings are both slower than native. Retain the active accuracy-recovered 15.5 allocation. Official-code input-Hessian proxy, not paper action intervention; 15.5 is channel-average allocation, not full-model bits/weight. Packed tile-local floating GEMM, not integer MMA. Search: accuracy_boundary. |
| 3.3 | `efficient_vla_layer_prune03` | 8 | 96.5% (193) | -2.5 | 22.970 | 1.072× | 13.80 | 93.95% (1879) | layer_prune03 gives 193/200, fastest valid timing among tested configurations meeting >= 192/200. Measured below-target boundary: layer_prune04=181/200, layer_prune08=135/200. EfficientVLA calibrated static non-contiguous layer core, not dynamic routing or the full pipeline. Search: accuracy_boundary. |
| 5.1 | `a_mmr_current_frame_keep4375` | 8 | 98.0% (196) | -1.0 | 22.801 | 1.052× | 14.61 | 95.70% (1914) | Keep43.75% gives 196/200 and improves over keep50%; keep37.5% falls to 191 and keep25% to 181. Independent current-frame A-MMR core only; historical-memory/full-system claims excluded. Search: accuracy_boundary. |
| 5.1 | `actuation_slack_refresh_deletion` | 8 | 97.5% (195) | -1.5 | 44.000 | 0.563× | 14.69 | 95.75% (1915) | Halving retention to5% lowers success without improving full-cost timing; dense auxiliary refresh remains included. Deletion core; not real-robot overlap. Search: latency_plateau. |
| 5.1 | `adp_keep1875` | 8 | 97.5% (195) | -1.5 | 20.140 | 1.260× | 14.68 | 95.85% (1917) | Keep18.75% gives195/200 and the fastest eligible timing; keep12.5% falls to189/200 without further measured speed benefit. Adaptive pruning. Search: accuracy_boundary. |
| 5.1 | `dart_keep1875` | 8 | 97.5% (195) | -1.5 | 16.131 | 1.535× | 14.60 | 94.85% (1897) | Keep18.75% is the fastest tested eligible point (195/200); keep12.5% gives191/200. Pivot-based token selection, later pruning layer. Search: accuracy_boundary. |
| 5.1 | `depth_cache_primary` | 8 | 98.0% (196) | -1.0 | 25.041 | 0.990× | 14.68 | 96.10% (1922) | Ceiling0.90,progressive window2,and minimum merge0.25 all retain196/200 but do not improve full-cost timing. Keep reference-close primary-view core; wrist and ambiguous carrying logic remain excluded. Primary-view depth merging only; unknown wrist/carrying rules excluded. Search: latency_plateau. |
| 5.1 | `divprune` | 8 | 97.0% (194) | -2.0 | 23.605 | 1.039× | 14.62 | 95.90% (1918) | Keep50% remains the only eligible tested point; keep43.75%,37.5%,25% give191,187,188/200. Diversity-based token selection. Search: accuracy_boundary. |
| 5.1 | `efficient_vla_keep96` | 8 | 98.5% (197) | -0.5 | 21.083 | 1.180× | 14.61 | 94.35% (1887) | Keep96/view gives197/200; stronger80 and64 give191/200 and182/200. Preserve all selection components. Declared OFT token-reduction core. Search: accuracy_boundary. |
| 5.1 | `eta_vla_ilsa_keep35` | 8 | 96.5% (193) | -2.5 | 21.026 | 1.180× | 14.61 | 95.60% (1912) | Keep35% gives 193/200; keep27.5% and keep20% both fall to 191. All semantic/diversity selection overhead is included. ETA-VLA ILSA core only; TFM and training-dependent optimization excluded. Search: accuracy_boundary. |
| 5.1 | `fastv_keep3125` | 8 | 96.5% (193) | -2.5 | 17.404 | 1.432× | 14.60 | 93.90% (1878) | Keep31.25% is the fastest tested eligible point (193/200); keep25% falls to183/200. Attention-based visual token pruning. Search: accuracy_boundary. |
| 5.1 | `flash_vla_keep192` | 8 | 98.0% (196) | -1.0 | 26.203 | 0.937× | 14.66 | 96.75% (1935) | Keep176 retains 196/200 but is not faster; retain keep192 and report the full-cost slowdown. ICS core only; action reuse is unconfirmed and excluded. Search: latency_plateau. |
| 5.1 | `safe_pruner_keep1875` | 8 | 96.0% (192) | -3.0 | 16.233 | 1.528× | 14.68 | 95.55% (1911) | Keep18.75% meets192/200; stronger keep12.5% falls to189/200. Task-aware token pruning. Search: accuracy_boundary. |
| 5.1 | `sparsevlm_v2_scale0875` | 8 | 97.0% (194) | -2.0 | 16.859 | 1.481× | 14.60 | 95.70% (1914) | Profile 192 / scale 0.875 gives 194/200 and 1.481x, the fastest measured eligible setting. Scales 0.75/0.625 give 191/190, and profile 128 gives 189. Scales 1.0/1.5 give 195/198 but are slower. Official SparseVLM+ V2.0 priority-head/RoPE-corrected selection; native-grid budget mapping, no recycling. Search: accuracy_boundary. |
| 5.1 | `specprune_vla_dynamic80` | 8 | 97.5% (195) | -1.5 | 18.730 | 1.307× | 14.59 | 93.50% (1870) | Dynamic keep80% gives195/200; keep60% gives190/200. Score-first migration scope remains explicit. Score-before-first-prune variant, not author-default scheduling. Search: accuracy_boundary. |
| 5.1 | `ties_keep3125_h48` | 8 | 96.0% (192) | -3.0 | 19.896 | 1.251× | 14.60 | 94.00% (1880) | Keep31.25%/refresh48 gives 192/200. At refresh48, keep25% falls to 183; at keep31.25%, refresh96 falls to 188. Choose the fastest eligible tested configuration with all prefix-probe cost included. OpenReview revision core; explicitly approved OFT K=192 ranking-window transfer, L=16 and Dev-selected refresh48 executed controls versus reference H=20. Search: accuracy_boundary. |
| 5.1 | `tome` | 8 | 98.5% (197) | -0.5 | 33.607 | 0.734× | 14.98 | 97.00% (1940) | r=4 gives 197/200. r=8/9 offers less than 1% timing improvement; r=10 falls to 176/200. Report the full-cost slowdown. Classic ToMe comparison on native paired vision branches; dense projector bridge remains charged. Search: accuracy_boundary. |
| 5.1 | `visionzip_keep25` | 8 | 97.0% (194) | -2.0 | 16.897 | 1.454× | 14.59 | 94.95% (1899) | Keep25% gives 194/200. Stronger 21.875% and 18.75% fall to 191/189; select the fastest eligible tested budget. VisionZip dominant/contextual token core with paired native patch membership. Search: accuracy_boundary. |
| 5.1 | `vla_pruner_keep125` | 8 | 96.5% (193) | -2.5 | 17.668 | 1.393× | 14.68 | 93.35% (1867) | Keep12.5% gives193/200; stronger9.375% and6.25% give181/200 and171/200. Preserve dual-score selection. Declared progressive visual pruning core. Search: accuracy_boundary. |
| 5.2 | `actuation_slack_refresh_kv_keep175` | 8 | 97.0% (194) | -2.0 | 46.145 | 0.533× | 14.98 | 95.25% (1905) | asr_kv_keep175 gives 194/200, fastest valid timing among tested configurations meeting >= 192/200. Measured below-target boundary: asr_kv_keep10=185/200, asr_kv_keep05=168/200. Retain the full-cost slowdown; auxiliary work is not free actuation overlap. ASR immutable dense-base KV reuse with unconditional same-frame full auxiliary refresh; no composition. Search: accuracy_boundary. |
| 5.2 | `temporal_redundancy_visual` | 8 | 96.0% (192) | -3.0 | 25.941 | 0.947× | 15.19 | 95.30% (1906) | Update50% gives192/200; update25% falls to182/200 without a timing benefit. Report the slowdown. Visual attention core; trained action branch excluded. Search: accuracy_boundary. |
| 5.2 | `vla_cache_budget256_protect50` | 8 | 97.0% (194) | -2.0 | 24.345 | 1.043× | 15.03 | 96.20% (1924) | Budget256 with50 protected tokens gives194/200 and1.043x full-cost speed;25 protected tokens falls to190/200. Task protection and complete entropy statistics remain active. Query-row reuse with retained visual KV. Search: accuracy_boundary. |
| 5.2 | `vla_infoentropy_sdpa192` | 8 | 97.5% (195) | -1.5 | 25.065 | 0.996× | 14.98 | 95.55% (1911) | Native-SDPA budget 192 gives 195/200 and 0.996x, the fastest measured eligible setting. Budgets 224/256 give 188/191; budget 150 gives 196 but is slower. No reliable acceleration is claimed. Paper-based complete entropy protection and full indexed-KV reuse; FP32 layer entropy, compact complete statistics and native SDPA. Search: accuracy_boundary. |
| 5.3 | `eco_vla_keep625` | 8 | 96.5% (193) | -2.5 | 23.660 | 1.037× | 17.04 | 95.50% (1910) | eco_keep625 gives 193/200, fastest valid timing among tested configurations meeting >= 192/200. Measured below-target boundary: eco_keep60=186/200, eco_keep50=190/200. EcoVLA dynamic FFN-channel and whole-head core; declared trigger/grouping conventions and frozen training prior. Search: accuracy_boundary. |
| 5.3 | `sp_vla_z025_min_actions3` | 8 | 97.5% (195) | -1.5 | 20.921 | 1.184× | 14.68 | 95.95% (1919) | Minimum three native actions in the unchanged six-command history, with pruning Z threshold 0.25, gives 195/200. Requiring only two native actions gives 191/200. Keep pruning and scheduling together. Pruning plus executed-action scheduling. Search: accuracy_boundary. |
| 5.5 | `pace_p75` | adaptive ≤8 | 98.5% (197) | -0.5 | 25.420 | 0.969× | 14.98 | 96.60% (1932) | Percentile 75 gives 197/200 and the least overhead among measured eligible presets, but remains slower than native. Its replay uses 57 H8 queries; seven Dev episodes still trigger shorter prefixes. Demonstration-calibrated Cartesian core on native eight-action predictions; no padded or extended action head. Search: mechanism_limit. |
| 6.1 | `awq_fp16_kernel` | 8 | 97.5% (195) | -1.5 | 21.359 | 1.161× | 6.12 | 96.55% (1931) | FP16 is the faster eligible option among both measured supported kernel precisions. Explicit FP16 cast boundary; training-only calibration. Search: mechanism_limit. |

## Useful auxiliary configurations

Secondary evidence only; these do not enter the default final-evaluation queue.

| Backend / preset | Execution H | Dev SR (n/200) | ms/control | Speedup | Reason / scope |
|---|---|---:|---:|---:|---|
| openvla_oft / `awq` | 8 | 99.0% (198) | 25.626 | 0.967× | Retained accuracy/precision alternative; not a second primary. BF16-kernel W4A16. |
| cogact / `awq` | 8 | 91.5% (183) | 19.262 | 1.197× | Retained accuracy/precision alternative; not a second primary. BF16-kernel W4A16. |
| cogact / `actuation_slack_refresh_keep50` | 8 | 91.0% (182) | 45.692 | 0.512× | 182/200 at higher full refresh cost than keep25. Deletion core; full auxiliary cost. |
| cogact / `action_cache_zero_threshold90` | 8 | 89.5% (179) | 21.877 | 1.040× | Verified zero-denoiser-call cache-hit endpoint; 179/200 with less speed than the two-step primary. A distinct useful core endpoint, not a second default final run. Native DDIM refinement/clean-endpoint transfer; 1412 pairs from 77/80 independent episodes, no online population. |

## Not certified for final evaluation

Diagnostic measurements below do NOT certify the unresolved core or authorize final runs.

| Backend / method | Dev SR (n/200) | ms/control | Speedup | Reason |
|---|---:|---:|---:|---|
| cogact / `st_merge` | 81.0% (162) | 19.809 | 1.177× | Eq. 4 consumes source into target, unlike the provisional anchor-survival convention; one-to-many collision/mass rules remain unspecified. |
| openvla_oft / `st_merge` | 98.5% (197) | 23.994 | 1.033× | Eq. 4 consumes source into target, unlike the provisional anchor-survival convention; one-to-many collision/mass rules remain unspecified. |
| openvla_oft / `team_vla` | 83.5% (167) | 19.226 | 1.279× | Joint text/action top-M aggregation, Eq. 5-9 direction and normalization remain unverified; author source has no executable resolution. |
| openvla_oft / `vla_iap` | 94.5% (189) | 24.090 | 1.021× | The semantic encoder/pooling remains unspecified. The post-cutoff IAprune v2 changes core equations, so it cannot silently repair the in-scope v1. |

## Evidence index

Raw measurements and resource locations are ignored local evidence, not public files.
Use the selection manifest and generated readiness JSON to resolve exact records.
Recorded revisions predating history cleanup can be resolved through the
[history map](../scripts/history_commit_map.json); recorded source IDs are not rewritten.

| Backend / preset | Dev experiment | Final experiment | Timing record | Dev / timing commit | Record digest |
|---|---|---|---|---|---|
| cogact / `baseline` | `a293c4ce24a5b840f961549f41769a1f59f1952c990b14d0ebef644ce8bda358` | `9c717076b91643a09d4c8826b56aeb4de1abf54684601c093388a19ce69ba0c0` | `compact-41-4-cogact-1--native-pre` | `e86cbe2edc52d3bc69122a66dc727a8aa1109d93` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `c3be4cb17862b40f8519c7c2a4a52805b268f4e547991a93fae666ee733fb862` |
| cogact / `quantvla` | `d43d502b3eb64d4175b45d554dcae02d2142e2937ec25443361866afe27c8c0b` | `8d6f9b6191e053b5724d923ac24d4e19acac83a1dc96f24ed5ad7000f93173f2` | `quant-p2-optimized-cogact-r2-quantvla` | `df77fd603b4d74da8696a9f4a1c9ea0abbaa3a75` / `39b5443ce96f387694ef77a64982990e85e74df2` | `d074a3a2e46ce6860ee7b8f95cb5f396e2fde2fd8247eeb50eccdf06fa1f66fc` |
| cogact / `efficient_vla_layer_prune04` | `2c4f114dbefc70fd9001735c927d158cfb17bd28fee9d21786a3676ebed679fd` | `76b777e78321e1a8034a8ba887934874b5eb62a6fd5b9121634a2a7a01d7b3eb` | `routing-layers-v1-124-7-cogact--layer_prune04` | `0281aa8f0f59a2ef08380f36ea95362b27748d90` / `0281aa8f0f59a2ef08380f36ea95362b27748d90` | `46a1ecab3fb1b69ae5b614c6cd22c0afc3d625fa338d906ee0c9bcb2a2fcbd21` |
| cogact / `a_mmr_current_frame_keep4375` | `49d963de78d72de16e8800148ab541df1654c905c5d74eaf5e8525d416921764` | `e5453aed86afe9a1df175b7ad92087e0b36b8af8bc4920a5d4fc13e6d2952fde` | `scoped-retained-r3-r1-41-g1-cogact--a_mmr_keep4375` | `f5bd8e8379633253bf85cfa8d19799c7af1565a6` / `bb7a27cc53544214055dcaacb5fd6435396f7641` | `8404057c726543c4c0498f2736342be5bb30043e0a232aa59d68383ebee4ef8d` |
| cogact / `actuation_slack_refresh_keep25` | `1745acc6a866125188a9997d38d3a193bf3f2238f71a72e8522ed6a2933d2115` | `64c0acc7f982a2d55ca87d1e362fe53e35b45f2041f5b1c437e0ec3b95d22f3b` | `compact-41-4-cogact-1--actuation_slack_refresh_keep25` | `1dc47be1a1e7d1cd8623a50e329d3a72e5daa8ef` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `ba45d59001f53064891ce78afc87457c6fcfb42d74aae8f49bd7b9ed91127717` |
| cogact / `adp_keep125` | `c903c16ec47f230039b17243336af1865e56397c01e2dab8090c51685f02efe1` | `937b21f82705e58bec70126a778be37c79bceae82fbb63cbf32af36b1890c06a` | `search-r16-41-7-cogact--adp_keep125` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `523c898b888194f2503aa3d509b855a544a1c3d428dc003e0b64f13d84e0b222` |
| cogact / `dart_keep25` | `4d0507e9f3f98e0b60899e6bb2a488f0ffbf66ec8b09cad2cde88a13c1f7d9c5` | `d281982921c24a8984770b4c3ccc48c830a7badc0fe7931550c94d53a0394a98` | `search-r1-45-5-cogact-0--dart_keep25` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `04571245becc1aaf59e7862c21dfd61687b3e70af4d7def0eda78c69bcccba8e` |
| cogact / `depth_cache_primary_floor75` | `7d96c64920f5c4d64b45d896c858a33e184162a558c5beaa4f093c393720d34a` | `ad97d71a1d88f0f25c52f0dcc2ac8342895e59f92efc05803add1caf15d55e21` | `search-r27-41-7-cogact--depth_cache_primary_floor75` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `c60bc8918786bb101d5da7dfb2270c9f7c94e9e7b3e0bef92a4a6ad38c93c9f7` |
| cogact / `divprune_keep375` | `aa02db4d4648252bb86bbb9ce105d0cf3e3ee94946947a46d744454db64ec2bc` | `aebdcfa59eec03c8054642b055250277b3957151f6e0124dfa155b95d5827b01` | `search-r10-recheck-41-7-cogact--divprune_keep375` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `527d0f3634ed62094523ca80d6aa4479ebcbee75a5fb619640f8f5b6fed76754` |
| cogact / `eta_vla_ilsa_keep35` | `d2517995bb45aa422ce16688e054240a515dc5ca237a95f1f4676d0b642b3f37` | `997011b93b22f70d800827108141ad0d07d4f004741b986d31ce15af47306c74` | `scoped-ilsa-strong-r1-124-g5-cogact--ilsa_keep35` | `aaca6132f0b9328568be5958b5ed86bd08792f43` / `bb7a27cc53544214055dcaacb5fd6435396f7641` | `1dd4aaf126335636c01dcf5fad28ccb2fa225bdb4cbd568742e3d3c694f76049` |
| cogact / `fastv_keep25` | `a581c15da97c796e59b7c3a2ceb059e77d7abc62ff64e4b11386f5ac9ed724bf` | `0058104b920285ffaa1de3a2f045d65d650ed0efed296dab7ea18895e9c762f9` | `search-r1-45-5-cogact-0--fastv_keep25` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `ecef890ff4e4d70a080d005228ee81213473005d2814f362e2a61e29059e1b15` |
| cogact / `flash_vla_keep232` | `78e61c2a5e331e0593f782b04a4ccec96cf5e2a17892ffc603699eabb99d3a44` | `00bcb1f52400214024b3d5b574d2605cc4eee43fd807ba8989f114847e7874cf` | `search-r17-125-5-cogact--flash_vla_keep232` | `96c738e00f6342e85790130ea2e660eec6d30dee` / `faabaa5959b49364a808c7327559087f3cf1c892` | `cc458a77aefc8f5d3149096540bdccdee1d5d239f87d5245b7cd72d9f45a9388` |
| cogact / `safe_pruner` | `bdc7820db41f431415a887ed411fead64ed546d22e056708ae5ef56344353f26` | `bd3fb201da67e23871552d3765a7138f711d18d331aa1581df7bd66e95a34405` | `compact-43-0-cogact-token-retry--safe_pruner` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `2a49223fe7b558d8ca59b6ceab6557f06fc354b9` | `3ba866bc10cd8608a9e9300225f35f6f3a43c3d9c1397d8f0897837bde572a1d` |
| cogact / `sparsevlm_v2_128` | `bcf7adcaf806a3f66f2750eaed0ab7d2b5ba26bd4f9495e9b7f3642c6944e601` | `a31a5aa28f9af91b92d829d922e5310698183d71081bf96ce3a9c9ec896a40bf` | `se2-r1-repaired-cogact--sparsevlm_v2_128` | `7fe9c44403fa693ce426b47517fda2d14bc4d126` / `7fe9c44403fa693ce426b47517fda2d14bc4d126` | `660e85ce6de311f62b7a93c7c89da631d4af9351533df2443a5956039b1234dd` |
| cogact / `specprune_vla_static112` | `d1a060bd5b0c4e31b979a8f9e8d98c7a9ae441766e68e7ca093b06fcbda92f0e` | `7b78beccd5aa1fe3879940daef6aeaa38e932b97581655013c67c87535d2008c` | `search-r1-recheck2-41-7-cogact--specprune_vla_static112` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `b9d53df2dc6d5ca8f0c09c47384cd94f376351308573344ce8c9a53c7fe2029a` |
| cogact / `ties_keep575_h48` | `61427584620a852d356394b0988b2ff5ecb1d4693e6fded4547f29b9d0359e45` | `8a25ecfe2e909a3bcca5d0d428244c9154f5d526e3aafbd0e2463390e0b328c1` | `scoped-ties-cog-boundary-r1-124-g4-cogact--ties_keep575_h48` | `1ef12e2e3439449dd51fca253398412f996f6dd8` / `1ef12e2e3439449dd51fca253398412f996f6dd8` | `efa5cd4493f59c2337d53ad31a2db8376667e9f36a869f8d2fd77a63d3a25698` |
| cogact / `tome` | `8f1aac48a4d142d8ea5c03a9538b91a329c717f15ee4bb3ed002b01317424f42` | `205ed612276d1f2eb6ae21316891aa2ccc41f2f0c643d74eac737d8efa62928d` | `final-r1-45-g3-cogact-tome--tome` | `1d3353bf4c16692eba22ce5d9c64dba80f39a077` / `1d3353bf4c16692eba22ce5d9c64dba80f39a077` | `ae617146e897aef4ac1e10dc96f1253267ba1e7402ae5fadabee632542f208ba` |
| cogact / `visionzip` | `582fa08229544e485c67f97a3035e7ee7f30a16d96bb99063b5a99dae859215d` | `41c3c44b3a92c29db454ab5288706319caf122c056637cbc69f0abca24c4f1ec` | `final-r15-41-g1-cogact-visionzip--visionzip` | `4590ed87b95f450edd1539a7798d9188a9719593` / `4590ed87b95f450edd1539a7798d9188a9719593` | `b35f87305c0c097808abfdda5f54e00b8afdcdc361b2c7550540ead0ff4f3665` |
| cogact / `action_cache_threshold80` | `2a8c58aa70873bd5b38a2b902d32a7b002bd68e2474d53d5d48e5725d8f32c54` | `ee1e2717e49b0d61ca2a76afc0c4e271a9d8936879f7d8f70ab936904cdbd46b` | `scoped-action-zero90-thermal-replacement-r1-40-g1-cogact--action_few2_t080` | `bb7a27cc53544214055dcaacb5fd6435396f7641` / `bb7a27cc53544214055dcaacb5fd6435396f7641` | `f3e7c505bcd341b036a756feb1671c41172ad4e3f2b8de402df7d5cee0c630b5` |
| cogact / `actuation_slack_refresh_kv_keep50` | `a335934e5a3e5f57336b356f9da9fd8d227ab7e9155a6fc2b2f13d17f5a74f10` | `73f0d2ee91f9530709c1371d9f32641f61a31c03cbc6f68e2364c54a9ff8b21d` | `routing-asr-v2-45-6-cogact--asr_kv_keep50` | `86343119fa1e50550eb90a994b9e4a6cebd779b3` / `86343119fa1e50550eb90a994b9e4a6cebd779b3` | `98252b428fb9dc09532504ff45493504345b8a1eab3a4bbe7a0ae9800440dd64` |
| cogact / `bac_updates4` | `ead4b40ef77bafdf2f26a73525b15a847f5459460cbf7dd48a5eceb065cfc8ee` | `0860c0ca3e5787b8ca6d4fec776a511de46ec4c97f09f1ebaa17e23b15eba795` | `bac-v1-41-7-recheck4-cogact--bac_updates4` | `d338c60a1fe7d206157b70cb47227c35c664f2ac` / `d338c60a1fe7d206157b70cb47227c35c664f2ac` | `007e78c7dcac729f7653dccccd38164545312bd8c4192240e6e4ae925548bc0a` |
| cogact / `c3ache_conditioned_residual_prefix6` | `c692772b868aa42a4d774070b8028db7bf93ab031712cb7cb5d7b65d9cd03d6e` | `1574f167b556aa28ec5dedb409f1f88cee6b87488828cab7c49e91c64d472f5b` | `cache-v2-45-6-cogact--c3ache_conditioned_residual_prefix6` | `d3c46d34696eb47f753dba3306526bad5ad461c7` / `d3c46d34696eb47f753dba3306526bad5ad461c7` | `539f67019c803f6333f2f804c123a83bd0f31e373402b7c5fc321eab70d1662a` |
| cogact / `dp_cache_interval6` | `2243cf3b022aafef382570f434f2e1fc298613e6434860a84873d27be7ca3d16` | `c30044201a0e7226d77767cddbc8b5de767e853336d51186e9a66346dc163df6` | `search-r1112-124-6-cogact--dp_cache_interval6` | `faabaa5959b49364a808c7327559087f3cf1c892` / `96c738e00f6342e85790130ea2e660eec6d30dee` | `6cc089a6b63d02a883e2f84613bb972e5b3daceebb692db6142c4778323e59a5` |
| cogact / `efficient_vla_diffusion_cache_interval4` | `d1b43467534ec99b6bf7a6d3eb7523fd184068eb8834ff9d9d5e4a2d6d8d1f31` | `fd563da545d101550fd5714d0a97fd6ee9b63cb0505543cc640292cded69292f` | `search-r1-recheck-41-7-cogact-5--efficient_vla_diffusion_cache_interval4` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `513e6fb4b5cd17be3ac8adabfe3493923cfcd6da35be5c5ba6ce3774e013a3c2` |
| cogact / `evo_budget168` | `d653e21d06192fafc907bcf141442ab7c5e7d0d604c36c323215b48c87412cc2` | `9844be2b332d78d99c201e998476ac19928838c2fbb3912ab0d5daa83220517e` | `evo-v1-124-7-cogact--evo_budget168` | `195d27fa6e65889e964e9a23483d43319b57e50c` / `195d27fa6e65889e964e9a23483d43319b57e50c` | `fae1a7db1d8fb71f002d70731365848c236d3202250687bc7ab2dd56fada4bca` |
| cogact / `teacache_threshold05` | `5ec401498f2f120061fc6f6fd368690e08fe5f54c529d0b72f5dd735094bb659` | `9f24fa40e146f1aa6719c1081e6a492d06acfb9751057e0a98df5dfb32a9c063` | `final-r4-41-g4-cogact-teacache--teacache_strong` | `3a8c368834b4560044c5afa7a6cb0e7fc8974ea6` / `3a8c368834b4560044c5afa7a6cb0e7fc8974ea6` | `842e8526e6e7d3d6c716ec1befa2a16a9bcaede0334ca8244a1c04e34d6f124b` |
| cogact / `temporal_redundancy_visual_update75` | `fd8de3e4c0776b169a8a488af5f09a840a22c597869235a171387ed9ab2adca5` | `13ff11ee93a1e588cdb6d07bc918d729a9fc61a4609047e840669656e33957b5` | `cache-v3-45-6-cogact--temporal_redundancy_visual_update75` | `d3c46d34696eb47f753dba3306526bad5ad461c7` / `d3c46d34696eb47f753dba3306526bad5ad461c7` | `35d15909a0b0e2be29c00576968e94183efe13ea35c330a6d70b828780b89793` |
| cogact / `v_aefusion_s1` | `1195ea8e4181a5a4bf42f16aa8077a9344695797ff52f31f050cab5d4de04539` | `6ab76827219c3e9a2bb553bc61e9e3d86ceb7eb749646a4753f3b3a866ae905e` | `v-aefusion-v1-125-7-drift-recheck-cogact--v_aefusion_s1` | `f21d38df23182cf63c82159c4e1b7834d9259752` / `f21d38df23182cf63c82159c4e1b7834d9259752` | `1cf44684809e41e1c1ade1316d2f5720095b7c47c926f7528191922f92c08c47` |
| cogact / `vla_cache_budget256_protect75` | `9927e2ef608b3b72e97501649002c6a57b707b0566404abae53f571a1422666f` | `41fda8f8a9dff3dbead6f3f87223a4278f287d2c2da33b78d52ffe1ce41f0f16` | `search-r25-125-5-cogact--vla_cache_budget256_protect75` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `9a790606de30f1470c6aa17f60f15bb3253c85fe273d641090a7df9f82b65292` |
| cogact / `vla_infoentropy_protect140` | `85f2b3711700b9ca9428932bba6ae435f7e40798274003e841e06ee2a93a7398` | `fb608c729d5179132dc106af1322764a30a49365d3d533e074fa7cbaa4cbf8c6` | `se2-r1-repaired-cogact--vla_infoentropy_protect140` | `7fe9c44403fa693ce426b47517fda2d14bc4d126` / `7fe9c44403fa693ce426b47517fda2d14bc4d126` | `b0613e72af81bc8fc7f5fa7460ecae582809a042e7f1b950361f2abafc375111` |
| cogact / `eco_vla_keep60` | `9b3df9db8fb40e0e1a2088fc5ec34728f4cef5d25c363dff38c9bc22ecf8a308` | `6f2ad8c5447c5877aec95203e30c92f72a0b631ccf2ad0a41992990c7817d07f` | `routing-eco-dyn1-45-7-cogact--eco_keep60` | `0281aa8f0f59a2ef08380f36ea95362b27748d90` / `a7c6b45a939067ff2dd25b438068333659391ed2` | `4a8fd532058d373edabd54bf8fa874bfca0d9ead3c2433f15df23d7a03cc7794` |
| cogact / `sp_vla_z025_min_actions3` | `e5ff0e4f5f44dd1112561436b4d6c982a84db488c0c2af8cdb0c3228fdc657a8` | `628578f7e9925c496e08e139819740c391e3520e164c3afed794f9c010f86752` | `search-r20-41-7-cogact--sp_vla_z025_history3` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `1f5fb02f8a6417f3a9c787db7838353c79a99c4b2111031977193c1ca41774f9` |
| cogact / `deis_steps4_log_snr` | `dc1e9de85d1f757ecdd3858b4c1afcef07824477522fee1941e070d34b31d4f4` | `a84e756e2aa9ef5c379946e50dc94a8fc44d24cd8c2e39409403568536ea872d` | `action-boundary-v1-41-4--deis_steps4_log_snr` | `63619e6e99cc51608aac453854df7d7532141c89` / `63619e6e99cc51608aac453854df7d7532141c89` | `c23f67a9b7d2e56704b223075b4ad878b59506b849cdd5e9362d14a7655644c6` |
| cogact / `dpm_solver_pp_steps5_log_snr` | `1b547fe83c074247bb74416d6adbcbc23da32485504606d80e450a554027cde5` | `b607dc55ed78a0cbf040aa8c9ca681ee067d4b4d9462ac90b74846fab935d9ea` | `solver-logsnr-v1-124-3--dpm_solver_pp_steps5_log_snr` | `cef999e911ddb656189e19a85f226cca6fc47b63` / `63a662a7bc49f0acf2fb7ed668d832baa5824236` | `acecbcef7427b3d7cf26bdcff09fea191621e3b87074fbbe6e107f07f494d65d` |
| cogact / `falcon` | `a535296a1663098a15937a0e061b1b48e9f6979a489acaffdac7e11b586370ae` | `1543f48ea45d1e10228cd40a008a6eaf70e6ff16da2d76e2b6c8a88803c56ceb` | `final-r24-41-g1-cogact-falcon-boundary--falcon` | `eda9a9066f1801819563ea99642f1233d165f980` / `eda9a9066f1801819563ea99642f1233d165f980` | `b318190f9ee000a271950fd05ba46b970a8a8874d93076336d88fa9c624d7635` |
| cogact / `keystone_k3` | `f13cedd6f644a8494c160c48269b65b6ea280edf87ec9a40223127add688bd2a` | `9871d3adb9f2b83c8295c1010c7a83d68e5ec83c26e0f3ef0684b9484a1eb2b5` | `keystone-v1-125-6-cogact--keystone_k3` | `93f4848e25b37cd2928241719cd8557f4e652bf8` / `93f4848e25b37cd2928241719cd8557f4e652bf8` | `ac0630d0ddc3f1f1ff62e255ed92748eacbf230a7ec28842e29181187b9d7196` |
| cogact / `probeflow_eps0015` | `b71a6562b1dcf2888da0ea30fe58e81609ab848e884364482b68948389d09403` | `dd40c2d109b7aea431fb16b7833343e017a5c55b739ec3259a5561726e4ef909` | `probeflow-eps0015-v1-41-7-cogact--probeflow_eps0015` | `b7db84fde6b0dcbd96eac64c37b43b7376929991` / `b7db84fde6b0dcbd96eac64c37b43b7376929991` | `43f7ec53f3b6aebe8fcf70d6d266fd38511d1a846f32c040c5e8898ef32ba25a` |
| cogact / `unipc_steps5_log_snr` | `cf8068e163cf89f2debf324104e2e7cd79a8499929585b4551622218df296333` | `d5b3de75f33edfe3ed35554e95f9553944a061080fac1b00cab21789bf88a998` | `unipc-grid-v1-41-4--unipc_steps5_log_snr` | `168c84610a7dc64b647ed6ad456bca1064bc3210` / `168c84610a7dc64b647ed6ad456bca1064bc3210` | `fe131c6b183d5dd1d54380816f695af70e586cb9398508d3323c1f76d195b717` |
| cogact / `a3_official_k4_steps5` | `4a6e0c9054ae649beed3fb1d79f361ce7b12b27ee8895d3f42fb91fe93d0950b` | `45f2080ed55fb8d83cbd95875dbb543c549f172a0feaea9336b8a10f459fa4c7` | `a3-official-refine-v1-124-7--a3_official_k4_steps5` | `5a2cf2105904790ab001853f1e63862ca938adc7` / `5a2cf2105904790ab001853f1e63862ca938adc7` | `754fd1158e84ebdb227d23aed139d068233256ec4342415f3d9c7d4b5726074a` |
| cogact / `aac_k4_m10` | `fa975a896114f7549968864960fe908aa4c1fdc1bae29e9f0bc28e0e6a179542` | `6f798c6b0702c8ed4bd7cacc576ef1270106228253e9b137568134e23e3e0ae0` | `aac-midpoint-v1-124-7--aac_k4_m10` | `4eae04f22d67e2067a3dc423f9f88b89ca88ad05` / `4eae04f22d67e2067a3dc423f9f88b89ca88ad05` | `2bdee731805b58425ad78211f209fd951e119680f07db28790f620f2c6f09fed` |
| cogact / `autohorizon_zero_plateau` | `8d7946f564951304d00c6c201e4e29020090c9c8bd637646e4805ef2f478bdbb` | `7ac64d87aa82499786c817cab7e0ffbd3630d6619fb41b5a53f322f4e506d5d0` | `terminal-boundary-v1-124-7--autohorizon_zero_plateau` | `b036649764b2e8ce3606c652809e63346545d76e` / `b036649764b2e8ce3606c652809e63346545d76e` | `4b683d06bee64403aab4b0a284678b65b0c8ddda44b4f8057bd769ce0df4e2d6` |
| cogact / `dvac_alpha16` | `013a316e42b4389e0133d66b66fe8d5171c2ecc6d718893044ba79122c793675` | `cf9f1e15c0804217f2bacfad4fa6edee57cee1c2fd9ea09882142491af0fe8b4` | `action-boundary-v1-41-4--dvac_alpha16` | `63619e6e99cc51608aac453854df7d7532141c89` / `63619e6e99cc51608aac453854df7d7532141c89` | `07cfe833ef99c4b73c4eedb9d951d9176f18bc8ee2a481cd2d94588790d184fe` |
| cogact / `pace_p95` | `1be3bc0431ea8a702c1293340180bfd62cb49d12277955aca6700897aa10140a` | `da2bc190d0a88e881d25b36b3fe7c273e07efc65d2bd69212c0bfb6fe1783eaf` | `action-boundary-v1-41-4--pace_p95` | `63619e6e99cc51608aac453854df7d7532141c89` / `63619e6e99cc51608aac453854df7d7532141c89` | `909a2072c0bb0bd9ed517d3601a868e663ee687488b32a7f5e760b269b67e588` |
| cogact / `vote_ensemble_h8` | `9ebbbb6d52e6b5e4131af9d6bc14a493375a387d02bb298bceedb70680ac660c` | `4e1dce6a092568ef27c705b559b97b5a4ad41b49dfcd794d379ab1561a7c1b97` | `search-vote-124-6-cogact--vote_ensemble_h8` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `8f775c6f832ad620cd65450fa517502a6cd0c29bc684c904f4710b310e2a0baa` |
| cogact / `awq_fp16_kernel` | `be03b017c6a40862b3ebc5afa78a0ce2cf31111d49e8763c1fe5d0650391c117` | `8bba1c63b720f89a924e839d7a2915f9c6c3b5787872b3751f62add28636ac07` | `compact-42-0-cogact-quant--awq_fp16_kernel` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `465838d6af15428eace0f168b48392ce6886137b` | `cb173d2eb42c17ac8ae3398af5f884071e082d7e3ed7db8ac15db2f40bfa4a61` |
| openvla_oft / `baseline` | `83a2952de24bf599cafcbe504ce1b8e0c5f91824411122370deec9846dbaa71e` | `762c07b567785f8bb98fe4c60d0f3edba3e577eed1a17eddc06bdcfb7e7b1143` | `compact-124-7-openvla_oft-quant--native-pre` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `2a49223fe7b558d8ca59b6ceab6557f06fc354b9` | `77575dfbcb36ed385377a886f994a3d21e6cb2eb1de5d620f896aea2f9f493cd` |
| openvla_oft / `actquant` | `c71d808134c35b6e3b6c8d20ed967f21926e6fcd509958df5e7a71baefa822b2` | `a1cdf4d13bcd8eaec04e5e89f30e150a859e3e31fe22e70d19cef2836a67dd88` | `quant-p2-short-actquant-isolated-method` | `39b5443ce96f387694ef77a64982990e85e74df2` / `168a3c3ac01867125a92c3297048023e03aff529` | `fc41fe9219a96cc9fba570f3d6521eb26cf722e03fc927fe7edf2da5af2b4cd3` |
| openvla_oft / `qvla_b15p5` | `cd77a61b32450d514848081c08e1c852693eda7ce5d2abaedf7afe6f115696a6` | `586bbea87b3a831a46ecd5ed08c567ffcf9da7006a22319f1f122687d88b4904` | `quant-p2-short-qvla_b15p5-isolated-method` | `168a3c3ac01867125a92c3297048023e03aff529` / `298178facf2d8fb125507604e14d0de411172981` | `ebb3ce369c9d8716cd48bf035a54d6f30f24c15f2693255fee42208ee5e4d2d1` |
| openvla_oft / `efficient_vla_layer_prune03` | `4d5841c74fa9f501a353b8e29e29afb085d26b3b1af36d17906060b696e5c792` | `0666df6171bafac15ffb2ddd3dc68096cbcb4170da384484da506266f081021e` | `routing-layers-v3-45-5-openvla_oft--layer_prune03` | `0281aa8f0f59a2ef08380f36ea95362b27748d90` / `a7c6b45a939067ff2dd25b438068333659391ed2` | `2de43370c80511d8e1c3f3be88da768559dfa66708f3ef0c17638717831fcde6` |
| openvla_oft / `a_mmr_current_frame_keep4375` | `66a171af08f81a0606d70a2d11896f249e8f7bb72659d192c72d7ebdb2aece2e` | `960f94b9ec9a7a3538b363ee144888ac6e88c50be8001d244cea3a9655be2c04` | `scoped-r1-41-g0-openvla_oft--a_mmr_keep4375` | `f5bd8e8379633253bf85cfa8d19799c7af1565a6` / `f5bd8e8379633253bf85cfa8d19799c7af1565a6` | `467d569453b59d2945b5edf55550b9d16f51b037fd4e43fd6b720d5437f9213f` |
| openvla_oft / `actuation_slack_refresh_deletion` | `2e2b75988b34636d5fa39e24e573aa4a159c179d48954e47203a23897921b39c` | `70b5d25d688d38a6df2abe347976d6f813a3e544a610b85264ea1509fe9252d6` | `compact-45-5-openvla_oft-recheck--actuation_slack_refresh_deletion` | `f63c95a6f22a8fd2c1fcdc9fed2441ecb376379d` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `515f46e72adb7e0bd549ce44c0e3716db4d7b3573c3fdf3ad56c826d9df12c2d` |
| openvla_oft / `adp_keep1875` | `58bdb42316badeba12a20a4e3522c93743998b542c79bde2983afc2f661b1e64` | `a43b6c4194e67836fa035e40c5f00e4af94b3038a8813f50a005d10c5c1d9734` | `search-recovery-125-6-openvla_oft-0--adp_keep1875` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `ced38686f07822d35bfb5812d002d8793d6a0dd20e0f4559297f674b220b4990` |
| openvla_oft / `dart_keep1875` | `66630bb5525cf103faf07c776372466697caa7b3d157a532be1923dda958a0ab` | `8389262478d0f8f76c58bb45e0bca32ba5529cca9e370cba8314f6fc8a441fc7` | `search-r78-124-6-openvla_oft--dart_keep1875` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `81bfc1a595ebaba3c4bc9b478f5db1734e7eb0195e082ac346c6d4ba43e3027c` |
| openvla_oft / `depth_cache_primary` | `23ee03e39b8e019f7c9674baabb00b045f8f303a25ea442dd81e7a27516cb331` | `394e621495ac0944653e1452d975d1170ffebe8a04ca19e9811a422c10f10955` | `compact-45-5-openvla_oft-recheck--depth_cache_primary` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `498b1401c4d447a11878f85934fb8fac87172fb74647f47716cf0afae77c7fc8` |
| openvla_oft / `divprune` | `4482ae63177ce1099a9ba5360b83ab01317cb8d5519527dbd8aecd75cb083aea` | `9fdb97ee0fc87e65a54eb3a76d8a72c251937d023efefa7114cef04a6f3dbfae` | `compact-45-6-openvla_oft-1--divprune` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `2985f18b349a71a33698d80f742a62ab0fe8ff8250dc1e1ae0b1008dae27f93c` |
| openvla_oft / `efficient_vla_keep96` | `3695d50ab9056a7d393c7c8d6715f33a53c29ae90306facde37abedda3e5ff38` | `8c8f351576419ffac965cdc2abd5831d3bca7a4bd841b6485fc21ce3aa890a7f` | `search-r10-124-6-openvla_oft--efficient_vla_keep96` | `96c738e00f6342e85790130ea2e660eec6d30dee` / `faabaa5959b49364a808c7327559087f3cf1c892` | `727a146c27ad5151ef3a033b967b953692982dde0bd971acdf52dffd3e9b623b` |
| openvla_oft / `eta_vla_ilsa_keep35` | `3157a825330b27f0933447b028ad50eca148110d1f88720eb0b5a27522a35ac5` | `1dab75fb8bf093d850126b2f64b449f3a35ccf3bb92bcf785cfea9b5340479e6` | `scoped-ilsa-strong-r1-124-g4-openvla_oft--ilsa_keep35` | `aaca6132f0b9328568be5958b5ed86bd08792f43` / `bb7a27cc53544214055dcaacb5fd6435396f7641` | `ca12a7439e91a141d5d6204ac3fa066beb896dea295ce7fc69e104e26ec49ee7` |
| openvla_oft / `fastv_keep3125` | `0070caa61e8205fba10bab5c41cfc52a5789f0750f949eaa45166da55407fc23` | `a541ee8fae263010e280e01c831ffb807bed95792aae7d61427172f14c86bdce` | `search-r456-124-6-openvla_oft-0--fastv_keep3125` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `b69a94d8578b4fab49d9dd0127410f3014f59fbf173d0cebe0a215a359341e6e` |
| openvla_oft / `flash_vla_keep192` | `143755ed1aee192a41d545ed818b591dcfb00fc7d0f0c93802b56b060fb7f84e` | `666cecfededd53b09f155c50195b0cfa74062f69d0ed9f8c51ddc360c2772e2c` | `compact-45-6-openvla_oft-2--flash_vla_keep192` | `1d954ec8f661f0f700810985e2e3d8b8ae5205aa` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `50392782613813f5ba30917361231127bbaf4cda0de1347d6322856fbd2853a6` |
| openvla_oft / `safe_pruner_keep1875` | `c6224353d91f72c020e1c61e427e9cec0964ba782497b05e1fe7db110a06ab07` | `5ee1e92391b9e6be9eb282cfe776f8a80381dd19403f7450f4eeeca140d2fc78` | `search-r23-124-6-openvla_oft-0--safe_pruner_keep1875` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `ac964d850c6460e87b66c5c538681330e9653528fbff8cd2f16d075d81b2ab74` |
| openvla_oft / `sparsevlm_v2_scale0875` | `03d079a435ae20348a6cdde1bc3d3a1a7ec36ea57dada24f84d7c4599e9c84d0` | `52d8df90c2e690881d6c3029e6ece2bdb2ecb772dc2597a90d7a8673c8b01140` | `se2-r2-openvla_oft--sparsevlm_v2_scale0875` | `7fe9c44403fa693ce426b47517fda2d14bc4d126` / `7fe9c44403fa693ce426b47517fda2d14bc4d126` | `71295c2c6d00071f1777170e2401e723e57e72df04e5a429b0ab21fa206ef312` |
| openvla_oft / `specprune_vla_dynamic80` | `3f7f62a0d04cbdcedc73cbab0a908e099270f7205d965fe8d0a7e4558eadb206` | `aa2d76d5d2df88a1a4bbef1c1a154e7d29e60036e4f4d877d1dae5a36fe6eda6` | `search-r1-45-6-openvla_oft-4--specprune_vla_dynamic80` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `ac88a1a6b771d873bd5c35f046d8dffc0b7cd3ef6b2f418f471ffa1ce0d3e4d9` |
| openvla_oft / `ties_keep3125_h48` | `31254c8a85efccc60e16a1fde54c6b3d5fbd8c73f4ec67cc6c360207afd9b1a2` | `1a454c6a93e13d37eb57ddd2e9f935186ddd91fa66067010ffe251b199e1e0d3` | `scoped-ties-oft-refine-r1-124-g4-openvla_oft--ties_keep3125_h48` | `1ef12e2e3439449dd51fca253398412f996f6dd8` / `1ef12e2e3439449dd51fca253398412f996f6dd8` | `3ddccf2c3311bf9559419e7a20b894af5e6a273e62b48cbdc4f160bbaadb4aa5` |
| openvla_oft / `tome` | `c6ea7e725aff569e8d525b3f060537340a0b2d95c15ecd4483ca57b3b42af6fe` | `28edc2514fa81ab2172bc368258753d66a2731c6a635abc678d78a703c138eb9` | `final-r1-45-g1-openvla_oft-tome--tome` | `1d3353bf4c16692eba22ce5d9c64dba80f39a077` / `1d3353bf4c16692eba22ce5d9c64dba80f39a077` | `14729b76f0b280ff9c70e95c9682bd491954ac7e1f02cc22c5ffc3cb33aadf86` |
| openvla_oft / `visionzip_keep25` | `55a7bc9b7524a7e075006d2ffd6d2e87e9650f71bdafc82966fe32fd08e2f822` | `6beda175703ec35c56724574805cc873c3256b9dd7c725e4690ca36d53f8fbdd` | `final-r15-41-g4-openvla_oft-visionzip--visionzip_strong` | `4590ed87b95f450edd1539a7798d9188a9719593` / `4590ed87b95f450edd1539a7798d9188a9719593` | `85c1acf32778114276d202683534ed22223db6a6f53d56ee02d41a20072fd94f` |
| openvla_oft / `vla_pruner_keep125` | `67e27456b57078bbcc5d7a69c66702def13efe6de9858c124b549353013a056f` | `647cc54702dfbbb8e122712e46e132e97a2caa49f7cfee96056e627e8c7d78c1` | `search-r1-45-5-openvla_oft-3--vla_pruner_keep125` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `20eb17c08bc7c89f797f7164cb7f7d1ed5e3a74bea773ada8cef4ee64a21a542` |
| openvla_oft / `actuation_slack_refresh_kv_keep175` | `05a0210e3be30b69e8bfdf6a755eb172425012bd4ed16f6ede86b35f62e27ddf` | `17e68d7c3cdc945df65ab9f325249e86deb953b6fded2741a6bea51b49e62c93` | `routing-asr-v3-45-5-openvla_oft--asr_kv_keep175` | `86343119fa1e50550eb90a994b9e4a6cebd779b3` / `86343119fa1e50550eb90a994b9e4a6cebd779b3` | `99ccf2f4666d0668321c8343bd81c5cf3a04ceac869e12cf5684e6ca157024da` |
| openvla_oft / `temporal_redundancy_visual` | `a0480729b2ee658d04b2ce8dc01bd888cadca71e070a130a2cb0143cf2efd768` | `ba44f6f6290f17f0e5bf2b22a19ca04d88af0fc7c4aadee8f235ec50fae959f1` | `cache-v1-45-7-oft-recheck--temporal_redundancy_visual` | `92b64fc3044235db11f39fe1e0f9c26df4ac1781` / `92b64fc3044235db11f39fe1e0f9c26df4ac1781` | `2ccdeeaaf24de6c4d950289190a72706b8eadb5bdeaea93e0df62f9ec9c71e15` |
| openvla_oft / `vla_cache_budget256_protect50` | `561783bb97b9dc7ae23abc9105ba90b1f11eb6468d675580da273605d05e9a77` | `f189eba40aaaab575b62cabb7152d15cbb96ee9cc1162cbcc76b113bfab4876b` | `search-recovery-125-6-openvla_oft-0--vla_cache_budget256_protect50` | `96c738e00f6342e85790130ea2e660eec6d30dee` / `faabaa5959b49364a808c7327559087f3cf1c892` | `d65232be7392f26af0b9ef1d1d0eadddaee7b903ad5c7da580edec30905e9c0e` |
| openvla_oft / `vla_infoentropy_sdpa192` | `079a455f6d895e8890c4f6a38e259960b313394edd4847c536c38e3278996825` | `c752dbb8d03d125a38f139ce3db414edceb07152a431594646fa63aeb4148597` | `se2-r2-openvla_oft--vla_infoentropy_sdpa192` | `7fe9c44403fa693ce426b47517fda2d14bc4d126` / `7fe9c44403fa693ce426b47517fda2d14bc4d126` | `3132220e8953d6bf7ecea8ad6f7e68781c08277b5f20cda2c0913c99c405f832` |
| openvla_oft / `eco_vla_keep625` | `420ed8a10ec07dafaa45251f29e974609a77d157b8a62e3fd5c2212a0924d73a` | `3008ee6833075c4252ace343437b1327dbd402a16e0f0648c8cc86dc5d45ef70` | `routing-eco-dyn3-45-5-openvla_oft--eco_keep625` | `0281aa8f0f59a2ef08380f36ea95362b27748d90` / `a7c6b45a939067ff2dd25b438068333659391ed2` | `ec654378d1c2e997a3b0fe1c03a643623644a7fb818c14d65535b2dd3055fb02` |
| openvla_oft / `sp_vla_z025_min_actions3` | `1d80b73d202ee632e4220288cfd3bec1fc5f2a62bb5baa5325fe800195a10572` | `9c6fa71c177c94980aa7a6725380ffc730e21a99550860a145d81a7c7ee24f51` | `search-r78-124-6-openvla_oft--sp_vla_z025_history3` | `faabaa5959b49364a808c7327559087f3cf1c892` / `faabaa5959b49364a808c7327559087f3cf1c892` | `bd1c4113caccb6ebfa41678e0b431f392f067adb6bc589dc2db7166a1de11913` |
| openvla_oft / `pace_p75` | `2e34e92ba91fbe225e913e7cd5341acf71609375e3f7ed860e05437c00266be9` | `10d6db50064759363fed6fceeeee6b4aba74d418b0437c5236a5e24d7e00fa0f` | `pace-p75-infra-recheck-v1-124-3-openvla_oft--pace_p75` | `63a662a7bc49f0acf2fb7ed668d832baa5824236` / `63a662a7bc49f0acf2fb7ed668d832baa5824236` | `bcdb72e29eff8cfbc4fa0c130ea1f84dcaa72cf41d4134ac7419a6ff3938321f` |
| openvla_oft / `awq_fp16_kernel` | `566ed69c9ce2db872ba40344f9123dac0ba1db2f05ef3c240c7f156406fe6d66` | `0e7e0f7f0e16adf681d21762a8a8226467693774fbe211d80959e75378d08c43` | `compact-124-7-openvla_oft-quant--awq_fp16_kernel` | `e86cbe2edc52d3bc69122a66dc727a8aa1109d93` / `2a49223fe7b558d8ca59b6ceab6557f06fc354b9` | `b1a7536157e29af90a22cc43bb47a37850b6d3ea9e3d8b499df6089f0311cab1` |
| openvla_oft / `awq` | `5d64703c2acfcfe4a33ed22f76215b16457c1d61cc0be948e134f97124f509cf` | — | `compact-124-7-openvla_oft-quant--awq` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `2a49223fe7b558d8ca59b6ceab6557f06fc354b9` | `8720ab41aa47c710bc8c10a98479089045df80598caa44b3cf79b9645c6b8a8d` |
| cogact / `awq` | `a223cf526c859a8a6071ba21da16d2be890c981c5048f52ab863fda22eace8dc` | — | `compact-42-0-cogact-quant--awq` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `465838d6af15428eace0f168b48392ce6886137b` | `2b53ecaeb6d71307c5630e29c56920594ad7c3f6663d15eb0660d780aebc8f2f` |
| cogact / `actuation_slack_refresh_keep50` | `3e9ab07826e30785e609207a39c4c964aadc37185ff424e01b3643cfb527bd3c` | — | `compact-41-4-cogact-1--actuation_slack_refresh_keep50` | `1dc47be1a1e7d1cd8623a50e329d3a72e5daa8ef` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `c688748fc315d380b58c73462d81222c806f6ed88f9cf027c16ad63c3ca73762` |
| cogact / `action_cache_zero_threshold90` | `cafc1f861659729e267488d2e62539245572e80056ea6a4ae0a1c5bc6f17ad79` | — | `scoped-action-zero90-thermal-replacement-r1-40-g1-cogact--action_zero_t090` | `bb7a27cc53544214055dcaacb5fd6435396f7641` / `bb7a27cc53544214055dcaacb5fd6435396f7641` | `02416cb58dd00bb5b5794ef07c1e376eda396f7a9568bd261d3b807a6f131672` |
| cogact / `st_merge_keep192` | `2715131409b114d95dc8dc1349d3ea4da8b4baa5ab6f0e78127aff0d3dc431eb` | — | `compact-43-2-cogact-3--st_merge_keep192` | `1dc47be1a1e7d1cd8623a50e329d3a72e5daa8ef` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `c67baa5ef382b3aa4fbdce5a312500f136981ba4179eb5163cb4f29398bb594f` |
| openvla_oft / `st_merge_keep192` | `d6e8965949d65ac92a7079a13bbe478bc63fd38ef9d9657f513d8051b75f9a46` | — | `compact-45-5-openvla_oft-recheck--st_merge_keep192` | `1dc47be1a1e7d1cd8623a50e329d3a72e5daa8ef` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `945d70c1affb3d3c243e7eed3a688112f005fbf591f8e87be257572bd16dc62a` |
| openvla_oft / `team_vla_merge160` | `02fc7c9f038ee76b07c85196460993479de227eba74c4343cca09310552f327c` | — | `compact-45-7-openvla_oft-3--team_vla_merge160` | `1dc47be1a1e7d1cd8623a50e329d3a72e5daa8ef` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `5f45fec789ab8a8b09723be9fa5fcb66ebf42d78ee443fa28c56b7be88d7a7bd` |
| openvla_oft / `vla_iap` | `0a6a2c9c71aefa83b1f9ce71ac8d7813d3bb4ed9c4494769d977fdf123756ae0` | — | `compact-45-7-openvla_oft-3--vla_iap` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `c51e783e5187bfc02f9c44dd8c91c8309d4a26383cb18a190dfa3dd0378a862b` |

## Independent action codec reconstruction

### Raw-command reconstruction — H40

Frozen corpus `01a8ae346abe98d43ddc307e2ab3db08c99a0229b58074e7b199ec2cd4500762`; 40-frame windows, one per episode, disjoint calibration/evaluation episodes; 4 evaluated suites.
This is in-domain demonstration reconstruction, not policy success or VLA speedup.
Latency includes representation transforms and CPU/device transfers; native CPU and GPU implementations are labeled separately. Nominal bits are not entropy-coded size.
Rate variants are retained rate–distortion points, not extra methods. Representation transfers and state-conditioned results are separate groups.
Pretraining overlap is not excluded; native block sizes/receptive fields differ.

| Codec / preset | Device | Windows | Raw MSE | Raw MAE | Gripper sign | Tokens/frame | Bits/frame | Encode ms/window | Decode ms/window |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform_binning | AMD EPYC 7532 32-Core Processor | 128 | 0.0002136 | 0.00302 | 1.0000 | 7.000 | 56.000 | 0.095 | 0.038 |
| fast_plus | AMD EPYC 7532 32-Core Processor | 128 | 0.0004145 | 0.01077 | 1.0000 | 1.661 | 18.266 | 0.374 | 0.182 |
| beast / degree=4, gripper_zero_order=False, num_basis=5 | AMD EPYC 7532 32-Core Processor | 128 | 0.0110235 | 0.04374 | 0.9955 | 0.875 | 7.000 | 5.469 | 2.674 |
| beast / degree=4, gripper_zero_order=False, num_basis=10 | AMD EPYC 7532 32-Core Processor | 128 | 0.0043754 | 0.02078 | 0.9990 | 1.750 | 14.000 | 16.852 | 8.359 |
| beast / degree=4, gripper_zero_order=False, num_basis=20 | AMD EPYC 7532 32-Core Processor | 128 | 0.0018109 | 0.00951 | 0.9994 | 3.500 | 28.000 | 40.405 | 20.032 |
| vq_vla | NVIDIA GeForce RTX 3090 | 128 | 0.0026424 | 0.02868 | 0.9996 | 0.800 | 6.400 | 99.680 | 83.789 |
| minivla_rvq | NVIDIA GeForce RTX 3090 | 128 | 0.0141198 | 0.03429 | 0.9795 | 0.875 | 6.125 | 7.831 | 1.107 |
| actioncodec | NVIDIA GeForce RTX 3090 | 128 | 0.0010591 | 0.01865 | 1.0000 | 0.800 | 8.800 | 17.782 | 17.403 |
| actioncodec_rvq / quantizers=1, variant=rvq | NVIDIA GeForce RTX 3090 | 128 | 0.0010912 | 0.01893 | 1.0000 | 0.800 | 8.800 | 20.151 | 17.612 |
| actioncodec_rvq / quantizers=2, variant=rvq | NVIDIA GeForce RTX 3090 | 128 | 0.0005652 | 0.01364 | 1.0000 | 1.600 | 17.600 | 20.027 | 17.452 |
| actioncodec_rvq / quantizers=3, variant=rvq | NVIDIA GeForce RTX 3090 | 128 | 0.0003758 | 0.01127 | 1.0000 | 2.400 | 26.400 | 20.054 | 17.751 |

| Codec run | Measurement commit | Summary SHA256 |
|---|---|---|
| `codec-uniform-binning-h40-v3` | `1d128a49b70e30631ece3fd7aaf15c7458dc7bc1` | `137753166fdecd3aa99e95b0f194fd71f31ad2d5055606ae42566791139a16f1` |
| `codec-fast-plus-h40-v3` | `1d128a49b70e30631ece3fd7aaf15c7458dc7bc1` | `c209ad2e9da2d8e61b5856a43df20c61e012a59ffd09dc06527e0db56817ab32` |
| `codec-beast-basis5-h40-v3` | `1d128a49b70e30631ece3fd7aaf15c7458dc7bc1` | `407ef7fba709921768fa66a6e6763f5bbaa6afeb3234448b6c402f7f0f9798ef` |
| `codec-beast-h40-v3` | `1d128a49b70e30631ece3fd7aaf15c7458dc7bc1` | `145859a697d2cf996f41926482c94ed35e34a9df67e3f4d1211fcf9c099c8760` |
| `codec-beast-basis20-h40-v3` | `1d128a49b70e30631ece3fd7aaf15c7458dc7bc1` | `9c7bd63a48a9c6759638685886f784469859945654cdee60c1448c22a1a068db` |
| `codec-vq-vla-h40-v3` | `1d128a49b70e30631ece3fd7aaf15c7458dc7bc1` | `73add831e40a994cb85446df03bf833392c8a7c92314e9c61afc7b8519c6eef4` |
| `codec-minivla-rvq-h40-v3` | `1d128a49b70e30631ece3fd7aaf15c7458dc7bc1` | `c209a591da39e8f2ba6df4d66cd3bfcfb695c69317a612e98492ea49f7608ef3` |
| `codec-actioncodec-h40-v3` | `1d128a49b70e30631ece3fd7aaf15c7458dc7bc1` | `7d7ce48c63464ad33042b34437c783c2f80236d1f8d2b8c68c5ec73ba7ec53ad` |
| `codec-actioncodec-rvq-q1-h40-v3` | `1d128a49b70e30631ece3fd7aaf15c7458dc7bc1` | `1a36dbc71e5929fcb4cd71e40d972decbc2e7187c5bc4a29a52bc877e8c1f020` |
| `codec-actioncodec-rvq-q2-h40-v3` | `1d128a49b70e30631ece3fd7aaf15c7458dc7bc1` | `0fa8f0a4908b730a888a09ae5ea73075cee762c20dac775a6f802b5111a0396a` |
| `codec-actioncodec-rvq-h40-v3` | `1d128a49b70e30631ece3fd7aaf15c7458dc7bc1` | `91cb50fe5c86214780f3e29b97cfadb6115ebc5d0bac9c3cacbbe0e5484846f3` |

<details>
<summary>Codec device checks (auxiliary, not default rows)</summary>

The small spline workload is faster on native CPU; both devices use four OpenMP threads and the same corpus. Retain the CUDA comparison, not a second primary.

| Run | Device | Raw MSE | Encode / decode ms per window | Summary SHA256 |
|---|---|---:|---:|---|
| `codec-beast-v2` | NVIDIA GeForce RTX 3090 | 0.0043754 | 45.569 / 22.683 | `b12c622bb6c26123c969eaaec6b41187f1ef829ee51cff00eb2d925e2325b575` |
| `codec-beast-basis5-v2` | NVIDIA GeForce RTX 3090 | 0.0110235 | 13.278 / 6.622 | `436c77b75ccdf01c1824675dce9e767066ab145d398e8fc7c23e26a97f2b46b8` |
| `codec-beast-basis20-v2` | NVIDIA GeForce RTX 3090 | 0.0018109 | 110.388 / 54.901 | `0785ce3eaa2bdc7a03aeb3bccb4cacd95ffb4f6da40eb880e63b507ad5bf4ad6` |

</details>

### Raw-command reconstruction — H16

Frozen corpus `1d3d60da1098bbbb8bf7a248b12ccf566e1e97124535cb390fcb504c1223af58`; 16-frame windows, one per episode, disjoint calibration/evaluation episodes; 4 evaluated suites.
This is in-domain demonstration reconstruction, not policy success or VLA speedup.
Latency includes representation transforms and CPU/device transfers; native CPU and GPU implementations are labeled separately. Nominal bits are not entropy-coded size.
Rate variants are retained rate–distortion points, not extra methods. Representation transfers and state-conditioned results are separate groups.
Pretraining overlap is not excluded; native block sizes/receptive fields differ.

| Codec / preset | Device | Windows | Raw MSE | Raw MAE | Gripper sign | Tokens/frame | Bits/frame | Encode ms/window | Decode ms/window |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform_binning | AMD EPYC 7532 32-Core Processor | 128 | 0.0002101 | 0.00311 | 1.0000 | 7.000 | 56.000 | 0.083 | 0.035 |
| fast_plus | AMD EPYC 7532 32-Core Processor | 128 | 0.0003643 | 0.00916 | 1.0000 | 1.562 | 17.188 | 0.255 | 0.137 |
| minivla_rvq | NVIDIA GeForce RTX 3090 | 128 | 0.0115903 | 0.03269 | 0.9839 | 0.875 | 6.125 | 3.222 | 0.525 |
| mint_sdat | NVIDIA GeForce RTX 3090 | 128 | 0.0017260 | 0.02497 | 1.0000 | 0.438 | 3.938 | 5.796 | 6.572 |

| Codec run | Measurement commit | Summary SHA256 |
|---|---|---|
| `codec-uniform-binning-h16-v3` | `1d128a49b70e30631ece3fd7aaf15c7458dc7bc1` | `6e83e6d9de31b2f5a7b4a8de76437f35ba8624845b992afc19b0ab0fff464d9e` |
| `codec-fast-plus-h16-v3` | `1d128a49b70e30631ece3fd7aaf15c7458dc7bc1` | `5b10e18760b0a6a3415999cf3494cb3fa59a827b31deb499c50eaf410fce6c3b` |
| `codec-minivla-rvq-h16-v3` | `1d128a49b70e30631ece3fd7aaf15c7458dc7bc1` | `c98a7af8389d8850237aa8daa7f2e3dea3a18d9f4d82d89cb78276f7d3ae96e6` |
| `codec-mint-sdat-h16-v3` | `1d128a49b70e30631ece3fd7aaf15c7458dc7bc1` | `ba4602d7b1154bc9397c2760b9905f25daba2f7115af251d2b1f27764e5d6288` |

### Quantizer coordinate-transfer diagnostic — H40

Frozen corpus `01a8ae346abe98d43ddc307e2ab3db08c99a0229b58074e7b199ec2cd4500762`; 40-frame windows, one per episode, disjoint calibration/evaluation episodes; 4 evaluated suites.
This is in-domain demonstration reconstruction, not policy success or VLA speedup.
Latency includes representation transforms and CPU/device transfers; native CPU and GPU implementations are labeled separately. Nominal bits are not entropy-coded size.
Rate variants are retained rate–distortion points, not extra methods. Representation transfers and state-conditioned results are separate groups.
Pretraining overlap is not excluded; native block sizes/receptive fields differ.

| Codec / preset | Device | Windows | Raw MSE | Raw MAE | Gripper sign | Tokens/frame | Bits/frame | Encode ms/window | Decode ms/window |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spatialvla | AMD EPYC 7532 32-Core Processor | 128 | 0.0015589 | 0.02027 | 1.0000 | 3.000 | 42.000 | 0.209 | 0.160 |

| Codec run | Measurement commit | Summary SHA256 |
|---|---|---|
| `codec-spatialvla-h40-v3` | `1d128a49b70e30631ece3fd7aaf15c7458dc7bc1` | `8c003a762b336ee609b37ef36e162e1b31a6866644c7730ef4ce7a9c537892ce` |

### State-conditioned representation-transfer diagnostic — H40

Frozen corpus `01a8ae346abe98d43ddc307e2ab3db08c99a0229b58074e7b199ec2cd4500762`; 40-frame windows, one per episode, disjoint calibration/evaluation episodes; 4 evaluated suites.
This is in-domain demonstration reconstruction, not policy success or VLA speedup.
Latency includes representation transforms and CPU/device transfers; native CPU and GPU implementations are labeled separately. Nominal bits are not entropy-coded size.
Rate variants are retained rate–distortion points, not extra methods. Representation transfers and state-conditioned results are separate groups.
Pretraining overlap is not excluded; native block sizes/receptive fields differ.

| Codec / preset | Device | Windows | Raw MSE | Raw MAE | Gripper sign | Tokens/frame | Bits/frame | Encode ms/window | Decode ms/window |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| x_tokenizer | NVIDIA GeForce RTX 3090 | 128 | 0.4746804 | 0.33120 | 0.4721 | 1.000 | 11.000 | 7.914 | 9.133 |

| Codec run | Measurement commit | Summary SHA256 |
|---|---|---|
| `codec-x-tokenizer-h40-v3` | `1d128a49b70e30631ece3fd7aaf15c7458dc7bc1` | `d37d0e0f1397c0c5a4eb8d6e73e1ea76bea35043322d2c9c8b0746d41f02bc64` |

Only compare rows within a group. H16 is a frozen prefix of H40 with the same episode identities, but its calibration statistics and workload differ. Conditional/context-transfer diagnostics are not a codec leaderboard.

## Scoped revalidation — 2026-10

SparseVLM uses pinned official SparseVLM+ V2.0 with native-grid budget mapping.
VLA-InfoEntropy uses native SDPA with complete compact statistics and FP32 layer entropy.
Each revised primary has a new frozen Final-2000 run, not transferred old-version
outcomes. The other 67 primary Final results retain their original identities.
CogACT Final losses versus native are 3.55 pp for SparseVLM and 4.20 pp for
InfoEntropy; the 3 pp Dev target is not a guaranteed full-set accuracy bound.
Local selection evidence: `outputs/sparse-entropy-v2/`; current complete readiness,
primary CSV and scoped Final evidence: `outputs/sparse-entropy-final2000-v2/`.

<!-- supplementary-native-controls -->
## Supplementary native controls — 2026-10-09

These are three owner-fixed native CogACT controls, not new methods or primary-selection candidates.
All use the same joint-suite checkpoint and prediction horizon 16. Dev-200 validates each fixed
configuration without tuning; DDIM4 is retained despite missing the primary-method Dev target.
All three have a fresh, complete Final-2000 run. The existing 71 primary rows and codec evidence
are unchanged; seven compatible retained references were audited, not rerun.

### Fixed execution horizon

| Configuration | Dev n/200 | Final SR (n/2000) | ms/control | Speedup | Replay queries | ms/query |
|---|---:|---:|---:|---:|---:|---:|
| Native DDIM10/H8 | 183 | 90.60% (1812) | 23.399 | 1.000× | 63 | 185.189 |
| Native DDIM10/H16 | 180 | 89.95% (1799) | 11.988 | 1.961× | 32 | 186.617 |
| PACE | 180 | 89.95% (1799) | 12.062 | 1.946× | 32 | 187.690 |
| DVAC | 179 | 90.85% (1817) | 13.080 | 1.795× | 35 | 186.177 |
| AutoHorizon | 182 | 90.75% (1815) | 13.632 | 1.716× | 36 | 188.655 |

PACE and H16 have identical success/failure outcomes on all 2,000 paired episode identities.
Only two episode query counts differ (21,805 versus 21,803 total); one episode step count differs.
Thus this selected PACE configuration is close to fixed-H16 query behavior on this workload;
the approximately 2× gain cannot be credited solely to adaptive selection. This does not assert
identical actions or disprove PACE in general. DVAC/AutoHorizon trade more queries for
observed success gains of 0.90/0.80 pp over fixed H16.

### Native sampling budgets

| Configuration | Dev n/200 | Final SR (n/2000) | ms/control | Speedup |
|---|---:|---:|---:|---:|
| Native DDIM10/H8 | 183 | 90.60% (1812) | 23.399 | 1.000× |
| Native DDIM4/H8 | 176 | 87.40% (1748) | 19.760 | 1.189× |
| DEIS (4 NFE) | 181 | 87.45% (1749) | 19.396 | 1.210× |
| Native DDIM5/H8 | 184 | 89.65% (1793) | 20.165 | 1.170× |
| DPM-Solver++ (5 NFE) | 186 | 90.50% (1810) | 19.486 | 1.201× |
| UniPC (5 NFE) | 183 | 90.50% (1810) | 20.275 | 1.162× |

DEIS4 and native DDIM4 differ by only one successful episode overall. The two 5-NFE solvers
each exceed DDIM5 by 17 episodes (0.85 pp); this is a descriptive operating-point comparison,
not statistical significance or an isolated solver-formula effect.

Actual descending prediction grids are DDIM10 [90,80,70,60,50,40,30,20,10,0],
DDIM4 [75,50,25,0], DDIM5 [80,60,40,20,0], DEIS4 [90,51,9,0], and
DPM-Solver++5/UniPC5 [90,65,24,5,0]. One paired CFG batch counts as one NFE.
Native eta-zero DDIM still consumes discarded random draws: 4/5 for the new reduced-step
controls, versus ten retained by the existing solver ports. Episode seeds are the same,
but subsequent-query noise paths are not forcibly coupled. Grids, starts, RNG consumption
and previously selected solver settings therefore limit causal attribution.

### Measurement scope and provenance

All ten configurations round to 14.73 GiB peak allocated memory. Query counts/means above
refer to the same 499-control replay, not closed-loop totals. Speedups use each row's own
same-GPU native DDIM10/H8 pair, not division by the displayed retained native latency.
Small timing differences are not reliable rankings; there is no repeated-run significance claim.

| preset | Spatial /500 | Object /500 | Goal /500 | Long /500 | Untuned /1800 |
|---|---:|---:|---:|---:|---:|
| `native_h16` | 454 | 479 | 451 | 415 | 1619 (89.94%) |
| `native_ddim4` | 458 | 484 | 438 | 368 | 1572 (87.33%) |
| `native_ddim5` | 464 | 482 | 458 | 389 | 1609 (89.39%) |

| preset | Final experiment ID | Timing record |
|---|---|---|
| `native_h16` | `44339df2aaaa2bd350f62ba8271ce35e6daac16c535b7346b4899f56bbce9d20` | `chapter7-native_h16-v2--native_h16` |
| `native_ddim4` | `9a5d88a1c61591bd93df3a17e62d350df184abe14368f493549ea0484454ef4e` | `chapter7-native_ddim4-v1--native_ddim4` |
| `native_ddim5` | `4d3e31d7d29d67b12e936462ce3a27fe7cbed7b04f932a1baede0aa92e456e3b` | `chapter7-native_ddim5-v1--native_ddim5` |

New measurement commit: `0c9b8ae16fa0a839aab36bb94f743faf0eeb181d`.
Canonical evidence: `outputs/chapter7-controls-v1/`; `results.csv` contains the three
new controls, `comparison.csv` the ten distinct new/reference rows, `evidence.json`
the Dev/Final/timing joins, and `audit.json` the strict acceptance checks.
`reference-audit.json` records old-result compatibility; `horizon-comparison-audit.json`
records the paired PACE/H16 check. Existing reference measurement identities remain unchanged.

CSV SHA256: `d0e47d343bfc86b135f35b8e75231bbad0f0df6bf7fc284ed64d10ea44a8ba3e`.
Comparison CSV SHA256: `2568e3c022f9b2690863da890b30eb93555969ffca32a3b0208c84f14b550690`.
Evidence SHA256: `f12ade5e1b92cf4384d132e6cc0d06ec9fa69c73a445931913eab4a1f33d3e8e`.

Integrity checks cover 600 Dev and 6,000 Final episodes, exact frozen identities/seeds,
four suites per configuration, clean measured sources, and zero episode runtime errors.
Two external-memory collisions occurred before any episode and were rescheduled without
discarding outcomes. One thermally invalid H16 timing cohort was rejected; the first valid
replacement was retained, not the fastest repeat. Its obsolete raw runs/logs were removed;
compact retirement evidence and all valid failures remain.

<!-- supplementary-quantization-runtime -->
## Quantization runtime scope — 2026-10-15

The three new primary rows are real packed execution, not fake-quantization
latencies. Existing AWQ results remain unchanged; separate regression timing found
no unexplained speed discrepancy. Calibration is training-only; Dev-200 selects
one configuration and Final-2000 is run once after freezing it.

- QVLA follows the released input-Hessian proxy, not the paper's action-space
  intervention. The selected 15.5 budget averages output channels, not full-model
  bits/weight; 1,741,383 of 1,854,960 target channels remain 16-bit. Low-budget
  failure and the selected slowdown describe this scoped port, not the paper's
  original checkpoint results.
- QuantVLA stores packed W4 weights and A8 activations, with tile-local floating
  GEMM. It is real packed runtime but not integer-MMA execution. Rotations,
  activation conversion and ATM/OHB overhead are included, including the slowdown.
- ActQuant uses the author's GGUF engine and mixed projection types (achieved
  4.4998 bits/weight), with internal Q8_1 activations. HSIC/AMF uses 200 balanced
  training episodes, one H8 chunk each, with zero optimizer updates. This bounded
  calibration/readout adaptation is disclosed, not original-data-budget parity.
  Vision, projectors and action head remain unquantized. Cross-runtime speedup
  includes engine changes; missing Torch memory cannot be read as zero memory.

### Fixed floating-engine control

Auxiliary only, not another algorithm or an additional Final-2000 campaign.
The control uses BF16 matrix weights, F32 vectors/patch convolution and FP16 KV;
it is not pure-BF16 arithmetic. It retains native inputs, checkpoint and H8.

| Control | Dev SR (n/200) | ms/control | Native-paired speedup | Final SR |
|---|---:|---:|---:|---|
| ActQuant floating GGUF runtime | 97.0% (194) | 27.898 | 0.886× | Not scheduled |

Timing record: `quant-p2-isolated-floating-control-method`; digest: `b880d195c5925126a513d072755682002fef1ebb07dc14bafa3d871b34c69d53`.
Dev experiment: `028f521b21ef337c36171e4577c8101a46b4b0059eb88697588ae00d92a61e76`. The control's independent native pair
passes the same timing-quality checks; it does not replace the native reference.
Its raw and aggregate evidence is indexed by `floating-control-audit.json` in the
ignored quantization phase evidence, separate from the primary selection manifest.
