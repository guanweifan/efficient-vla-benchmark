# Benchmark results

Success rates cover 2,000 LIBERO episodes per configuration (81/81 configurations measured). Hardware: NVIDIA GeForce RTX 3090.
Latency is full policy cost per control step, including preprocessing, method work and postprocessing. Speedup uses each configuration's paired native measurement.
All native references use H8; adaptive methods choose their executed prefix. Peak memory is CUDA allocation. Missing measurements are shown as —.

Configuration names link to runnable presets; [method descriptions](../README.md#supported-methods) specify implemented mechanisms.

## CogACT

| Configuration | Success rate | ms/control | Speedup | Peak GiB | Execution H |
|---|---:|---:|---:|---:|---:|
| [`baseline`](../configs/experiments/cogact/baseline.yaml) | 90.60% (1812) | 23.399 | 1.000× | 14.73 | 8 |
| [`awq`](../configs/experiments/cogact/awq.yaml) | 90.20% (1804) | 17.214 | 1.369× | 5.90 | 8 |
| [`da_mpa`](../configs/experiments/cogact/da_mpa.yaml) | 90.40% (1808) | 17.983 | 1.296× | 5.89 | 8 |
| [`dyq_vla`](../configs/experiments/cogact/dyq_vla.yaml) | 90.10% (1802) | 20.437 | 1.155× | 5.71 | 8 |
| [`holoq`](../configs/experiments/cogact/holoq.yaml) | 89.25% (1785) | 22.091 | 1.061× | 5.71 | 8 |
| [`quantvla`](../configs/experiments/cogact/quantvla.yaml) | 89.60% (1792) | 20.508 | 1.143× | 6.11 | 8 |
| [`efficient_vla_layer`](../configs/experiments/cogact/efficient_vla_layer.yaml) | 90.15% (1803) | 21.797 | 1.071× | 13.19 | 8 |
| [`a_mmr_current_frame`](../configs/experiments/cogact/a_mmr_current_frame.yaml) | 89.25% (1785) | 17.543 | 1.306× | 14.62 | 8 |
| [`actuation_slack_refresh_deletion`](../configs/experiments/cogact/actuation_slack_refresh_deletion.yaml) | 87.90% (1758) | 41.122 | 0.569× | 14.74 | 8 |
| [`adp`](../configs/experiments/cogact/adp.yaml) | 89.50% (1790) | 18.845 | 1.228× | 14.73 | 8 |
| [`dart`](../configs/experiments/cogact/dart.yaml) | 84.75% (1695) | 15.292 | 1.529× | 14.57 | 8 |
| [`depth_cache_primary`](../configs/experiments/cogact/depth_cache_primary.yaml) | 89.90% (1798) | 19.903 | 1.159× | 14.73 | 8 |
| [`divprune`](../configs/experiments/cogact/divprune.yaml) | 91.15% (1823) | 15.937 | 1.449× | 14.61 | 8 |
| [`eta_vla_ilsa`](../configs/experiments/cogact/eta_vla_ilsa.yaml) | 89.80% (1796) | 16.190 | 1.456× | 14.60 | 8 |
| [`fastv`](../configs/experiments/cogact/fastv.yaml) | 89.45% (1789) | 15.104 | 1.548× | 14.57 | 8 |
| [`flash_vla`](../configs/experiments/cogact/flash_vla.yaml) | 87.45% (1749) | 23.173 | 1.017× | 14.71 | 8 |
| [`safe_pruner`](../configs/experiments/cogact/safe_pruner.yaml) | 88.60% (1772) | 15.805 | 1.460× | 14.74 | 8 |
| [`sparsevlm`](../configs/experiments/cogact/sparsevlm.yaml) | 87.05% (1741) | 15.916 | 1.453× | 14.56 | 8 |
| [`specprune_vla`](../configs/experiments/cogact/specprune_vla.yaml) | 89.60% (1792) | 20.290 | 1.139× | 14.63 | 8 |
| [`ties`](../configs/experiments/cogact/ties.yaml) | 89.15% (1783) | 18.594 | 1.263× | 14.65 | 8 |
| [`tome`](../configs/experiments/cogact/tome.yaml) | 90.15% (1803) | 27.692 | 0.828× | 14.73 | 8 |
| [`visionzip`](../configs/experiments/cogact/visionzip.yaml) | 90.90% (1818) | 15.804 | 1.453× | 14.63 | 8 |
| [`action_cache`](../configs/experiments/cogact/action_cache.yaml) | 88.75% (1775) | 19.354 | 1.175× | 14.75 | 8 |
| [`actuation_slack_refresh_kv_reuse`](../configs/experiments/cogact/actuation_slack_refresh_kv_reuse.yaml) | 89.70% (1794) | 46.395 | 0.499× | 14.74 | 8 |
| [`bac`](../configs/experiments/cogact/bac.yaml) | 87.70% (1754) | 21.983 | 1.061× | 14.73 | 8 |
| [`c3ache_conditioned_residual`](../configs/experiments/cogact/c3ache_conditioned_residual.yaml) | 90.90% (1818) | 22.971 | 1.012× | 14.73 | 8 |
| [`dp_cache`](../configs/experiments/cogact/dp_cache.yaml) | 90.05% (1801) | 21.215 | 1.110× | 14.73 | 8 |
| [`efficient_vla_cache`](../configs/experiments/cogact/efficient_vla_cache.yaml) | 89.25% (1785) | 21.208 | 1.093× | 14.73 | 8 |
| [`evo`](../configs/experiments/cogact/evo.yaml) | 87.65% (1753) | 22.656 | 1.030× | 14.73 | 8 |
| [`teacache`](../configs/experiments/cogact/teacache.yaml) | 90.95% (1819) | 21.455 | 1.093× | 14.73 | 8 |
| [`temporal_redundancy_visual`](../configs/experiments/cogact/temporal_redundancy_visual.yaml) | 90.30% (1806) | 23.954 | 0.970× | 14.85 | 8 |
| [`v_aefusion`](../configs/experiments/cogact/v_aefusion.yaml) | 90.55% (1811) | 24.227 | 0.973× | 14.74 | 8 |
| [`vla_cache`](../configs/experiments/cogact/vla_cache.yaml) | 88.05% (1761) | 18.407 | 1.272× | 14.75 | 8 |
| [`vla_infoentropy`](../configs/experiments/cogact/vla_infoentropy.yaml) | 86.40% (1728) | 21.170 | 1.092× | 14.75 | 8 |
| [`eco_vla`](../configs/experiments/cogact/eco_vla.yaml) | 89.90% (1798) | 20.089 | 1.162× | 15.77 | 8 |
| [`sp_vla`](../configs/experiments/cogact/sp_vla.yaml) | 90.95% (1819) | 17.825 | 1.292× | 14.73 | 8 |
| [`deis`](../configs/experiments/cogact/deis.yaml) | 87.45% (1749) | 19.396 | 1.210× | 14.73 | 8 |
| [`dpm_solver_pp`](../configs/experiments/cogact/dpm_solver_pp.yaml) | 90.50% (1810) | 19.486 | 1.201× | 14.73 | 8 |
| [`falcon`](../configs/experiments/cogact/falcon.yaml) | 88.55% (1771) | 23.523 | 0.978× | 14.73 | 8 |
| [`keystone`](../configs/experiments/cogact/keystone.yaml) | 91.00% (1820) | 23.715 | 0.996× | 14.73 | 8 |
| [`probeflow`](../configs/experiments/cogact/probeflow.yaml) | 91.20% (1824) | 21.510 | 1.083× | 14.73 | 8 |
| [`unipc`](../configs/experiments/cogact/unipc.yaml) | 90.50% (1810) | 20.275 | 1.162× | 14.73 | 8 |
| [`a3_official`](../configs/experiments/cogact/a3_official.yaml) | 89.60% (1792) | 15.852 | 1.479× | 14.73 | adaptive ≤16 |
| [`aac`](../configs/experiments/cogact/aac.yaml) | 90.20% (1804) | 14.073 | 1.664× | 14.73 | adaptive ≤16 |
| [`autohorizon`](../configs/experiments/cogact/autohorizon.yaml) | 90.75% (1815) | 13.632 | 1.716× | 14.73 | adaptive ≤16 |
| [`dvac`](../configs/experiments/cogact/dvac.yaml) | 90.85% (1817) | 13.080 | 1.795× | 14.73 | adaptive ≤16 |
| [`pace`](../configs/experiments/cogact/pace.yaml) | 89.95% (1799) | 12.062 | 1.946× | 14.73 | adaptive ≤16 |
| [`vote_ensemble`](../configs/experiments/cogact/vote_ensemble.yaml) | 89.35% (1787) | 23.720 | 0.996× | 14.73 | 8 |
| [`smoothquant`](../configs/experiments/cogact/smoothquant.yaml) | 90.35% (1807) | 18.733 | 1.250× | 8.71 | 8 |
| [`gptq`](../configs/experiments/cogact/gptq.yaml) | 91.10% (1822) | 17.167 | 1.351× | 5.90 | 8 |

## OpenVLA-OFT

| Configuration | Success rate | ms/control | Speedup | Peak GiB | Execution H |
|---|---:|---:|---:|---:|---:|
| [`baseline`](../configs/experiments/openvla_oft/baseline.yaml) | 96.60% (1932) | 24.793 | 1.000× | 14.98 | 8 |
| [`actquant`](../configs/experiments/openvla_oft/actquant.yaml) | 97.10% (1942) | 23.115 | 1.060× | — | 8 |
| [`awq`](../configs/experiments/openvla_oft/awq.yaml) | 96.55% (1931) | 21.321 | 1.150× | 6.12 | 8 |
| [`dyq_vla`](../configs/experiments/openvla_oft/dyq_vla.yaml) | 94.10% (1882) | 20.929 | 1.186× | 5.93 | 8 |
| [`efficient_vla_layer`](../configs/experiments/openvla_oft/efficient_vla_layer.yaml) | 93.95% (1879) | 22.970 | 1.072× | 13.80 | 8 |
| [`a_mmr_current_frame`](../configs/experiments/openvla_oft/a_mmr_current_frame.yaml) | 95.70% (1914) | 22.801 | 1.052× | 14.61 | 8 |
| [`actuation_slack_refresh_deletion`](../configs/experiments/openvla_oft/actuation_slack_refresh_deletion.yaml) | 95.75% (1915) | 44.000 | 0.563× | 14.69 | 8 |
| [`adp`](../configs/experiments/openvla_oft/adp.yaml) | 95.85% (1917) | 20.140 | 1.260× | 14.68 | 8 |
| [`dart`](../configs/experiments/openvla_oft/dart.yaml) | 94.85% (1897) | 16.131 | 1.535× | 14.60 | 8 |
| [`depth_cache_primary`](../configs/experiments/openvla_oft/depth_cache_primary.yaml) | 96.10% (1922) | 25.041 | 0.990× | 14.68 | 8 |
| [`divprune`](../configs/experiments/openvla_oft/divprune.yaml) | 95.90% (1918) | 23.605 | 1.039× | 14.62 | 8 |
| [`efficient_vla_token`](../configs/experiments/openvla_oft/efficient_vla_token.yaml) | 94.35% (1887) | 21.083 | 1.180× | 14.61 | 8 |
| [`eta_vla_ilsa`](../configs/experiments/openvla_oft/eta_vla_ilsa.yaml) | 95.60% (1912) | 21.026 | 1.180× | 14.61 | 8 |
| [`fastv`](../configs/experiments/openvla_oft/fastv.yaml) | 93.90% (1878) | 17.404 | 1.432× | 14.60 | 8 |
| [`flash_vla`](../configs/experiments/openvla_oft/flash_vla.yaml) | 96.75% (1935) | 26.203 | 0.937× | 14.66 | 8 |
| [`safe_pruner`](../configs/experiments/openvla_oft/safe_pruner.yaml) | 95.55% (1911) | 16.233 | 1.528× | 14.68 | 8 |
| [`sparsevlm`](../configs/experiments/openvla_oft/sparsevlm.yaml) | 95.70% (1914) | 16.859 | 1.481× | 14.60 | 8 |
| [`specprune_vla`](../configs/experiments/openvla_oft/specprune_vla.yaml) | 93.50% (1870) | 18.730 | 1.307× | 14.59 | 8 |
| [`ties`](../configs/experiments/openvla_oft/ties.yaml) | 94.00% (1880) | 19.896 | 1.251× | 14.60 | 8 |
| [`tome`](../configs/experiments/openvla_oft/tome.yaml) | 97.00% (1940) | 33.607 | 0.734× | 14.98 | 8 |
| [`visionzip`](../configs/experiments/openvla_oft/visionzip.yaml) | 94.95% (1899) | 16.897 | 1.454× | 14.59 | 8 |
| [`vla_pruner`](../configs/experiments/openvla_oft/vla_pruner.yaml) | 93.35% (1867) | 17.668 | 1.393× | 14.68 | 8 |
| [`actuation_slack_refresh_kv_reuse`](../configs/experiments/openvla_oft/actuation_slack_refresh_kv_reuse.yaml) | 95.25% (1905) | 46.145 | 0.533× | 14.98 | 8 |
| [`temporal_redundancy_visual`](../configs/experiments/openvla_oft/temporal_redundancy_visual.yaml) | 95.30% (1906) | 25.941 | 0.947× | 15.19 | 8 |
| [`vla_cache`](../configs/experiments/openvla_oft/vla_cache.yaml) | 96.20% (1924) | 24.345 | 1.043× | 15.03 | 8 |
| [`vla_infoentropy`](../configs/experiments/openvla_oft/vla_infoentropy.yaml) | 95.55% (1911) | 25.065 | 0.996× | 14.98 | 8 |
| [`eco_vla`](../configs/experiments/openvla_oft/eco_vla.yaml) | 95.50% (1910) | 23.660 | 1.037× | 17.04 | 8 |
| [`sp_vla`](../configs/experiments/openvla_oft/sp_vla.yaml) | 95.95% (1919) | 20.921 | 1.184× | 14.68 | 8 |
| [`pace`](../configs/experiments/openvla_oft/pace.yaml) | 96.60% (1932) | 25.420 | 0.969× | 14.98 | adaptive ≤8 |
| [`smoothquant`](../configs/experiments/openvla_oft/smoothquant.yaml) | 95.70% (1914) | 21.719 | 1.139× | 8.95 | 8 |
| [`gptq`](../configs/experiments/openvla_oft/gptq.yaml) | 96.20% (1924) | 21.082 | 1.168× | 6.12 | 8 |

## Useful auxiliary configurations

These configurations have tuning measurements only.

| Backend / preset | Tuning SR (n/200) | ms/control | Speedup | Scope |
|---|---:|---:|---:|---|
| openvla_oft / `awq` | 99.0% (198) | 25.409 | 0.965× | BF16-kernel W4A16 accuracy/precision alternative, not a second primary. |
| cogact / `awq` | 91.5% (183) | 19.658 | 1.199× | BF16-kernel W4A16 accuracy/precision alternative, not a second primary. |
| cogact / `actuation_slack_refresh_keep50` | 91.0% (182) | 45.692 | 0.512× | Deletion core; full auxiliary cost. |
| cogact / `action_cache_zero_threshold90` | 89.5% (179) | 21.877 | 1.040× | Native DDIM refinement/clean-endpoint transfer; 1412 pairs from 77/80 independent episodes, no online population. |

<details>
<summary>Incomplete implementations</summary>

| Backend / method | Status | Reason |
|---|---|---|
| cogact / `st_merge_keep192` | pending_core | Eq. 4 consumes source into target, unlike the provisional anchor-survival convention; one-to-many collision/mass rules remain unspecified. |
| openvla_oft / `st_merge_keep192` | pending_core | Eq. 4 consumes source into target, unlike the provisional anchor-survival convention; one-to-many collision/mass rules remain unspecified. |
| openvla_oft / `team_vla_merge160` | pending_core | Joint text/action top-M aggregation, Eq. 5-9 direction and normalization remain unverified; author source has no executable resolution. |
| openvla_oft / `vla_iap` | pending_core | The semantic encoder/pooling remains unspecified. The post-cutoff IAprune v2 changes core equations, so it cannot silently repair the in-scope v1. |
| cogact / `eaqvla` | excluded | Not a formal survey experiment: the authors withdrew arXiv:2505.21567v2 on 2025-07-31 citing a paper error. Existing measurements remain diagnostic, not a reason to exclude an unfavorable outcome. |
| cogact / `motion_vq_t1` | excluded | Not a full VQVLA GPU acceleration comparison: this component includes dual codebooks/router but omits the defining spatial/temporal multiplication reuse. Preserve diagnostic evidence; do not attribute its speed to the complete paper. |
| openvla_oft / `eaqvla` | excluded | Not a formal survey experiment: the authors withdrew arXiv:2505.21567v2 on 2025-07-31 citing a paper error. Existing measurements remain diagnostic, not a reason to exclude an unfavorable outcome. |
| openvla_oft / `motion_vq_t1` | excluded | Not a full VQVLA GPU acceleration comparison: this component includes dual codebooks/router but omits the defining spatial/temporal multiplication reuse. Preserve diagnostic evidence; do not attribute its speed to the complete paper. |
| openvla_oft / `qvla` | excluded | Last bounded primary-source review did not resolve action-space estimator versus official input-Hessian proxy, marginal/staged allocation or pruning regularization. Exact proxy parity and full recalibration are insufficient for full-core fidelity; no arbitrary pruning caps or claimed GPU incompatibility. |

</details>

<details>
<summary>Measurement sources</summary>

Selected configurations and timing records are indexed in the [selection manifest](../configs/evaluation/primary_selection.yaml). Recorded source revisions resolve through the [history map](../scripts/history_commit_map.json).

`0281aa8f0f59a2ef08380f36ea95362b27748d90`, `0d6613bc0f5e9adff408233173ebfc13079e373f`, `0f4feebd18efca1c82467b7e6b57234b8ca32688`, `168c84610a7dc64b647ed6ad456bca1064bc3210`, `195d27fa6e65889e964e9a23483d43319b57e50c`, `1d3353bf4c16692eba22ce5d9c64dba80f39a077`, `1d954ec8f661f0f700810985e2e3d8b8ae5205aa`, `1dc47be1a1e7d1cd8623a50e329d3a72e5daa8ef`, `1ef12e2e3439449dd51fca253398412f996f6dd8`, `2a49223fe7b558d8ca59b6ceab6557f06fc354b9`, `3200aa7bc685a43d770a7955e931c7324a3495f1`, `39b5443ce96f387694ef77a64982990e85e74df2`, `3a8c368834b4560044c5afa7a6cb0e7fc8974ea6`, `4590ed87b95f450edd1539a7798d9188a9719593`, `4c9379ae1096ab03c8c6b50d92cafb8d35df1128`, `4eae04f22d67e2067a3dc423f9f88b89ca88ad05`, `5a2cf2105904790ab001853f1e63862ca938adc7`, `63619e6e99cc51608aac453854df7d7532141c89`, `63a662a7bc49f0acf2fb7ed668d832baa5824236`, `7fe9c44403fa693ce426b47517fda2d14bc4d126`, `833cc24b779fb949ee406fdace8d218c39046b1f`, `86343119fa1e50550eb90a994b9e4a6cebd779b3`, `92b64fc3044235db11f39fe1e0f9c26df4ac1781`, `93f4848e25b37cd2928241719cd8557f4e652bf8`, `96c738e00f6342e85790130ea2e660eec6d30dee`, `a7c6b45a939067ff2dd25b438068333659391ed2`, `aaca6132f0b9328568be5958b5ed86bd08792f43`, `b036649764b2e8ce3606c652809e63346545d76e`, `b7db84fde6b0dcbd96eac64c37b43b7376929991`, `bb7a27cc53544214055dcaacb5fd6435396f7641`, `bf60e9ff70b38b989df57982bf1f92ea51fd08c9`, `cef999e911ddb656189e19a85f226cca6fc47b63`, `d338c60a1fe7d206157b70cb47227c35c664f2ac`, `d3c46d34696eb47f753dba3306526bad5ad461c7`, `e3dc770ef6ee3336559ad4850bc2ad9490c0a0ad`, `e6a91178f1a1f6f744878977c34856770ee9f1d9`, `e86cbe2edc52d3bc69122a66dc727a8aa1109d93`, `eda9a9066f1801819563ea99642f1233d165f980`, `ef922100b927df6ca750d33206f9be1e2bd297d9`, `f21d38df23182cf63c82159c4e1b7834d9259752`, `f5bd8e8379633253bf85cfa8d19799c7af1565a6`, `f63c95a6f22a8fd2c1fcdc9fed2441ecb376379d`, `faabaa5959b49364a808c7327559087f3cf1c892`, `fb5836546bc64c066b475b109ce03c7e23068a13`

</details>


## Independent action codec reconstruction

Reconstruction metrics measure action coding, not policy success or VLA speedup.
Timing includes representation transforms and device transfers; bits/frame is nominal payload.
[Codec setup and model assets](../src/efficient_vla_benchmark/action_codecs/README.md).

### Raw-command reconstruction — H40

40-frame windows from 4 LIBERO suites; calibration and evaluation use disjoint episodes.

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

<details>
<summary>Codec device checks (auxiliary, not default rows)</summary>

CPU and GPU checks use the same corpus and four OpenMP threads.

| Run | Device | Raw MSE | Encode / decode ms per window |
|---|---|---:|---:|
| `codec-beast-v2` | NVIDIA GeForce RTX 3090 | 0.0043754 | 45.569 / 22.683 |
| `codec-beast-basis5-v2` | NVIDIA GeForce RTX 3090 | 0.0110235 | 13.278 / 6.622 |
| `codec-beast-basis20-v2` | NVIDIA GeForce RTX 3090 | 0.0018109 | 110.388 / 54.901 |

</details>

### Raw-command reconstruction — H16

16-frame windows from 4 LIBERO suites; calibration and evaluation use disjoint episodes.

| Codec / preset | Device | Windows | Raw MSE | Raw MAE | Gripper sign | Tokens/frame | Bits/frame | Encode ms/window | Decode ms/window |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform_binning | AMD EPYC 7532 32-Core Processor | 128 | 0.0002101 | 0.00311 | 1.0000 | 7.000 | 56.000 | 0.083 | 0.035 |
| fast_plus | AMD EPYC 7532 32-Core Processor | 128 | 0.0003643 | 0.00916 | 1.0000 | 1.562 | 17.188 | 0.255 | 0.137 |
| minivla_rvq | NVIDIA GeForce RTX 3090 | 128 | 0.0115903 | 0.03269 | 0.9839 | 0.875 | 6.125 | 3.222 | 0.525 |
| mint_sdat | NVIDIA GeForce RTX 3090 | 128 | 0.0017260 | 0.02497 | 1.0000 | 0.438 | 3.938 | 5.796 | 6.572 |

### Quantizer coordinate-transfer diagnostic — H40

40-frame windows from 4 LIBERO suites; calibration and evaluation use disjoint episodes.

| Codec / preset | Device | Windows | Raw MSE | Raw MAE | Gripper sign | Tokens/frame | Bits/frame | Encode ms/window | Decode ms/window |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spatialvla | AMD EPYC 7532 32-Core Processor | 128 | 0.0015589 | 0.02027 | 1.0000 | 3.000 | 42.000 | 0.209 | 0.160 |

### State-conditioned representation-transfer diagnostic — H40

40-frame windows from 4 LIBERO suites; calibration and evaluation use disjoint episodes.

| Codec / preset | Device | Windows | Raw MSE | Raw MAE | Gripper sign | Tokens/frame | Bits/frame | Encode ms/window | Decode ms/window |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| x_tokenizer | NVIDIA GeForce RTX 3090 | 128 | 0.4746804 | 0.33120 | 0.4721 | 1.000 | 11.000 | 7.914 | 9.133 |

H16/H40 and representation-transfer groups use separate workloads and are compared within groups.
Measured source: `1d128a49b70e30631ece3fd7aaf15c7458dc7bc1`.

<!-- supplementary-native-controls -->
## Native controls

Fixed CogACT execution and sampling budgets provide references for adaptive methods.
Success rates use 2,000 episodes; replay query counts use the same 499-control workload.

### Fixed execution horizon

| Configuration | Success rate | ms/control | Speedup | Replay queries | ms/query |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Native DDIM10/H8](../configs/experiments/cogact/baseline.yaml) | 90.60% (1812) | 23.399 | 1.000× | 63 | 185.189 |
| [Native DDIM10/H16](../configs/experiments/cogact/native_h16.yaml) | 89.95% (1799) | 11.988 | 1.961× | 32 | 186.617 |
| [PACE](../configs/experiments/cogact/pace.yaml) | 89.95% (1799) | 12.062 | 1.946× | 32 | 187.690 |
| [DVAC](../configs/experiments/cogact/dvac.yaml) | 90.85% (1817) | 13.080 | 1.795× | 35 | 186.177 |
| [AutoHorizon](../configs/experiments/cogact/autohorizon.yaml) | 90.75% (1815) | 13.632 | 1.716× | 36 | 188.655 |

PACE and H16 have identical success/failure outcomes in these runs and nearly identical
query counts. The fixed-H16 control accounts for most of PACE's observed speedup.

### Native sampling budgets

| Configuration | Success rate | ms/control | Speedup |
| --- | ---: | ---: | ---: |
| [Native DDIM10/H8](../configs/experiments/cogact/baseline.yaml) | 90.60% (1812) | 23.399 | 1.000× |
| [Native DDIM4/H8](../configs/experiments/cogact/native_ddim4.yaml) | 87.40% (1748) | 19.760 | 1.189× |
| [DEIS (4 NFE)](../configs/experiments/cogact/deis.yaml) | 87.45% (1749) | 19.396 | 1.210× |
| [Native DDIM5/H8](../configs/experiments/cogact/native_ddim5.yaml) | 89.65% (1793) | 20.165 | 1.170× |
| [DPM-Solver++ (5 NFE)](../configs/experiments/cogact/dpm_solver_pp.yaml) | 90.50% (1810) | 19.486 | 1.201× |
| [UniPC (5 NFE)](../configs/experiments/cogact/unipc.yaml) | 90.50% (1810) | 20.275 | 1.162× |

The reduced-step methods use their selected prediction grids and noise sequences.
These rows compare complete sampler configurations. Measured source for the new native
controls: `0c9b8ae16fa0a839aab36bb94f743faf0eeb181d`.

<!-- supplementary-quantization-runtime -->
## Quantization runtimes

- AWQ/GPTQ use FP16 packed kernels; SmoothQuant uses INT8/INT32 GEMM;
  DyQ-VLA uses its mixed-integer temporal core.
- HoloQ is the official-code variant. DA-MPA implements drift allocation without CSRC;
  QuantVLA retains rotations, ATM and OHB. See each method README for target layers and precision.
- HoloQ/QuantVLA timings include fixed-shape DiT CUDA-graph execution. Initial capture
  took 8.86/4.27 s, respectively, outside warmed inference. ActQuant uses a separate GGUF runtime.

<details>
<summary>Floating-runtime control</summary>

The ActQuant floating GGUF control uses BF16 matrix weights, F32 vectors/patch convolution
and FP16 KV. Its 200-episode diagnostic success rate is 97.0% (194/200); measured cost is
27.898 ms/control and paired speedup is 0.886×. This control has no 2,000-episode result.

</details>
