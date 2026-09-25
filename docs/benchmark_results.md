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

## Fixed Dev-200 validation and historical speed

Five fixed initial states per task; forty tasks, fifty episodes per suite.
Every configuration uses the same episode manifest and policy seeds.
Manifest SHA256: `9563d840b665740fd8febf728e66d1647d5f8d13c6352b85214f835a1abdf580`.
This is a **development/tuning subset**, not the formal 2,000-episode success rate.
The remaining 1,800 episodes were not evaluated by this campaign. No failed
episode is replaced by a successful retry. Complete rows have no recorded Python errors.

Speed columns reuse explicitly selected historical fixed-replay records below,
matched by checkpoint, method parameters, precision and execution cadence.
They are **not simultaneous measurements on these 200 trajectories**. Historical
source/cohort differences remain visible through the timing record links; the
join does not certify identical implementation revisions. Concurrent evaluation
wall time is never used as policy latency. Delta is percentage points against
the same backend's native baseline at matched H; VOTE therefore uses H=1.

### cogact

| Preset / timing record | Section | H | Spatial /50 | Object /50 | Goal /50 | Long /50 | Dev-200 | Delta pp | ms/control | Speedup | Peak GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline [R94](#r94) | 5.1 Token Pruning and Context Compression | 8 | 47 | 49 | 46 | 41 | 183/200 (91.5%) | +0.0 | 23.54 | 1.000× | 14.73 |
| baseline_horizon1 [R13](#r13) | 0. Native references | 1 | 47 | 41 | 39 | 34 | 161/200 (80.5%) | +0.0 | 185.82 | 1.000× | 14.73 |
| actuation_slack_refresh_deletion [R93](#r93) | 5.1 Token Pruning and Context Compression | 8 | 42 | 38 | 35 | 22 | 137/200 (68.5%) | -23.0 | 38.10 | 0.618× | 14.74 |
| adp [R29](#r29) | 5.1 Token Pruning and Context Compression | 8 | 47 | 48 | 46 | 41 | 182/200 (91.0%) | -0.5 | 21.44 | 1.092× | 14.73 |
| awq [R10](#r10) | 6.1 Quantization and Low-Precision Inference | 8 | 49 | 48 | 46 | 40 | 183/200 (91.5%) | +0.0 | 19.81 | 1.192× | 5.90 |
| awq_fp16_kernel [R11](#r11) | 6.1 Quantization and Low-Precision Inference | 8 | 48 | 49 | 47 | 38 | 182/200 (91.0%) | -0.5 | 17.31 | 1.363× | 5.90 |
| dart [R96](#r96) | 5.1 Token Pruning and Context Compression | 8 | 49 | 48 | 45 | 42 | 184/200 (92.0%) | +0.5 | 17.05 | 1.380× | 14.64 |
| depth_cache_primary [R97](#r97) | 5.1 Token Pruning and Context Compression | 8 | 47 | 49 | 45 | 39 | 180/200 (90.0%) | -1.5 | 23.80 | 0.989× | 14.73 |
| divprune [R98](#r98) | 5.1 Token Pruning and Context Compression | 8 | 47 | 47 | 45 | 43 | 182/200 (91.0%) | -0.5 | 17.82 | 1.321× | 14.63 |
| efficient_vla_diffusion_cache [R21](#r21) | 5.2 Caching and Computation Reuse | 8 | 42 | 49 | 41 | 33 | 165/200 (82.5%) | -9.0 | 21.08 | 1.119× | 14.73 |
| fastv [R99](#r99) | 5.1 Token Pruning and Context Compression | 8 | 47 | 48 | 45 | 41 | 181/200 (90.5%) | -1.0 | 16.95 | 1.389× | 14.64 |
| flash_vla [R32](#r32) | 5.1 Token Pruning and Context Compression | 8 | 33 | 32 | 42 | 19 | 126/200 (63.0%) | -28.5 | 20.07 | 1.167× | 14.66 |
| safe_pruner [R22](#r22) | 5.1 Token Pruning and Context Compression | 8 | 44 | 50 | 44 | 39 | 177/200 (88.5%) | -3.0 | 16.04 | 1.470× | 14.74 |
| sp_vla [R15](#r15) | 5.3 Layer Skipping, Early Exit, and Model Routing | 8 | 49 | 48 | 45 | 39 | 181/200 (90.5%) | -1.0 | 20.89 | 1.130× | 14.73 |
| sparsevlm [R100](#r100) | 5.1 Token Pruning and Context Compression | 8 | 46 | 49 | 46 | 44 | 185/200 (92.5%) | +1.0 | 18.77 | 1.254× | 14.64 |
| specprune_vla [R23](#r23) | 5.1 Token Pruning and Context Compression | 8 | 37 | 29 | 30 | 19 | 115/200 (57.5%) | -34.0 | 18.26 | 1.291× | 14.56 |
| st_merge [R24](#r24) | 5.1 Token Pruning and Context Compression | 8 | 9 | 14 | 7 | 4 | 34/200 (17.0%) | -74.5 | 16.59 | 1.421× | 14.63 |
| vla_cache [R16](#r16) | 5.2 Caching and Computation Reuse | 8 | 48 | 48 | 45 | 37 | 178/200 (89.0%) | -2.5 | 20.20 | 1.169× | 14.75 |
| vla_infoentropy [R25](#r25) | 5.2 Caching and Computation Reuse | 8 | 48 | 47 | 46 | 34 | 175/200 (87.5%) | -4.0 | 21.68 | 1.088× | 14.75 |
| vote_ensemble [R18](#r18) | 5.5 Action Chunking and Replanning | 1 | 46 | 46 | 45 | 43 | 180/200 (90.0%) | +9.5 | 186.53 | 0.996× | 14.73 |

### openvla_oft

| Preset / timing record | Section | H | Spatial /50 | Object /50 | Goal /50 | Long /50 | Dev-200 | Delta pp | ms/control | Speedup | Peak GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline [R102](#r102) | 5.1 Token Pruning and Context Compression | 8 | 49 | 50 | 49 | 50 | 198/200 (99.0%) | +0.0 | 24.85 | 1.000× | 14.98 |
| baseline_horizon1 [R36](#r36) | 0. Native references | 1 | 49 | 40 | 38 | 31 | 158/200 (79.0%) | +0.0 | 199.83 | 1.000× | 14.98 |
| actuation_slack_refresh_deletion [R101](#r101) | 5.1 Token Pruning and Context Compression | 8 | 50 | 50 | 49 | 46 | 195/200 (97.5%) | -1.5 | 46.47 | 0.535× | 14.69 |
| adp [R79](#r79) | 5.1 Token Pruning and Context Compression | 8 | 49 | 50 | 50 | 48 | 197/200 (98.5%) | -0.5 | 24.58 | 1.015× | 14.68 |
| awq [R33](#r33) | 6.1 Quantization and Low-Precision Inference | 8 | 49 | 50 | 50 | 49 | 198/200 (99.0%) | +0.0 | 26.00 | 0.965× | 6.12 |
| awq_fp16_kernel [R34](#r34) | 6.1 Quantization and Low-Precision Inference | 8 | 49 | 49 | 49 | 48 | 195/200 (97.5%) | -1.5 | 21.61 | 1.162× | 6.12 |
| dart [R104](#r104) | 5.1 Token Pruning and Context Compression | 8 | 46 | 50 | 48 | 40 | 184/200 (92.0%) | -7.0 | 21.69 | 1.146× | 14.62 |
| depth_cache_primary [R105](#r105) | 5.1 Token Pruning and Context Compression | 8 | 50 | 48 | 49 | 49 | 196/200 (98.0%) | -1.0 | 25.47 | 0.976× | 14.68 |
| divprune [R106](#r106) | 5.1 Token Pruning and Context Compression | 8 | 49 | 50 | 48 | 47 | 194/200 (97.0%) | -2.0 | 24.00 | 1.035× | 14.62 |
| efficient_vla [R82](#r82) | 5.1 Token Pruning and Context Compression | 8 | 50 | 50 | 48 | 48 | 196/200 (98.0%) | -1.0 | 21.59 | 1.156× | 14.63 |
| fastv [R107](#r107) | 5.1 Token Pruning and Context Compression | 8 | 49 | 49 | 50 | 50 | 198/200 (99.0%) | +0.0 | 21.42 | 1.160× | 14.62 |
| flash_vla [R83](#r83) | 5.1 Token Pruning and Context Compression | 8 | 48 | 47 | 45 | 50 | 190/200 (95.0%) | -4.0 | 26.48 | 0.943× | 14.64 |
| safe_pruner [R85](#r85) | 5.1 Token Pruning and Context Compression | 8 | 49 | 50 | 48 | 48 | 195/200 (97.5%) | -1.5 | 18.62 | 1.340× | 14.68 |
| sp_vla [R05](#r05) | 5.3 Layer Skipping, Early Exit, and Model Routing | 8 | 50 | 49 | 48 | 48 | 195/200 (97.5%) | -1.5 | 22.41 | 1.117× | 14.68 |
| sparsevlm [R108](#r108) | 5.1 Token Pruning and Context Compression | 8 | 49 | 49 | 48 | 47 | 193/200 (96.5%) | -2.5 | 23.26 | 1.068× | 14.63 |
| specprune_vla [R86](#r86) | 5.1 Token Pruning and Context Compression | 8 | 36 | 45 | 44 | 33 | 158/200 (79.0%) | -20.0 | 20.42 | 1.222× | 14.60 |
| st_merge [R87](#r87) | 5.1 Token Pruning and Context Compression | 8 | 35 | 41 | 20 | 27 | 123/200 (61.5%) | -37.5 | 21.74 | 1.148× | 14.62 |
| team_vla [R88](#r88) | 5.1 Token Pruning and Context Compression | 8 | 30 | 49 | 7 | 11 | 97/200 (48.5%) | -50.5 | 17.71 | 1.409× | 14.61 |
| vla_cache [R89](#r89) | 5.2 Caching and Computation Reuse | 8 | 50 | 48 | 47 | 48 | 193/200 (96.5%) | -2.5 | 26.16 | 0.954× | 15.03 |
| vla_iap [R90](#r90) | 5.1 Token Pruning and Context Compression | 8 | 50 | 50 | 43 | 46 | 189/200 (94.5%) | -4.5 | 24.51 | 1.018× | 14.69 |
| vla_infoentropy [R91](#r91) | 5.2 Caching and Computation Reuse | 8 | 48 | 50 | 49 | 48 | 195/200 (97.5%) | -1.5 | 27.08 | 0.922× | 15.03 |
| vla_pruner [R92](#r92) | 5.1 Token Pruning and Context Compression | 8 | 49 | 50 | 49 | 47 | 195/200 (97.5%) | -1.5 | 20.75 | 1.203× | 14.68 |
| vote_ensemble [R41](#r41) | 5.5 Action Chunking and Replanning | 1 | 48 | 50 | 46 | 44 | 188/200 (94.0%) | +15.0 | 199.62 | 1.001× | 14.98 |

<details>
<summary>Development configuration identities</summary>

Raw evidence is retained under each campaign's `runs/<backend>--<preset>--sNNN/`.
The aggregate retains every episode outcome, source revision, shard and file hash.
A timing record is an audited historical reference, not a new measurement.

| Configuration | Experiment identity | Runtime source | Shards |
|---|---|---|---:|
| cogact/actuation_slack_refresh_deletion | `b53ec6957ea787d2670b39019271cea2a415f75fce5b3213e5132d201e1c4c50` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/adp | `42449ae56acdde4b772b59d1e06aae9fd967c8b2218fc23decbb18768ba3af66` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/awq | `a223cf526c859a8a6071ba21da16d2be890c981c5048f52ab863fda22eace8dc` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/awq_fp16_kernel | `be03b017c6a40862b3ebc5afa78a0ce2cf31111d49e8763c1fe5d0650391c117` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/baseline | `a293c4ce24a5b840f961549f41769a1f59f1952c990b14d0ebef644ce8bda358` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/baseline_horizon1 | `14ac367a1ff2fcdccd71c123235a5879d950eca0ebed49cf9c4aaf76769d6989` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/dart | `703d93d8a87981631d66e7a6466a8793d795eecd9b2805b7525f54bf65efa389` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/depth_cache_primary | `8f0ed815f5f86fb2088cb19f63a81b6e9c0637fab2ddf178cd6f1bcf9fa202d6` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/divprune | `0450741919be84698dcad7bdb2ac98811edad1574c09cabcf4871983a05ff1ae` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/efficient_vla_diffusion_cache | `2e99d4df59fd76c5961cbf0ce7b5e263c64eb81b2b7f72f60da5893bac5920e3` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/fastv | `27489ccc7f64a43601970748ec169d8a89d2c6547eaa078682debf490749bc32` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/flash_vla | `a3241c13b054520b72fe7d7a8afe665de8ccc3521dd177151f134b84c24b33ba` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/safe_pruner | `bdc7820db41f431415a887ed411fead64ed546d22e056708ae5ef56344353f26` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/sp_vla | `27d4793d914e2be7c1eb929c254b8661581eaefb1a68530fc1c0746d0cd764d9` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/sparsevlm | `19e55b9daed5d94fe5e3c51402cfe296a0a1cb52a519fa5e24cc0025772342e2` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/specprune_vla | `200e2a5546ff87e327bf91696fb9a35b9f0a400baf9f744101f68e74fa752304` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/st_merge | `2a63e1b14d73cbda0afa52b35c299e770af5a3dfa14025eedb31eca4c39edb65` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/vla_cache | `f9612931a47706abef36d9251e82bcf9c0ceeb249fce9e68412d8214f5a29fcb` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/vla_infoentropy | `3e7f5140c644c221a9a02a45731cf1e949650a3564053c9d662f85e37b11cabb` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/vote_ensemble | `f6b608bfced978188d89978e816e2ba29cbd6e9d12a4dc5656cb146d2d4a955f` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/actuation_slack_refresh_deletion | `2e2b75988b34636d5fa39e24e573aa4a159c179d48954e47203a23897921b39c` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/adp | `83f9aeebdff8a240e92f929ab699eaabc470de89ecc663c04837580dce3f412a` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/awq | `5d64703c2acfcfe4a33ed22f76215b16457c1d61cc0be948e134f97124f509cf` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/awq_fp16_kernel | `566ed69c9ce2db872ba40344f9123dac0ba1db2f05ef3c240c7f156406fe6d66` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/baseline | `83a2952de24bf599cafcbe504ce1b8e0c5f91824411122370deec9846dbaa71e` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/baseline_horizon1 | `d1432ffd8638abf10c9ab9ecc527a43f7e12f650a6c7b7bd4e66d09134d065c5` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/dart | `959698fd55bb54c37ba68a4b14bc589d41d41e4cab340d7023a14ce24af65841` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/depth_cache_primary | `23ee03e39b8e019f7c9674baabb00b045f8f303a25ea442dd81e7a27516cb331` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/divprune | `4482ae63177ce1099a9ba5360b83ab01317cb8d5519527dbd8aecd75cb083aea` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/efficient_vla | `fa758c551bfb39b4f8a0a122e5dd9b278d42ceb1728352c67697012322438fb3` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/fastv | `66828e8ed4865b10d7301a447b0fa90beb10e83827a02f0aac1c8551701e69d1` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/flash_vla | `d8ddc05d8a8948df0fe5d2418b2a39e1c0f438f1e183ed4d010b0d5b2c66caad` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/safe_pruner | `21e11d22bc36089d4007bafe5b33d9b790947dbef1e904ab92e73336bf26ffec` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/sp_vla | `c321f89069f8c73c936a1388ee57c00bab318a500a5f81d0ec5b9074a94a23a8` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/sparsevlm | `071e8857510d6e9ed29997e514942f2950f8af42284c5bba9ff50ba0b2c7f6ef` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/specprune_vla | `db98812e4d43c083caa9734b047770a7de7e2af0f2446e06e730cf3e3be7cb34` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/st_merge | `03eda8c029c9b6266595a28fb15e9b26b06271c75879bd824270d01f09393698` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/team_vla | `01da23f467ef2f675afa00620cd10be36f2366225c2ab2bc1d4a69d012ea1d43` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/vla_cache | `05bde38fdbf83dfb4f0928768c7d460b51ef106a49e29eeb6aaf7783da1da17d` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/vla_iap | `0a6a2c9c71aefa83b1f9ce71ac8d7813d3bb4ed9c4494769d977fdf123756ae0` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/vla_infoentropy | `fc132144e01cbf7c6a681a8b9f59e0010c7cb55a4742809365070e9632b55733` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/vla_pruner | `bdc953860094a1e5dc0a255f8c0bc53142570fbd610242920303d1b1a1216310` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/vote_ensemble | `fe1c47c9ecd7373e5744dfcb53a13ba295d4b6543240f380be2724ba3453f792` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |

</details>

## Dev-200 diagnostic round 1: validation and paired fixed-replay speed

Five fixed initial states per task; forty tasks, fifty episodes per suite.
Every configuration uses the same episode manifest and policy seeds.
Manifest SHA256: `9563d840b665740fd8febf728e66d1647d5f8d13c6352b85214f835a1abdf580`.
This is a **development/tuning subset**, not the formal 2,000-episode success rate.
The remaining 1,800 episodes were not evaluated by this campaign. No failed
episode is replaced by a successful retry. Complete rows have no recorded Python errors.

Speed columns reuse explicitly selected historical fixed-replay records below,
matched by checkpoint, method parameters, precision and execution cadence.
They are **not simultaneous measurements on these 200 trajectories**. Historical
source/cohort differences remain visible through the timing record links; the
join does not certify identical implementation revisions. Concurrent evaluation
wall time is never used as policy latency. Delta is percentage points against
the same backend's native baseline at matched H; VOTE therefore uses H=1.

Simulator numerical-instability warnings were logged in: `cogact--st_merge_layer6--s003`.
These are distinct from Python exceptions. Original episode outcomes are retained;
buffered simulator logs support shard-level, not exact episode-level attribution.

### cogact

| Preset / timing record | Section | H | Spatial /50 | Object /50 | Goal /50 | Long /50 | Dev-200 | Delta pp | ms/control | Speedup | Peak GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline [R42](#r42) | 5.1 Token Pruning and Context Compression | 8 | 47 | 49 | 46 | 41 | 183/200 (91.5%) | +0.0 | 23.58 | 1.000× | 14.73 |
| actuation_slack_refresh_deletion [R57](#r57) | 5.1 Token Pruning and Context Compression | 8 | 42 | 38 | 35 | 22 | 137/200 (68.5%) | -23.0 | 43.67 | 0.532× | 14.74 |
| actuation_slack_refresh_keep25 [R58](#r58) | 5.1 Token Pruning and Context Compression | 8 | 48 | 48 | 46 | 36 | 178/200 (89.0%) | -2.5 | 43.29 | 0.537× | 14.74 |
| actuation_slack_refresh_keep50 [R59](#r59) | 5.1 Token Pruning and Context Compression | 8 | 50 | 49 | 44 | 39 | 182/200 (91.0%) | -0.5 | 44.68 | 0.520× | 14.74 |
| efficient_vla_cache2 [R54](#r54) | 5.2 Caching and Computation Reuse | 8 | 48 | 48 | 45 | 36 | 177/200 (88.5%) | -3.0 | 22.12 | 1.066× | 14.73 |
| efficient_vla_cache3 [R55](#r55) | 5.2 Caching and Computation Reuse | 8 | 47 | 48 | 45 | 37 | 177/200 (88.5%) | -3.0 | 21.70 | 1.086× | 14.73 |
| efficient_vla_diffusion_cache [R56](#r56) | 5.2 Caching and Computation Reuse | 8 | 42 | 49 | 41 | 33 | 165/200 (82.5%) | -9.0 | 21.08 | 1.118× | 14.73 |
| flash_vla [R44](#r44) | 5.1 Token Pruning and Context Compression | 8 | 33 | 32 | 42 | 19 | 126/200 (63.0%) | -28.5 | 20.09 | 1.173× | 14.66 |
| flash_vla_keep192 [R45](#r45) | 5.1 Token Pruning and Context Compression | 8 | 44 | 40 | 44 | 26 | 154/200 (77.0%) | -14.5 | 21.47 | 1.098× | 14.69 |
| flash_vla_layer4 [R62](#r62) | 5.1 Token Pruning and Context Compression | 8 | 27 | 40 | 36 | 10 | 113/200 (56.5%) | -35.0 | 20.41 | 1.139× | 14.67 |
| specprune_vla [R46](#r46) | 5.1 Token Pruning and Context Compression | 8 | 37 | 29 | 30 | 19 | 115/200 (57.5%) | -34.0 | 17.93 | 1.315× | 14.56 |
| specprune_vla_score_first [R47](#r47) | 5.1 Token Pruning and Context Compression | 8 | 47 | 48 | 39 | 32 | 166/200 (83.0%) | -8.5 | 18.14 | 1.300× | 14.56 |
| specprune_vla_static128 [R48](#r48) | 5.1 Token Pruning and Context Compression | 8 | 47 | 47 | 45 | 35 | 174/200 (87.0%) | -4.5 | 20.07 | 1.175× | 14.62 |
| st_merge [R49](#r49) | 5.1 Token Pruning and Context Compression | 8 | 9 | 14 | 7 | 4 | 34/200 (17.0%) | -74.5 | 16.54 | 1.425× | 14.63 |
| st_merge_keep192 [R50](#r50) | 5.1 Token Pruning and Context Compression | 8 | 37 | 47 | 46 | 32 | 162/200 (81.0%) | -10.5 | 19.98 | 1.180× | 14.68 |
| st_merge_layer6 [R51](#r51) | 5.1 Token Pruning and Context Compression | 8 | 14 | 13 | 21 | 8 | 56/200 (28.0%) | -63.5 | 16.60 | 1.421× | 14.63 |

### openvla_oft

| Preset / timing record | Section | H | Spatial /50 | Object /50 | Goal /50 | Long /50 | Dev-200 | Delta pp | ms/control | Speedup | Peak GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline [R63](#r63) | 5.1 Token Pruning and Context Compression | 8 | 49 | 50 | 49 | 50 | 198/200 (99.0%) | +0.0 | 24.26 | 1.000× | 14.98 |
| dart [R73](#r73) | 5.1 Token Pruning and Context Compression | 8 | 46 | 50 | 48 | 40 | 184/200 (92.0%) | -7.0 | 21.14 | 1.162× | 14.62 |
| dart_keep75 [R74](#r74) | 5.1 Token Pruning and Context Compression | 8 | 47 | 50 | 50 | 43 | 190/200 (95.0%) | -4.0 | 23.81 | 1.032× | 14.66 |
| dart_layer4 [R75](#r75) | 5.1 Token Pruning and Context Compression | 8 | 50 | 50 | 49 | 48 | 197/200 (98.5%) | -0.5 | 21.64 | 1.135× | 14.63 |
| specprune_vla [R76](#r76) | 5.1 Token Pruning and Context Compression | 8 | 36 | 45 | 43 | 33 | 157/200 (78.5%) | -20.5 | 19.71 | 1.247× | 14.60 |
| specprune_vla_score_first [R77](#r77) | 5.1 Token Pruning and Context Compression | 8 | 49 | 48 | 50 | 46 | 193/200 (96.5%) | -2.5 | 19.60 | 1.254× | 14.60 |
| specprune_vla_static128 [R78](#r78) | 5.1 Token Pruning and Context Compression | 8 | 45 | 50 | 46 | 41 | 182/200 (91.0%) | -8.0 | 24.32 | 1.010× | 14.62 |
| st_merge [R65](#r65) | 5.1 Token Pruning and Context Compression | 8 | 35 | 41 | 21 | 27 | 124/200 (62.0%) | -37.0 | 21.07 | 1.151× | 14.62 |
| st_merge_keep192 [R66](#r66) | 5.1 Token Pruning and Context Compression | 8 | 50 | 50 | 47 | 50 | 197/200 (98.5%) | -0.5 | 23.71 | 1.023× | 14.66 |
| st_merge_layer6 [R67](#r67) | 5.1 Token Pruning and Context Compression | 8 | 42 | 46 | 46 | 40 | 174/200 (87.0%) | -12.0 | 21.04 | 1.153× | 14.62 |
| team_vla [R68](#r68) | 5.1 Token Pruning and Context Compression | 8 | 30 | 49 | 7 | 11 | 97/200 (48.5%) | -50.5 | 17.10 | 1.418× | 14.61 |
| team_vla_context50 [R69](#r69) | 5.1 Token Pruning and Context Compression | 8 | 24 | 49 | 0 | 9 | 82/200 (41.0%) | -58.0 | 18.28 | 1.327× | 14.62 |
| team_vla_merge160 [R70](#r70) | 5.1 Token Pruning and Context Compression | 8 | 40 | 50 | 43 | 34 | 167/200 (83.5%) | -15.5 | 18.83 | 1.288× | 14.62 |

<details>
<summary>Development configuration identities</summary>

Raw evidence is retained under each campaign's `runs/<backend>--<preset>--sNNN/`.
The aggregate retains every episode outcome, source revision, shard and file hash.
A timing record is an audited historical reference, not a new measurement.

| Configuration | Experiment identity | Runtime source | Shards |
|---|---|---|---:|
| cogact/actuation_slack_refresh_deletion | `b53ec6957ea787d2670b39019271cea2a415f75fce5b3213e5132d201e1c4c50` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/actuation_slack_refresh_keep25 | `1745acc6a866125188a9997d38d3a193bf3f2238f71a72e8522ed6a2933d2115` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/actuation_slack_refresh_keep50 | `3e9ab07826e30785e609207a39c4c964aadc37185ff424e01b3643cfb527bd3c` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/baseline | `a293c4ce24a5b840f961549f41769a1f59f1952c990b14d0ebef644ce8bda358` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/efficient_vla_cache2 | `0fe58b9b38744521cbee052de8735a466a80268edd77387812966ed72bd71eae` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/efficient_vla_cache3 | `e265084237e3a715d4af23bf9fd4aa704cc121ee7b3e70d5addd79efc2792590` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/efficient_vla_diffusion_cache | `2e99d4df59fd76c5961cbf0ce7b5e263c64eb81b2b7f72f60da5893bac5920e3` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/flash_vla | `a3241c13b054520b72fe7d7a8afe665de8ccc3521dd177151f134b84c24b33ba` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/flash_vla_keep192 | `db0475cbd71de3b42e3295cedd878c5523399e86ac6afa49fd522724ba6b56b0` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/flash_vla_layer4 | `929ebfad8fec553862d092952ebe3fb227b4e5f294a4e48a086c41b7bccf0505` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/specprune_vla | `200e2a5546ff87e327bf91696fb9a35b9f0a400baf9f744101f68e74fa752304` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/specprune_vla_score_first | `fe7382cd56d89700e7438270dea06822428a12da4c859470ad533a82a41f3541` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/specprune_vla_static128 | `b222be6835713b1bb7b490fd083736c4ac616b7a52b34f8e38a94c347d58c257` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/st_merge | `2a63e1b14d73cbda0afa52b35c299e770af5a3dfa14025eedb31eca4c39edb65` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/st_merge_keep192 | `2715131409b114d95dc8dc1349d3ea4da8b4baa5ab6f0e78127aff0d3dc431eb` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| cogact/st_merge_layer6 | `a95be08d8cbc582a2558aa351954afb2b8579a12e27af9696ec22604a65e9a82` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/baseline | `83a2952de24bf599cafcbe504ce1b8e0c5f91824411122370deec9846dbaa71e` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/dart | `959698fd55bb54c37ba68a4b14bc589d41d41e4cab340d7023a14ce24af65841` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/dart_keep75 | `eedd9ce438b44b17e173e87a2dfafe2ede3602748c923c433b50b0819c542c5c` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/dart_layer4 | `03a1390651f10da98428fda879b3bb34131f032a02463218beb1c31648b493b9` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/specprune_vla | `db98812e4d43c083caa9734b047770a7de7e2af0f2446e06e730cf3e3be7cb34` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/specprune_vla_score_first | `4ad4e29d8c6340d9596de67535f4e4319935c722593a0e5be25a3e5974120685` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/specprune_vla_static128 | `dbdcd2915f4c533eaf7435cfd041a87f7aeb0ad9760c4700a40df21b0ed5e149` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/st_merge | `03eda8c029c9b6266595a28fb15e9b26b06271c75879bd824270d01f09393698` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/st_merge_keep192 | `d6e8965949d65ac92a7079a13bbe478bc63fd38ef9d9657f513d8051b75f9a46` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/st_merge_layer6 | `9cc7912be87ead82ace90c6a604b0076969e51f865358e3ccc6fee58fa0d824f` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/team_vla | `01da23f467ef2f675afa00620cd10be36f2366225c2ab2bc1d4a69d012ea1d43` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/team_vla_context50 | `861133c20074d7799cf7c6549b0e5b5b1837e0af166c18fa8c903070686b8dd9` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |
| openvla_oft/team_vla_merge160 | `02fc7c9f038ee76b07c85196460993479de227eba74c4343cca09310552f327c` | `5b5740bec7bf22ac88b188de1afe1aca0373d62566c28df4a9498e4fdfcb340e` | 8 |

</details>

## 0. Native references

| Record / configuration | Backend | H | Queries / replacements | Query mean / p95 ms | ms/control step | Replay speedup | Peak GiB | Smoke |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [R01](#r01) Native H=8 (token batch 2) | openvla_oft | 8 | 216 / 0 | 197.31 / 199.10 | 25.03 | 1.000× | 14.98 | Not run |
| [R02](#r02) Native H=8 end recheck (token batch 2) | openvla_oft | 8 | 216 / 0 | 198.63 / 200.64 | 25.20 | 0.993× | 14.98 | Not run |
| [R12](#r12) CogACT native H8 / category examples | cogact | 8 | 321 / 0 | 185.37 / 186.99 | 23.60 | 1.000× | 14.73 | 3/4; errors 0 |
| [R13](#r13) CogACT native H1 / matched VOTE reference | cogact | 1 | 2523 / 0 | 185.82 / 186.85 | 185.82 | 1.000× | 14.73 | 4/4; errors 0 |
| [R14](#r14) CogACT native H8 / end-of-cohort recheck | cogact | 8 | 321 / 0 | 184.56 / 185.72 | 23.50 | 1.004× | 14.73 | Not run |
| [R19](#r19) CogACT native H8 / completion cohort | cogact | 8 | 321 / 0 | 185.18 / 187.03 | 23.58 | 1.000× | 14.73 | 3/4; errors 0 |
| [R20](#r20) CogACT native H8 / completion end recheck | cogact | 8 | 321 / 0 | 184.39 / 185.80 | 23.48 | 1.004× | 14.73 | Not run |
| [R26](#r26) CogACT native DDIM / primary runtime | cogact | 8 | 321 / 0 | 183.69 / 184.81 | 23.39 | 1.000× | 14.73 | 3/4; errors 0 |
| [R27](#r27) CogACT native DDIM / secondary runtime | cogact | 8 | 321 / 0 | 194.39 / 204.01 | 24.75 | 1.000× | 14.73 | 3/4; errors 0 |
| [R28](#r28) CogACT native DDIM / secondary recheck (thermal throttling) | cogact | 8 | 321 / 0 | 195.57 / 209.94 | 24.90 | 1.000× | 14.73 | 3/4; errors 0 |
| [R30](#r30) CogACT native H=8 (token batch 1) | cogact | 8 | 321 / 0 | 183.86 / 184.84 | 23.41 | 1.000× | 14.73 | 3/4; errors 0 |
| [R31](#r31) CogACT native H=8 end recheck (token batch 1) | cogact | 8 | 321 / 0 | 184.36 / 185.67 | 23.47 | 0.997× | 14.73 | Not run |
| [R35](#r35) Native BF16 / SDPA | openvla_oft | 8 | 216 / 0 | 197.86 / 199.86 | 25.10 | 1.000× | 14.98 | 4/4; errors 0 |
| [R36](#r36) Native BF16 / SDPA, matched H=1 | openvla_oft | 1 | 1704 / 0 | 199.83 / 201.42 | 199.83 | 1.000× | 14.98 | Not run |
| [R37](#r37) Native BF16 / SDPA, end recheck | openvla_oft | 8 | 216 / 0 | 198.91 / 201.15 | 25.23 | 0.995× | 14.98 | Not run |
| [R80](#r80) Native H=8 (fidelity audit) | openvla_oft | 8 | 216 / 0 | 196.74 / 198.44 | 24.96 | 1.000× | 14.98 | Not run |
| [R81](#r81) Native H=8 end recheck (fidelity audit) | openvla_oft | 8 | 216 / 0 | 197.84 / 199.24 | 25.10 | 0.994× | 14.98 | Not run |
| [R110](#r110) Native H=8 (token batch 1) | openvla_oft | 8 | 216 / 0 | 200.57 / 203.81 | 25.44 | 1.000× | 14.98 | Not run |
| [R111](#r111) Native H=8 end recheck (token batch 1) | openvla_oft | 8 | 216 / 0 | 198.77 / 200.57 | 25.21 | 1.009× | 14.98 | Not run |

## 5.1 Token Pruning and Context Compression

| Record / configuration | Backend | H | Queries / replacements | Query mean / p95 ms | ms/control step | Replay speedup | Peak GiB | Smoke |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [R03](#r03) Superseded — EfficientVLA token core (paper-based) | openvla_oft | 8 | 216 / 0 | 170.08 / 171.33 | 21.58 | 1.160× | 14.62 | 4/4; errors 0 |
| [R04](#r04) FlashVLA ICS core (paper-based) | openvla_oft | 8 | 216 / 0 | 238.64 / 241.15 | 30.27 | 0.827× | 14.64 | 4/4; errors 0 |
| [R07](#r07) Superseded — ST-Merge spatial VLA (paper-based) | openvla_oft | 8 | 216 / 0 | 171.42 / 172.94 | 21.75 | 1.151× | 14.62 | 4/4; errors 0 |
| [R22](#r22) SAFE-Pruner / CogACT cognition-query port | cogact | 8 | 321 / 0 | 125.95 / 128.48 | 16.04 | 1.470× | 14.74 | 3/4; errors 0 |
| [R23](#r23) SpecPrune-VLA / CogACT pruning and controller | cogact | 8 | 351 / 0 | 130.67 / 134.37 | 18.26 | 1.291× | 14.56 | 3/4; errors 0 |
| [R24](#r24) ST-Merge spatial VLA / CogACT | cogact | 8 | 321 / 0 | 130.27 / 131.58 | 16.59 | 1.421× | 14.63 | 0/4; errors 0 |
| [R29](#r29) ADP / CogACT (observed EEF) | cogact | 8 | 321 / 0 | 168.27 / 184.16 | 21.44 | 1.092× | 14.73 | 4/4; errors 0 |
| [R32](#r32) FlashVLA ICS / CogACT (complete SVD) | cogact | 8 | 321 / 0 | 157.56 / 166.44 | 20.07 | 1.167× | 14.66 | 4/4; errors 0 |
| [R39](#r39) SpecPrune-VLA | openvla_oft | 8 | 237 / 0 | 146.91 / 163.52 | 20.51 | 1.224× | 14.60 | 3/4; errors 0 |
| [R42](#r42) Native baseline (next token batch) / diagnostic baseline / cogact_a reference | cogact | 8 | 321 / 0 | 185.13 / 187.22 | 23.58 | 1.000× | 14.73 | Not run |
| [R43](#r43) Native baseline (next token batch) / diagnostic baseline / cogact_a recheck | cogact | 8 | 321 / 0 | 185.21 / 187.17 | 23.59 | 1.000× | 14.73 | Not run |
| [R44](#r44) FlashVLA ICS / CogACT (complete SVD) / diagnostic flash_vla | cogact | 8 | 321 / 0 | 157.77 / 166.81 | 20.09 | 1.173× | 14.66 | Not run |
| [R45](#r45) FlashVLA ICS / CogACT (complete SVD) / diagnostic flash_vla_keep192 | cogact | 8 | 321 / 0 | 168.57 / 170.81 | 21.47 | 1.098× | 14.69 | Not run |
| [R46](#r46) SpecPrune-VLA / CogACT pruning and controller / diagnostic specprune_vla | cogact | 8 | 351 / 0 | 128.14 / 133.84 | 17.93 | 1.315× | 14.56 | Not run |
| [R47](#r47) SpecPrune-VLA / CogACT pruning and controller / diagnostic specprune_vla_score_first | cogact | 8 | 348 / 0 | 130.80 / 136.55 | 18.14 | 1.300× | 14.56 | Not run |
| [R48](#r48) SpecPrune-VLA / CogACT pruning and controller / diagnostic specprune_vla_static128 | cogact | 8 | 363 / 0 | 138.82 / 141.85 | 20.07 | 1.175× | 14.62 | Not run |
| [R49](#r49) ST-Merge spatial VLA / CogACT / diagnostic st_merge | cogact | 8 | 321 / 0 | 129.87 / 131.83 | 16.54 | 1.425× | 14.63 | Not run |
| [R50](#r50) ST-Merge spatial VLA / CogACT / diagnostic st_merge_keep192 | cogact | 8 | 321 / 0 | 156.90 / 158.53 | 19.98 | 1.180× | 14.68 | Not run |
| [R51](#r51) ST-Merge spatial VLA / CogACT / diagnostic st_merge_layer6 | cogact | 8 | 321 / 0 | 130.28 / 132.87 | 16.60 | 1.421× | 14.63 | Not run |
| [R52](#r52) Native baseline (next token batch) / diagnostic baseline / cogact_b_clean_b reference | cogact | 8 | 321 / 0 | 185.08 / 187.70 | 23.57 | 1.000× | 14.73 | Not run |
| [R53](#r53) Native baseline (next token batch) / diagnostic baseline / cogact_b_clean_b recheck | cogact | 8 | 321 / 0 | 184.71 / 188.75 | 23.52 | 1.002× | 14.73 | Not run |
| [R57](#r57) ASR deletion (all refresh cost) / diagnostic actuation_slack_refresh_deletion | cogact | 8 | 321 / 0 | 342.95 / 362.42 | 43.67 | 0.532× | 14.74 | Not run |
| [R58](#r58) ASR deletion (all refresh cost) / diagnostic actuation_slack_refresh_keep25 | cogact | 8 | 321 / 0 | 340.01 / 359.44 | 43.29 | 0.537× | 14.74 | Not run |
| [R59](#r59) ASR deletion (all refresh cost) / diagnostic actuation_slack_refresh_keep50 | cogact | 8 | 321 / 0 | 350.89 / 366.54 | 44.68 | 0.520× | 14.74 | Not run |
| [R60](#r60) Native baseline (next token batch) / diagnostic baseline / cogact_b_stable reference | cogact | 8 | 321 / 0 | 182.59 / 183.72 | 23.25 | 1.000× | 14.73 | Not run |
| [R61](#r61) Native baseline (next token batch) / diagnostic baseline / cogact_b_stable recheck | cogact | 8 | 321 / 0 | 182.61 / 183.68 | 23.25 | 1.000× | 14.73 | Not run |
| [R62](#r62) FlashVLA ICS / CogACT (complete SVD) / diagnostic flash_vla_layer4 | cogact | 8 | 321 / 0 | 160.27 / 168.41 | 20.41 | 1.139× | 14.67 | Not run |
| [R63](#r63) Native baseline (next token batch) / diagnostic baseline / openvla_oft_a_stable2 reference | openvla_oft | 8 | 216 / 0 | 191.23 / 192.62 | 24.26 | 1.000× | 14.98 | Not run |
| [R64](#r64) Native baseline (next token batch) / diagnostic baseline / openvla_oft_a_stable2 recheck | openvla_oft | 8 | 216 / 0 | 192.26 / 193.83 | 24.39 | 0.995× | 14.98 | Not run |
| [R65](#r65) ST-Merge spatial VLA / key dot product / diagnostic st_merge | openvla_oft | 8 | 216 / 0 | 166.09 / 168.19 | 21.07 | 1.151× | 14.62 | Not run |
| [R66](#r66) ST-Merge spatial VLA / key dot product / diagnostic st_merge_keep192 | openvla_oft | 8 | 216 / 0 | 186.91 / 188.53 | 23.71 | 1.023× | 14.66 | Not run |
| [R67](#r67) ST-Merge spatial VLA / key dot product / diagnostic st_merge_layer6 | openvla_oft | 8 | 216 / 0 | 165.85 / 168.21 | 21.04 | 1.153× | 14.62 | Not run |
| [R68](#r68) TEAM-VLA / audited paper-based port / diagnostic team_vla | openvla_oft | 8 | 216 / 0 | 134.78 / 143.20 | 17.10 | 1.418× | 14.61 | Not run |
| [R69](#r69) TEAM-VLA / audited paper-based port / diagnostic team_vla_context50 | openvla_oft | 8 | 216 / 0 | 144.03 / 145.55 | 18.28 | 1.327× | 14.62 | Not run |
| [R70](#r70) TEAM-VLA / audited paper-based port / diagnostic team_vla_merge160 | openvla_oft | 8 | 216 / 0 | 148.43 / 156.39 | 18.83 | 1.288× | 14.62 | Not run |
| [R71](#r71) Native baseline (next token batch) / diagnostic baseline / openvla_oft_b_stable reference | openvla_oft | 8 | 216 / 0 | 193.68 / 195.10 | 24.57 | 1.000× | 14.98 | Not run |
| [R72](#r72) Native baseline (next token batch) / diagnostic baseline / openvla_oft_b_stable recheck | openvla_oft | 8 | 216 / 0 | 195.01 / 196.57 | 24.74 | 0.993× | 14.98 | Not run |
| [R73](#r73) DART token reduction / diagnostic dart | openvla_oft | 8 | 216 / 0 | 166.60 / 167.85 | 21.14 | 1.162× | 14.62 | Not run |
| [R74](#r74) DART token reduction / diagnostic dart_keep75 | openvla_oft | 8 | 216 / 0 | 187.65 / 189.15 | 23.81 | 1.032× | 14.66 | Not run |
| [R75](#r75) DART token reduction / diagnostic dart_layer4 | openvla_oft | 8 | 216 / 0 | 170.58 / 172.25 | 21.64 | 1.135× | 14.63 | Not run |
| [R76](#r76) SpecPrune-VLA / control recheck / diagnostic specprune_vla | openvla_oft | 8 | 237 / 0 | 140.84 / 157.34 | 19.71 | 1.247× | 14.60 | Not run |
| [R77](#r77) SpecPrune-VLA / control recheck / diagnostic specprune_vla_score_first | openvla_oft | 8 | 237 / 0 | 140.24 / 156.07 | 19.60 | 1.254× | 14.60 | Not run |
| [R78](#r78) SpecPrune-VLA / control recheck / diagnostic specprune_vla_static128 | openvla_oft | 8 | 237 / 0 | 174.16 / 178.77 | 24.32 | 1.010× | 14.62 | Not run |
| [R79](#r79) ADP / control recheck | openvla_oft | 8 | 216 / 0 | 193.63 / 198.37 | 24.58 | 1.015× | 14.68 | 4/4; errors 0 |
| [R82](#r82) EfficientVLA token core / corrected Q-K direction | openvla_oft | 8 | 216 / 0 | 170.19 / 172.02 | 21.59 | 1.156× | 14.63 | 4/4; errors 0 |
| [R83](#r83) FlashVLA ICS / automatic complete SVD | openvla_oft | 8 | 216 / 0 | 208.73 / 210.37 | 26.48 | 0.943× | 14.64 | 4/4; errors 0 |
| [R84](#r84) FlashVLA ICS / gesvd reference | openvla_oft | 8 | 216 / 0 | 237.86 / 240.28 | 30.17 | 0.827× | 14.64 | Not run |
| [R85](#r85) SAFE-Pruner / control recheck | openvla_oft | 8 | 216 / 0 | 146.76 / 198.59 | 18.62 | 1.340× | 14.68 | 4/4; errors 0 |
| [R86](#r86) SpecPrune-VLA / control recheck | openvla_oft | 8 | 237 / 0 | 146.13 / 162.17 | 20.42 | 1.222× | 14.60 | 3/4; errors 0 |
| [R87](#r87) ST-Merge spatial VLA / key dot product | openvla_oft | 8 | 216 / 0 | 171.36 / 172.73 | 21.74 | 1.148× | 14.62 | 3/4; errors 0 |
| [R88](#r88) TEAM-VLA / audited paper-based port | openvla_oft | 8 | 216 / 0 | 139.58 / 148.03 | 17.71 | 1.409× | 14.61 | 1/4; errors 0 |
| [R90](#r90) VLA-IAP / audited paper-based port | openvla_oft | 8 | 216 / 0 | 193.23 / 200.12 | 24.51 | 1.018× | 14.69 | 3/4; errors 0 |
| [R92](#r92) VLA-Pruner / control recheck | openvla_oft | 8 | 216 / 0 | 163.52 / 198.05 | 20.75 | 1.203× | 14.68 | 4/4; errors 0 |
| [R93](#r93) ASR deletion (all refresh cost) | cogact | 8 | 321 / 0 | 299.19 / 301.75 | 38.10 | 0.618× | 14.74 | 3/4; errors 0 |
| [R94](#r94) Native baseline (next token batch) | cogact | 8 | 321 / 0 | 184.84 / 186.29 | 23.54 | 1.000× | 14.73 | 3/4; errors 0 |
| [R95](#r95) Native end recheck (next token batch) | cogact | 8 | 321 / 0 | 184.73 / 185.95 | 23.52 | 1.001× | 14.73 | 3/4; errors 0 |
| [R96](#r96) DART token reduction | cogact | 8 | 321 / 0 | 133.85 / 135.12 | 17.05 | 1.380× | 14.64 | 4/4; errors 0 |
| [R97](#r97) DepthCache primary-view core | cogact | 8 | 321 / 0 | 186.87 / 190.49 | 23.80 | 0.989× | 14.73 | 4/4; errors 0 |
| [R98](#r98) DivPrune | cogact | 8 | 321 / 0 | 139.91 / 140.91 | 17.82 | 1.321× | 14.63 | 4/4; errors 0 |
| [R99](#r99) FastV released last-query | cogact | 8 | 321 / 0 | 133.04 / 134.15 | 16.95 | 1.389× | 14.64 | 4/4; errors 0 |
| [R100](#r100) SparseVLM v1 rank + recycling | cogact | 8 | 321 / 0 | 147.35 / 152.05 | 18.77 | 1.254× | 14.64 | 4/4; errors 0 |
| [R101](#r101) ASR deletion (all refresh cost) | openvla_oft | 8 | 216 / 0 | 366.36 / 385.42 | 46.47 | 0.535× | 14.69 | 4/4; errors 0 |
| [R102](#r102) Native baseline (next token batch) | openvla_oft | 8 | 216 / 0 | 195.90 / 197.68 | 24.85 | 1.000× | 14.98 | 4/4; errors 0 |
| [R103](#r103) Native end recheck (next token batch) | openvla_oft | 8 | 216 / 0 | 197.46 / 198.99 | 25.05 | 0.992× | 14.98 | 4/4; errors 0 |
| [R104](#r104) DART token reduction | openvla_oft | 8 | 216 / 0 | 170.99 / 172.23 | 21.69 | 1.146× | 14.62 | 3/4; errors 0 |
| [R105](#r105) DepthCache primary-view core | openvla_oft | 8 | 216 / 0 | 200.80 / 204.22 | 25.47 | 0.976× | 14.68 | 4/4; errors 0 |
| [R106](#r106) DivPrune | openvla_oft | 8 | 216 / 0 | 189.20 / 190.99 | 24.00 | 1.035× | 14.62 | 4/4; errors 0 |
| [R107](#r107) FastV released last-query | openvla_oft | 8 | 216 / 0 | 168.85 / 170.50 | 21.42 | 1.160× | 14.62 | 4/4; errors 0 |
| [R108](#r108) SparseVLM v1 rank + recycling | openvla_oft | 8 | 216 / 0 | 183.37 / 187.03 | 23.26 | 1.068× | 14.63 | 4/4; errors 0 |
| [R109](#r109) ADP | openvla_oft | 8 | 216 / 0 | 195.37 / 200.98 | 24.80 | 1.026× | 14.68 | 4/4; errors 0 |
| [R112](#r112) SAFE-Pruner | openvla_oft | 8 | 216 / 0 | 148.28 / 200.16 | 18.82 | 1.352× | 14.68 | 4/4; errors 0 |
| [R113](#r113) SpecPrune-VLA (token batch 1) | openvla_oft | 8 | 237 / 0 | 147.06 / 163.44 | 20.53 | 1.239× | 14.60 | 3/4; errors 0 |
| [R114](#r114) TEAM-VLA (paper-based) | openvla_oft | 8 | 216 / 0 | 139.47 / 148.07 | 17.70 | 1.438× | 14.61 | 1/4; errors 0 |
| [R115](#r115) VLA-IAP (paper-based) | openvla_oft | 8 | 216 / 0 | 196.31 / 204.09 | 24.90 | 1.022× | 14.69 | 3/4; errors 0 |
| [R116](#r116) VLA-Pruner | openvla_oft | 8 | 216 / 0 | 165.05 / 199.81 | 20.94 | 1.215× | 14.68 | 4/4; errors 0 |

## 5.2 Caching and Computation Reuse

| Record / configuration | Backend | H | Queries / replacements | Query mean / p95 ms | ms/control step | Replay speedup | Peak GiB | Smoke |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [R08](#r08) VLA-Cache full statistics / eager (token batch 2) | openvla_oft | 8 | 216 / 0 | 207.43 / 217.93 | 26.31 | 0.951× | 15.03 | Not run |
| [R09](#r09) VLA-InfoEntropy KV reuse (paper-based) | openvla_oft | 8 | 216 / 0 | 213.14 / 222.39 | 27.04 | 0.926× | 15.03 | 4/4; errors 0 |
| [R16](#r16) VLA-Cache full statistics eager / CogACT | cogact | 8 | 321 / 0 | 158.58 / 167.35 | 20.20 | 1.169× | 14.75 | 3/4; errors 0 |
| [R17](#r17) VLA-Cache full statistics SDPA / CogACT | cogact | 8 | 321 / 0 | 159.88 / 168.24 | 20.36 | 1.159× | 14.75 | Not run |
| [R21](#r21) EfficientVLA DiT cache core / CogACT (N=5) | cogact | 8 | 321 / 0 | 165.52 / 166.65 | 21.08 | 1.119× | 14.73 | 3/4; errors 0 |
| [R25](#r25) VLA-InfoEntropy / CogACT full statistics | cogact | 8 | 321 / 0 | 170.24 / 176.85 | 21.68 | 1.088× | 14.75 | 3/4; errors 0 |
| [R40](#r40) VLA-Cache, full statistics / eager | openvla_oft | 8 | 216 / 0 | 207.64 / 219.38 | 26.34 | 0.953× | 15.03 | 4/4; errors 0 |
| [R54](#r54) EfficientVLA DiT cache core / CogACT (N=5) / diagnostic efficient_vla_cache2 | cogact | 8 | 321 / 0 | 173.68 / 176.38 | 22.12 | 1.066× | 14.73 | Not run |
| [R55](#r55) EfficientVLA DiT cache core / CogACT (N=5) / diagnostic efficient_vla_cache3 | cogact | 8 | 321 / 0 | 170.40 / 172.25 | 21.70 | 1.086× | 14.73 | Not run |
| [R56](#r56) EfficientVLA DiT cache core / CogACT (N=5) / diagnostic efficient_vla_diffusion_cache | cogact | 8 | 321 / 0 | 165.49 / 167.04 | 21.08 | 1.118× | 14.73 | Not run |
| [R89](#r89) VLA-Cache full statistics / control recheck | openvla_oft | 8 | 216 / 0 | 206.24 / 216.85 | 26.16 | 0.954× | 15.03 | 4/4; errors 0 |
| [R91](#r91) VLA-InfoEntropy / audited entropy and KV reuse | openvla_oft | 8 | 216 / 0 | 213.47 / 222.80 | 27.08 | 0.922× | 15.03 | 4/4; errors 0 |

## 5.3 Layer Skipping, Early Exit, and Model Routing

| Record / configuration | Backend | H | Queries / replacements | Query mean / p95 ms | ms/control step | Replay speedup | Peak GiB | Smoke |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [R05](#r05) SP-VLA pruning + scheduling | openvla_oft | 8 | 201 / 153 | 189.63 / 201.81 | 22.41 | 1.117× | 14.68 | 4/4; errors 0 |
| [R06](#r06) SP-VLA scheduling-only ablation (token batch 2) | openvla_oft | 8 | 201 / 153 | 197.93 / 200.72 | 23.39 | 1.070× | 14.98 | Not run |
| [R15](#r15) SP-VLA pruning + scheduling / CogACT | cogact | 8 | 306 / 141 | 171.98 / 189.20 | 20.89 | 1.130× | 14.73 | 4/4; errors 0 |
| [R38](#r38) SP-VLA scheduling core | openvla_oft | 8 | 201 / 153 | 197.51 / 199.26 | 23.34 | 1.076× | 14.98 | 4/4; errors 0 |

## 5.5 Action Chunking and Replanning

| Record / configuration | Backend | H | Queries / replacements | Query mean / p95 ms | ms/control step | Replay speedup | Peak GiB | Smoke |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [R18](#r18) VOTE ensemble core / CogACT | cogact | 1 | 2523 / 0 | 186.53 / 187.57 | 186.53 | 0.996× | 14.73 | 3/4; errors 0 |
| [R41](#r41) VOTE ensemble core | openvla_oft | 1 | 1704 / 0 | 199.62 / 201.47 | 199.62 | 1.001× | 14.98 | 3/4; errors 0 |

## 6.1 Quantization and Low-Precision Inference

| Record / configuration | Backend | H | Queries / replacements | Query mean / p95 ms | ms/control step | Replay speedup | Peak GiB | Smoke |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [R10](#r10) AWQ W4A16 BF16 kernel / CogACT | cogact | 8 | 321 / 0 | 155.51 / 156.42 | 19.81 | 1.192× | 5.90 | 3/4; errors 0 |
| [R11](#r11) AWQ W4A16 explicit FP16 kernel / CogACT | cogact | 8 | 321 / 0 | 135.92 / 136.99 | 17.31 | 1.363× | 5.90 | 4/4; errors 0 |
| [R33](#r33) AWQ W4A16, BF16 kernel | openvla_oft | 8 | 216 / 0 | 204.95 / 206.69 | 26.00 | 0.965× | 6.12 | 4/4; errors 0 |
| [R34](#r34) AWQ W4A16, explicit FP16 kernel | openvla_oft | 8 | 216 / 0 | 170.32 / 171.73 | 21.61 | 1.162× | 6.12 | 4/4; errors 0 |

## Observed visual-token retention

Served-query-weighted counts, including dense served warmup/refresh; these
are not configured keep ratios or FLOPs. Final counts do not describe
earlier-layer cost. Per-layer sequence lengths remain in raw diagnostics.
Discarded auxiliary passes are counted separately below, not as served queries.

| Record / method | Backend | Final mean / min / max | Early mean | Mode counts (true/observed) |
|---|---|---:|---:|---|
| [R03](#r03) Superseded — EfficientVLA token core (paper-based) | openvla_oft | 256.0 / 256 / 256 | — | — |
| [R04](#r04) FlashVLA ICS core (paper-based) | openvla_oft | 320.0 / 320 / 320 | — | — |
| [R05](#r05) SP-VLA pruning + scheduling | openvla_oft | 418.9 / 158 / 512 | — | — |
| [R07](#r07) Superseded — ST-Merge spatial VLA (paper-based) | openvla_oft | 256.0 / 256 / 256 | — | — |
| [R15](#r15) SP-VLA pruning + scheduling / CogACT | cogact | 219.0 / 108 / 256 | — | — |
| [R22](#r22) SAFE-Pruner / CogACT cognition-query port | cogact | 73.0 / 64 / 256 | — | keyframe 15/321 |
| [R23](#r23) SpecPrune-VLA / CogACT pruning and controller | cogact | 32.6 / 30 / 51 | — | — |
| [R24](#r24) ST-Merge spatial VLA / CogACT | cogact | 128.0 / 128 / 128 | — | — |
| [R29](#r29) ADP / CogACT (observed EEF) | cogact | 223.7 / 192 / 256 | — | pruned 162/321 |
| [R32](#r32) FlashVLA ICS / CogACT (complete SVD) | cogact | 160.0 / 160 / 160 | — | — |
| [R44](#r44) FlashVLA ICS / CogACT (complete SVD) / diagnostic flash_vla | cogact | 160.0 / 160 / 160 | — | — |
| [R45](#r45) FlashVLA ICS / CogACT (complete SVD) / diagnostic flash_vla_keep192 | cogact | 192.0 / 192 / 192 | — | — |
| [R46](#r46) SpecPrune-VLA / CogACT pruning and controller / diagnostic specprune_vla | cogact | 32.6 / 30 / 51 | — | — |
| [R47](#r47) SpecPrune-VLA / CogACT pruning and controller / diagnostic specprune_vla_score_first | cogact | 32.3 / 30 / 48 | — | — |
| [R48](#r48) SpecPrune-VLA / CogACT pruning and controller / diagnostic specprune_vla_static128 | cogact | 80.6 / 75 / 87 | — | — |
| [R49](#r49) ST-Merge spatial VLA / CogACT / diagnostic st_merge | cogact | 128.0 / 128 / 128 | — | — |
| [R50](#r50) ST-Merge spatial VLA / CogACT / diagnostic st_merge_keep192 | cogact | 192.0 / 192 / 192 | — | — |
| [R51](#r51) ST-Merge spatial VLA / CogACT / diagnostic st_merge_layer6 | cogact | 128.0 / 128 / 128 | — | — |
| [R57](#r57) ASR deletion (all refresh cost) / diagnostic actuation_slack_refresh_deletion | cogact | 34.6 / 26 / 256 | — | — |
| [R58](#r58) ASR deletion (all refresh cost) / diagnostic actuation_slack_refresh_keep25 | cogact | 71.2 / 64 / 256 | — | — |
| [R59](#r59) ASR deletion (all refresh cost) / diagnostic actuation_slack_refresh_keep50 | cogact | 132.8 / 128 / 256 | — | — |
| [R62](#r62) FlashVLA ICS / CogACT (complete SVD) / diagnostic flash_vla_layer4 | cogact | 160.0 / 160 / 160 | — | — |
| [R65](#r65) ST-Merge spatial VLA / key dot product / diagnostic st_merge | openvla_oft | 256.0 / 256 / 256 | — | — |
| [R66](#r66) ST-Merge spatial VLA / key dot product / diagnostic st_merge_keep192 | openvla_oft | 384.0 / 384 / 384 | — | — |
| [R67](#r67) ST-Merge spatial VLA / key dot product / diagnostic st_merge_layer6 | openvla_oft | 256.0 / 256 / 256 | — | — |
| [R68](#r68) TEAM-VLA / audited paper-based port / diagnostic team_vla | openvla_oft | 80.0 / 80 / 80 | 269.1 | — |
| [R69](#r69) TEAM-VLA / audited paper-based port / diagnostic team_vla_context50 | openvla_oft | 80.0 / 80 / 80 | 349.8 | — |
| [R70](#r70) TEAM-VLA / audited paper-based port / diagnostic team_vla_merge160 | openvla_oft | 160.0 / 160 / 160 | 269.1 | — |
| [R73](#r73) DART token reduction / diagnostic dart | openvla_oft | 260.0 / 260 / 260 | — | — |
| [R74](#r74) DART token reduction / diagnostic dart_keep75 | openvla_oft | 388.0 / 388 / 388 | — | — |
| [R75](#r75) DART token reduction / diagnostic dart_layer4 | openvla_oft | 260.0 / 260 / 260 | — | — |
| [R76](#r76) SpecPrune-VLA / control recheck / diagnostic specprune_vla | openvla_oft | 68.5 / 60 / 108 | — | — |
| [R77](#r77) SpecPrune-VLA / control recheck / diagnostic specprune_vla_score_first | openvla_oft | 67.9 / 60 / 109 | — | — |
| [R78](#r78) SpecPrune-VLA / control recheck / diagnostic specprune_vla_static128 | openvla_oft | 146.9 / 135 / 162 | — | — |
| [R79](#r79) ADP / control recheck | openvla_oft | 458.7 / 384 / 512 | — | pruned 90/216 |
| [R82](#r82) EfficientVLA token core / corrected Q-K direction | openvla_oft | 256.0 / 256 / 256 | — | — |
| [R83](#r83) FlashVLA ICS / automatic complete SVD | openvla_oft | 320.0 / 320 / 320 | — | — |
| [R84](#r84) FlashVLA ICS / gesvd reference | openvla_oft | 320.0 / 320 / 320 | — | — |
| [R85](#r85) SAFE-Pruner / control recheck | openvla_oft | 170.7 / 128 / 512 | — | keyframe 24/216 |
| [R86](#r86) SpecPrune-VLA / control recheck | openvla_oft | 68.5 / 60 / 108 | — | — |
| [R87](#r87) ST-Merge spatial VLA / key dot product | openvla_oft | 256.0 / 256 / 256 | — | — |
| [R88](#r88) TEAM-VLA / audited paper-based port | openvla_oft | 80.0 / 80 / 80 | 269.1 | — |
| [R90](#r90) VLA-IAP / audited paper-based port | openvla_oft | 438.0 / 174 / 512 | — | warmup 24/216 |
| [R92](#r92) VLA-Pruner / control recheck | openvla_oft | 192.0 / 128 / 512 | — | — |
| [R93](#r93) ASR deletion (all refresh cost) | cogact | 34.6 / 26 / 256 | — | — |
| [R96](#r96) DART token reduction | cogact | 132.0 / 132 / 132 | — | — |
| [R97](#r97) DepthCache primary-view core | cogact | 251.8 / 233 / 256 | — | — |
| [R98](#r98) DivPrune | cogact | 128.0 / 128 / 128 | — | — |
| [R99](#r99) FastV released last-query | cogact | 128.0 / 128 / 128 | — | — |
| [R100](#r100) SparseVLM v1 rank + recycling | cogact | 114.7 / 113 / 118 | — | — |
| [R101](#r101) ASR deletion (all refresh cost) | openvla_oft | 294.8 / 282 / 512 | — | — |
| [R104](#r104) DART token reduction | openvla_oft | 260.0 / 260 / 260 | — | — |
| [R105](#r105) DepthCache primary-view core | openvla_oft | 508.2 / 482 / 512 | — | — |
| [R106](#r106) DivPrune | openvla_oft | 256.0 / 256 / 256 | — | — |
| [R107](#r107) FastV released last-query | openvla_oft | 256.0 / 256 / 256 | — | — |
| [R108](#r108) SparseVLM v1 rank + recycling | openvla_oft | 228.0 / 226 / 230 | — | — |
| [R109](#r109) ADP | openvla_oft | 458.7 / 384 / 512 | — | pruned 90/216 |
| [R112](#r112) SAFE-Pruner | openvla_oft | 170.7 / 128 / 512 | — | keyframe 24/216 |
| [R113](#r113) SpecPrune-VLA (token batch 1) | openvla_oft | 68.5 / 60 / 108 | — | — |
| [R114](#r114) TEAM-VLA (paper-based) | openvla_oft | 80.0 / 80 / 80 | 269.1 | — |
| [R115](#r115) VLA-IAP (paper-based) | openvla_oft | 438.0 / 174 / 512 | — | warmup 24/216 |
| [R116](#r116) VLA-Pruner | openvla_oft | 192.0 / 128 / 512 | — | — |

## Auxiliary dense-refresh work

Main replay rows include ALL auxiliary work. Primary-forward diagnostics exclude
shared preparation and are not end-to-end serving latency. Refresh timing does
not establish real-robot deadline feasibility or reduced total computation.

| Record / backend | Served / auxiliary calls | Primary forward mean ms | Refresh mean / max ms | Auxiliary visual tokens |
|---|---:|---:|---:|---:|
| R57 / cogact | 321 / 321 | 137.92 | 204.53 / 220.81 | 256.0 |
| R58 / cogact | 321 / 321 | 138.77 | 200.75 / 217.24 | 256.0 |
| R59 / cogact | 321 / 321 | 149.00 | 201.40 / 216.91 | 256.0 |
| R93 / cogact | 321 / 321 | 112.31 | 186.24 / 191.00 | 256.0 |
| R101 / openvla_oft | 216 / 216 | 162.33 | 179.93 / 201.00 | 512.0 |

## Computed visual queries and retained KV context

Query-weighted last-layer counts; skipping query/MLP rows does not remove KV context.

| Record / method | Backend | Computed queries: mean / min / max | Retained KV: mean / min / max |
|---|---|---:|---:|
| [R16](#r16) VLA-Cache full statistics eager / CogACT | cogact | 162.4 / 145 / 256 | 256.0 / 256 / 256 |
| [R17](#r17) VLA-Cache full statistics SDPA / CogACT | cogact | 162.4 / 145 / 256 | 256.0 / 256 / 256 |
| [R25](#r25) VLA-InfoEntropy / CogACT full statistics | cogact | 162.6 / 141 / 256 | 256.0 / 256 / 256 |

## Timing quality controls

Independent idle GPUs and fixed CPU affinity; no simulator runs on the timing host.
Native rechecks diagnose drift, not alternative denominators selected by speed.
Acceptance requires no observed thermal flags and absolute native drift at most 3%.

| Cohort | Reference / recheck | Native drift | Thermal flags / samples | Accepted |
|---|---|---:|---:|---|
| cogact_a | [R42](#r42) / [R43](#r43) | +0.04% | 0 / 392 | Yes |
| cogact_b_clean_b | [R52](#r52) / [R53](#r53) | -0.20% | 0 / 202 | Yes |
| cogact_b_stable | [R60](#r60) / [R61](#r61) | +0.01% | 0 / 271 | Yes |
| openvla_oft_a_stable2 | [R63](#r63) / [R64](#r64) | +0.54% | 0 / 123 | Yes |
| openvla_oft_b_stable | [R71](#r71) / [R72](#r72) | +0.69% | 0 / 127 | Yes |

Initial thermal-affected cohorts excluded from current speed joins: `cogact_b`, `cogact_b_clean_a`, `cogact_b_retime`, `cogact_b_retime2`, `openvla_oft_a`, `openvla_oft_a_clean`, `openvla_oft_a_retime`, `openvla_oft_b`, `openvla_oft_b_retime`, `openvla_oft_b_retime2`.
Their raw logs and incomplete attempts remain archived; no rollout outcome
was retried for this reason.

## Measurement cohorts

Each row uses its own paired baseline, never another cohort's denominator.
F = measured control frames; R = repeats; W = warmup queries.

| Cohort | Backend | Hardware | F / R / W | Source fingerprint | Checkpoint |
|---|---|---|---:|---|---|
| C01 | openvla_oft | NVIDIA GeForce RTX 3090 | 1704 / 3 / 10 | `8d179bafa867453407239838c7e82460837d3032bcedf549cc181385c3fd858c` | `b127ee4bbf88c6e61364da6af2df22efe10363140092c52d54d34f8dbd918fe1` |
| C02 | cogact | NVIDIA GeForce RTX 3090 | 2523 / 3 / 10 | `b4cbf022759a0ec4de476f24ee77efa21f1df093d410d467bbc98e763131734f` | `c8aa6908079a1e83fde009bfe5d50bc755aab2e83f160ea4e372f4a8ddc7fc9c` |
| C03 | cogact | NVIDIA GeForce RTX 3090 | 2523 / 3 / 10 | `e5ebd7de5905a6d7c9d729f0e0b67ad8fb882febc051e9a2753e0c9fbdb1e9ed` | `c8aa6908079a1e83fde009bfe5d50bc755aab2e83f160ea4e372f4a8ddc7fc9c` |
| C04 | cogact | NVIDIA GeForce RTX 3090 | 2523 / 3 / 10 | `d9505c5563884804dca293748aef8871017ace1e534cc9378cb9d59e1ec3e8ee` | `c8aa6908079a1e83fde009bfe5d50bc755aab2e83f160ea4e372f4a8ddc7fc9c` |
| C05 | cogact | NVIDIA GeForce RTX 3090 | 2523 / 3 / 10 | `dc290916d485bb0f7d3607956ee7075aa0f500d8fad11db0501315d03cd618c8` | `c8aa6908079a1e83fde009bfe5d50bc755aab2e83f160ea4e372f4a8ddc7fc9c` |
| C06 | cogact | NVIDIA GeForce RTX 3090 | 2523 / 3 / 10 | `dc290916d485bb0f7d3607956ee7075aa0f500d8fad11db0501315d03cd618c8` | `c8aa6908079a1e83fde009bfe5d50bc755aab2e83f160ea4e372f4a8ddc7fc9c` |
| C07 | cogact | NVIDIA GeForce RTX 3090 | 2523 / 3 / 10 | `dc290916d485bb0f7d3607956ee7075aa0f500d8fad11db0501315d03cd618c8` | `c8aa6908079a1e83fde009bfe5d50bc755aab2e83f160ea4e372f4a8ddc7fc9c` |
| C08 | openvla_oft | NVIDIA GeForce RTX 3090 | 1704 / 3 / 10 | `86810c762e0810442533cb422acdea45a9557d40415d30d49b88f7135c23c3e7` | `b127ee4bbf88c6e61364da6af2df22efe10363140092c52d54d34f8dbd918fe1` |
| C09 | cogact | NVIDIA GeForce RTX 3090 | 2523 / 3 / 10 | `2a2f254428f153f7e5c5a5203c8890ce387de9a4bffc697156760c233a079a98` | `c8aa6908079a1e83fde009bfe5d50bc755aab2e83f160ea4e372f4a8ddc7fc9c` |
| C10 | cogact | NVIDIA GeForce RTX 3090 | 2523 / 3 / 10 | `9d6b2f4d60074dbc20459d75e5c5b230c3d6eb773a998b3e16483547e4de5dca` | `c8aa6908079a1e83fde009bfe5d50bc755aab2e83f160ea4e372f4a8ddc7fc9c` |
| C11 | openvla_oft | NVIDIA GeForce RTX 3090 | 1704 / 3 / 10 | `9d6b2f4d60074dbc20459d75e5c5b230c3d6eb773a998b3e16483547e4de5dca` | `b127ee4bbf88c6e61364da6af2df22efe10363140092c52d54d34f8dbd918fe1` |
| C12 | openvla_oft | NVIDIA GeForce RTX 3090 | 1704 / 3 / 10 | `1ca39ee30a256e6b873adad4b37f284cb2e2cc93cb0ed1b36ff7136f60677cf8` | `b127ee4bbf88c6e61364da6af2df22efe10363140092c52d54d34f8dbd918fe1` |
| C13 | cogact | NVIDIA GeForce RTX 3090 | 2523 / 3 / 10 | `9ad6a3939f88440d04d3ba9a9b682218de7e5e2c6889f16ddeceb4826f614f57` | `c8aa6908079a1e83fde009bfe5d50bc755aab2e83f160ea4e372f4a8ddc7fc9c` |
| C14 | openvla_oft | NVIDIA GeForce RTX 3090 | 1704 / 3 / 10 | `9ad6a3939f88440d04d3ba9a9b682218de7e5e2c6889f16ddeceb4826f614f57` | `b127ee4bbf88c6e61364da6af2df22efe10363140092c52d54d34f8dbd918fe1` |
| C15 | openvla_oft | NVIDIA GeForce RTX 3090 | 1704 / 3 / 10 | `f1ac75f63d548db2c37f959f18d83bc5667500cba54122c1979468e88f307ab0` | `b127ee4bbf88c6e61364da6af2df22efe10363140092c52d54d34f8dbd918fe1` |

<details>
<summary>Run identities, presets and local-record checksums</summary>

Raw runs: `outputs/<run ID>/`; exported records: `outputs/result_records/<run ID>.json`.
The SHA256 identifies the full canonical JSON, which retains raw-file hashes,
parameters, baseline, runtime/precision and per-episode smoke outcomes.
The preset links are current. Timing revisions show original → cleaned Git ID.
The [commit map](../scripts/history_commit_map.json) resolves pre-cleanup IDs;
a removed documentation-only commit maps to its unchanged-source ancestor.
This is source lookup, not a new measurement: original record hashes, source
fingerprints and dirty flags are unchanged. Dirty runs need their original
working-tree evidence; checking out the mapped commit alone is insufficient.
Smoke-source labels: same / different / unknown / not run. Earlier smoke is
never promoted to a final-source or formal evaluation.

| Record | Run ID | Cohort | Preset | Timing commit: original → cleaned | Baseline | Smoke source | Record SHA256 |
|---|---|---|---|---|---|---|---|
| <a id="r01"></a>R01 | `batch2-final-benchmark-baseline-001` | C01 | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `01bf64629fd2db27edef978ffa76d9ec6ed0ef9d` → `d17dabf13d11112d7bf26d06f794e38bea10314e` | R01 | not run | `1a4a0a8b39ef12b830893c1531bfe016ebbcd342fd7fed50ef16be87be4eab67` |
| <a id="r02"></a>R02 | `batch2-final-benchmark-baseline_recheck-001` | C01 | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `01bf64629fd2db27edef978ffa76d9ec6ed0ef9d` → `d17dabf13d11112d7bf26d06f794e38bea10314e` | R01 | not run | `cb9f68ed3f840ea9843f56c02c027d61b83d404459b0039d016c57515f7a40ee` |
| <a id="r03"></a>R03 | `batch2-final-benchmark-efficient_vla-001` | C01 | [efficient_vla](../configs/experiments/openvla_oft/efficient_vla.yaml) | `01bf64629fd2db27edef978ffa76d9ec6ed0ef9d` → `d17dabf13d11112d7bf26d06f794e38bea10314e` | R01 | different | `903a047607388c797f10dd279ec97cb61cf2c2aa7fad2ecd74df5027c1a436fb` |
| <a id="r04"></a>R04 | `batch2-final-benchmark-flash_vla-001` | C01 | [flash_vla](../configs/experiments/openvla_oft/flash_vla.yaml) | `01bf64629fd2db27edef978ffa76d9ec6ed0ef9d` → `d17dabf13d11112d7bf26d06f794e38bea10314e` | R01 | different | `fe5edc1c157bb9e3274011309dcb858323dee6db4f4d423e8a00ca7ce1557934` |
| <a id="r05"></a>R05 | `batch2-final-benchmark-sp_vla-001` | C01 | [sp_vla](../configs/experiments/openvla_oft/sp_vla.yaml) | `01bf64629fd2db27edef978ffa76d9ec6ed0ef9d` → `d17dabf13d11112d7bf26d06f794e38bea10314e` | R01 | different | `a1495e644469ba669444d321923077cc4f38ca62a9561abf9c706a1b773deed6` |
| <a id="r06"></a>R06 | `batch2-final-benchmark-sp_vla_scheduling-001` | C01 | [sp_vla_scheduling](../configs/experiments/openvla_oft/sp_vla_scheduling.yaml) | `01bf64629fd2db27edef978ffa76d9ec6ed0ef9d` → `d17dabf13d11112d7bf26d06f794e38bea10314e` | R01 | not run | `b86562025ba0d8f6cdcdbf65edecece0dba652dbf810215228cde8abe7acc7e0` |
| <a id="r07"></a>R07 | `batch2-final-benchmark-st_merge-001` | C01 | [st_merge](../configs/experiments/openvla_oft/st_merge.yaml) | `01bf64629fd2db27edef978ffa76d9ec6ed0ef9d` → `d17dabf13d11112d7bf26d06f794e38bea10314e` | R01 | different | `7b11ea6a370d84f3b48e720f50be4dd8f997ce54bd498f00025ca4e82108bd87` |
| <a id="r08"></a>R08 | `batch2-final-benchmark-vla_cache-001` | C01 | [vla_cache](../configs/experiments/openvla_oft/vla_cache.yaml) | `01bf64629fd2db27edef978ffa76d9ec6ed0ef9d` → `d17dabf13d11112d7bf26d06f794e38bea10314e` | R01 | not run | `9a3d77d7648bf8abad73efc28a1a9472b4baf9dc399d2245cf3dc6481b1c6e2b` |
| <a id="r09"></a>R09 | `batch2-final-benchmark-vla_infoentropy-001` | C01 | [vla_infoentropy](../configs/experiments/openvla_oft/vla_infoentropy.yaml) | `01bf64629fd2db27edef978ffa76d9ec6ed0ef9d` → `d17dabf13d11112d7bf26d06f794e38bea10314e` | R01 | different | `9059b709cd8c5ea40bbee8ee3467cfd19e10dae004058a225ddd1496e55343e8` |
| <a id="r10"></a>R10 | `cogact-category-final-awq-001` | C02 | [awq](../configs/experiments/cogact/awq.yaml) | `ba55950deca55b2f56c946948d315fd2d84031ba` → `94949299c0eb723c880d30a79e1dbfc37c51bb8f` | R12 | different | `9634222837c9f4a5b56e735822cc2310f29d1d6bfdfb64ca39c891c6e00f5d65` |
| <a id="r11"></a>R11 | `cogact-category-final-awq_fp16_kernel-001` | C02 | [awq_fp16_kernel](../configs/experiments/cogact/awq_fp16_kernel.yaml) | `ba55950deca55b2f56c946948d315fd2d84031ba` → `94949299c0eb723c880d30a79e1dbfc37c51bb8f` | R12 | different | `09ef55635c0eaf4754fd2b5dfd4164bb3ac45b14cb9218a4e93ba8e70eae22f8` |
| <a id="r12"></a>R12 | `cogact-category-final-baseline-001` | C02 | [baseline](../configs/experiments/cogact/baseline.yaml) | `ba55950deca55b2f56c946948d315fd2d84031ba` → `94949299c0eb723c880d30a79e1dbfc37c51bb8f` | R12 | different | `f472eb7876f494ce3c0c597d848b6677db7c9d0266c8b5a45e80a74fe36fb268` |
| <a id="r13"></a>R13 | `cogact-category-final-baseline_horizon1-001` | C02 | [baseline_horizon1](../configs/experiments/cogact/baseline_horizon1.yaml) | `ba55950deca55b2f56c946948d315fd2d84031ba` → `94949299c0eb723c880d30a79e1dbfc37c51bb8f` | R13 | different | `4b2f45f7ac3d8b0ff318d0409c4335f1454fb971bf63220a528f747e45bfbe92` |
| <a id="r14"></a>R14 | `cogact-category-final-baseline_recheck-001` | C02 | [baseline](../configs/experiments/cogact/baseline.yaml) | `ba55950deca55b2f56c946948d315fd2d84031ba` → `94949299c0eb723c880d30a79e1dbfc37c51bb8f` | R12 | not run | `5fdb0c317370235bb24211600bc5372414828d0746500fce0184d5aaf9cfccd9` |
| <a id="r15"></a>R15 | `cogact-category-final-sp_vla-001` | C02 | [sp_vla](../configs/experiments/cogact/sp_vla.yaml) | `ba55950deca55b2f56c946948d315fd2d84031ba` → `94949299c0eb723c880d30a79e1dbfc37c51bb8f` | R12 | different | `e7b15f41e8ce983a77416c0e5e4b94e671b6a03bb18390e77c6a77c6feeedf10` |
| <a id="r16"></a>R16 | `cogact-category-final-vla_cache-001` | C02 | [vla_cache](../configs/experiments/cogact/vla_cache.yaml) | `ba55950deca55b2f56c946948d315fd2d84031ba` → `94949299c0eb723c880d30a79e1dbfc37c51bb8f` | R12 | different | `cdb080cf51322720c5cd5848e2fd53f9a95990d3b787ce0829215ccb736868a6` |
| <a id="r17"></a>R17 | `cogact-category-final-vla_cache_sdpa-001` | C02 | [vla_cache_sdpa](../configs/experiments/cogact/vla_cache_sdpa.yaml) | `ba55950deca55b2f56c946948d315fd2d84031ba` → `94949299c0eb723c880d30a79e1dbfc37c51bb8f` | R12 | not run | `50ba76327548d5e6d8eaeb783ba260fbd16f3f36a2ff91316cc30b311e74800e` |
| <a id="r18"></a>R18 | `cogact-category-final-vote_ensemble-001` | C02 | [vote_ensemble](../configs/experiments/cogact/vote_ensemble.yaml) | `ba55950deca55b2f56c946948d315fd2d84031ba` → `94949299c0eb723c880d30a79e1dbfc37c51bb8f` | R13 | different | `36344f4cd135853b4a2fa7a42afe1298f3ff7deb3831ecdbfe2cd596ab94d384` |
| <a id="r19"></a>R19 | `cogact-completion-benchmark-baseline-001` | C03 | [baseline](../configs/experiments/cogact/baseline.yaml) | `9daa44b6b82d3b55ffbca9267790d52cdf32fe3e` | R19 | same | `d030aa13ddf8daf05bd425848bf3a54fbb921452e0554615bc91c5043d659236` |
| <a id="r20"></a>R20 | `cogact-completion-benchmark-baseline_recheck-001` | C03 | [baseline](../configs/experiments/cogact/baseline.yaml) | `9daa44b6b82d3b55ffbca9267790d52cdf32fe3e` | R19 | not run | `9d6db47e640528d775a2dd3b7942abf4bfff81f9cb13f33e3bf0d2da89dd657d` |
| <a id="r21"></a>R21 | `cogact-completion-benchmark-efficient_vla_diffusion_cache-001` | C03 | [efficient_vla_diffusion_cache](../configs/experiments/cogact/efficient_vla_diffusion_cache.yaml) | `9daa44b6b82d3b55ffbca9267790d52cdf32fe3e` | R19 | same | `6f550714db6e13f6d6198759ac95852bd7d76d799a84254ea94f45128ed772a4` |
| <a id="r22"></a>R22 | `cogact-completion-benchmark-safe_pruner-001` | C03 | [safe_pruner](../configs/experiments/cogact/safe_pruner.yaml) | `9daa44b6b82d3b55ffbca9267790d52cdf32fe3e` | R19 | same | `5cbf780ae8ffb446a43ed651c1ba261d3aeba4a23be86aecd83a8e03dd0d8e5d` |
| <a id="r23"></a>R23 | `cogact-completion-benchmark-specprune_vla-001` | C03 | [specprune_vla](../configs/experiments/cogact/specprune_vla.yaml) | `9daa44b6b82d3b55ffbca9267790d52cdf32fe3e` | R19 | same | `ef75f46ab81f2f629311fde3fe7d3ce2ffd7e04b273b2728139929f53eda9a4f` |
| <a id="r24"></a>R24 | `cogact-completion-benchmark-st_merge-001` | C03 | [st_merge](../configs/experiments/cogact/st_merge.yaml) | `9daa44b6b82d3b55ffbca9267790d52cdf32fe3e` | R19 | same | `6261cf6a24ea040d06f29d8c3b86406b003800e52fbe75e077841d9a3837cfab` |
| <a id="r25"></a>R25 | `cogact-completion-benchmark-vla_infoentropy-001` | C03 | [vla_infoentropy](../configs/experiments/cogact/vla_infoentropy.yaml) | `9daa44b6b82d3b55ffbca9267790d52cdf32fe3e` | R19 | same | `8e6fdacbf87268b18c5576a15ceb95816cfbf5268406a9da3b3044cc795e3a1a` |
| <a id="r26"></a>R26 | `cogact-final-benchmark-baseline-001` | C04 | [baseline](../configs/experiments/cogact/baseline.yaml) | `a984ef33c463dd2af38be19cb4dee4f8e73a4f56` → `9a1f83ab56f5043851579d40e073b599d216d950` | R26 | same | `d95cfef815a1d179a566d5bd06f1d67043ab5d39af8267e56137f28bc57f8213` |
| <a id="r27"></a>R27 | `cogact-secondary-benchmark-baseline-001` | C04 | [baseline](../configs/experiments/cogact/baseline.yaml) | `a984ef33c463dd2af38be19cb4dee4f8e73a4f56` → `9a1f83ab56f5043851579d40e073b599d216d950` | R27 | same | `19e73138b950fe69654c83d6169a8525d7c86247b505976fcc90c8fdc2c70095` |
| <a id="r28"></a>R28 | `cogact-secondary-benchmark-recheck-001` | C04 | [baseline](../configs/experiments/cogact/baseline.yaml) | `a984ef33c463dd2af38be19cb4dee4f8e73a4f56` → `9a1f83ab56f5043851579d40e073b599d216d950` | R28 | same | `8f4b6c4d743d29ae60261d15c7524cb3c48e254e8880e7a357521661d8fe31f8` |
| <a id="r29"></a>R29 | `cogact-token-batch1-benchmark-adp-001` | C05 | [adp](../configs/experiments/cogact/adp.yaml) | `befad9866515af21713435bdc69f5ddde8b0592d` (dirty) → `c124d93b20da46f17666b0090003390367f3214a` | R30 | same | `0d193f6a157cb83fb6d652d88367bf8a7d9a7dce2b96993ba134f324b67071df` |
| <a id="r30"></a>R30 | `cogact-token-batch1-benchmark-baseline-001` | C05 | [baseline](../configs/experiments/cogact/baseline.yaml) | `befad9866515af21713435bdc69f5ddde8b0592d` (dirty) → `c124d93b20da46f17666b0090003390367f3214a` | R30 | same | `406891d6c9c3be93424bdb1bff40be6faad3737617e926510fc27ed18f9acc4a` |
| <a id="r31"></a>R31 | `cogact-token-batch1-benchmark-baseline_recheck-001` | C06 | [baseline](../configs/experiments/cogact/baseline.yaml) | `1d9d98ec3070e1d216738b2978bb7ab825ce015b` (dirty) → `761b7dd690e31f91b3c4aebef107c4c685c69fef` | R30 | not run | `92ca052fe9e981caaebfca1124d0ccab13a2b4722a429d89e3bc7afd8ef7c896` |
| <a id="r32"></a>R32 | `cogact-token-batch1-benchmark-flash_vla-001` | C07 | [flash_vla](../configs/experiments/cogact/flash_vla.yaml) | `b03667ab6362a42ac5215678b8d8829197dcc7a6` (dirty) → `0d6ef73dc3825df99438112713145349446ab2a6` | R30 | same | `9173edd82490a071f88397e3200389e8b3435650f06f8209ee77e5e606a00b48` |
| <a id="r33"></a>R33 | `control-methods-final-benchmark-awq-001` | C08 | [awq](../configs/experiments/openvla_oft/awq.yaml) | `be5be7d8e0e170e1f5956632ce537a3fa93cb653` → `4b3f8b39faab67f36c329a14368c73c7a1d3e176` | R35 | different | `7c45d89c736771315f940f7f90e5167f96df563b15676415340a70bab1a9b82d` |
| <a id="r34"></a>R34 | `control-methods-final-benchmark-awq_fp16_kernel-001` | C08 | [awq_fp16_kernel](../configs/experiments/openvla_oft/awq_fp16_kernel.yaml) | `be5be7d8e0e170e1f5956632ce537a3fa93cb653` → `4b3f8b39faab67f36c329a14368c73c7a1d3e176` | R35 | different | `82803a005adb4c6288906f1d683b6483fa8e446fb842a352400c6ca213fadb7e` |
| <a id="r35"></a>R35 | `control-methods-final-benchmark-baseline-001` | C08 | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `be5be7d8e0e170e1f5956632ce537a3fa93cb653` → `4b3f8b39faab67f36c329a14368c73c7a1d3e176` | R35 | different | `5cb6a9ec850c9f870dd24b7d9d8f46651871d7b403ee604cc9dc2a69a0e40b68` |
| <a id="r36"></a>R36 | `control-methods-final-benchmark-baseline_horizon1-001` | C08 | [baseline_horizon1](../configs/experiments/openvla_oft/baseline_horizon1.yaml) | `be5be7d8e0e170e1f5956632ce537a3fa93cb653` → `4b3f8b39faab67f36c329a14368c73c7a1d3e176` | R36 | not run | `f060e73d87b99b1ba3513e629711e774dad111286f407714c057af702f89a9d1` |
| <a id="r37"></a>R37 | `control-methods-final-benchmark-baseline_recheck-001` | C08 | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `be5be7d8e0e170e1f5956632ce537a3fa93cb653` → `4b3f8b39faab67f36c329a14368c73c7a1d3e176` | R35 | not run | `59d8e88fa26a72997a2cc1f68e1ad47a2f09c7c457be8d38eab276d1c40abbcd` |
| <a id="r38"></a>R38 | `control-methods-final-benchmark-sp_vla_scheduling-001` | C08 | [sp_vla_scheduling](../configs/experiments/openvla_oft/sp_vla_scheduling.yaml) | `be5be7d8e0e170e1f5956632ce537a3fa93cb653` → `4b3f8b39faab67f36c329a14368c73c7a1d3e176` | R35 | different | `a0273879758dae45be9f1842c1075c84ed1f2b62af0e85b77b26688c0cc813d1` |
| <a id="r39"></a>R39 | `control-methods-final-benchmark-specprune_vla-001` | C08 | [specprune_vla](../configs/experiments/openvla_oft/specprune_vla.yaml) | `be5be7d8e0e170e1f5956632ce537a3fa93cb653` → `4b3f8b39faab67f36c329a14368c73c7a1d3e176` | R35 | different | `d0cf29382c0b1da4527e97b3774cfeed605d59dbde35233d3250eccbc9b81fba` |
| <a id="r40"></a>R40 | `control-methods-final-benchmark-vla_cache-001` | C08 | [vla_cache](../configs/experiments/openvla_oft/vla_cache.yaml) | `be5be7d8e0e170e1f5956632ce537a3fa93cb653` → `4b3f8b39faab67f36c329a14368c73c7a1d3e176` | R35 | different | `8b7c57663e402e87953bc434a0f0fc3d21ac0234d8909bddd825e499ee9a24b1` |
| <a id="r41"></a>R41 | `control-methods-final-benchmark-vote_ensemble-001` | C08 | [vote_ensemble](../configs/experiments/openvla_oft/vote_ensemble.yaml) | `be5be7d8e0e170e1f5956632ce537a3fa93cb653` → `4b3f8b39faab67f36c329a14368c73c7a1d3e176` | R36 | different | `3784f976868209483ed74c9ec4bd1f4324dff8b2dbf44c0bbe4d13104c763f9b` |
| <a id="r42"></a>R42 | `diagnostic-v1-cogact_a-baseline` | C09 | [baseline](../configs/experiments/cogact/baseline.yaml) | `606f6350bf6c5b04521cee707004ef641ce24280` | R42 | not run | `f6b911cd160b7fc257f420051e40c863d5d447d3352df98a1046b93a9d295508` |
| <a id="r43"></a>R43 | `diagnostic-v1-cogact_a-baseline-recheck` | C09 | [baseline](../configs/experiments/cogact/baseline.yaml) | `606f6350bf6c5b04521cee707004ef641ce24280` | R42 | not run | `6e3ea5cc73235dd67cf942301c0f332561c023e7b2e39f6cbc268eaaedd017de` |
| <a id="r44"></a>R44 | `diagnostic-v1-cogact_a-flash_vla` | C09 | [flash_vla](../configs/experiments/cogact/flash_vla.yaml) | `606f6350bf6c5b04521cee707004ef641ce24280` | R42 | not run | `ca0d7fd468dbbb6f615170ff62f167957e3234eba8e6d1417e9dc84eb63bc2f3` |
| <a id="r45"></a>R45 | `diagnostic-v1-cogact_a-flash_vla_keep192` | C09 | [flash_vla_keep192](../configs/experiments/cogact/flash_vla_keep192.yaml) | `606f6350bf6c5b04521cee707004ef641ce24280` | R42 | not run | `39e18ef23d3b4883f7598b0c6e8632454e56dc55c1232a6366c63b465ff13b5c` |
| <a id="r46"></a>R46 | `diagnostic-v1-cogact_a-specprune_vla` | C09 | [specprune_vla](../configs/experiments/cogact/specprune_vla.yaml) | `606f6350bf6c5b04521cee707004ef641ce24280` | R42 | not run | `8589a0521ae55bbee19a1330d85e2a138fb9420779f755484a7322ec38a79c1e` |
| <a id="r47"></a>R47 | `diagnostic-v1-cogact_a-specprune_vla_score_first` | C09 | [specprune_vla_score_first](../configs/experiments/cogact/specprune_vla_score_first.yaml) | `606f6350bf6c5b04521cee707004ef641ce24280` | R42 | not run | `39893bcac8b027036ba98c719291eb64848daaf2b097b875de63d6d37fc9a7a0` |
| <a id="r48"></a>R48 | `diagnostic-v1-cogact_a-specprune_vla_static128` | C09 | [specprune_vla_static128](../configs/experiments/cogact/specprune_vla_static128.yaml) | `606f6350bf6c5b04521cee707004ef641ce24280` | R42 | not run | `c34aa0ba418110e1545fdd59828df3c0001f836abff179534533dc7617ef18a1` |
| <a id="r49"></a>R49 | `diagnostic-v1-cogact_a-st_merge` | C09 | [st_merge](../configs/experiments/cogact/st_merge.yaml) | `606f6350bf6c5b04521cee707004ef641ce24280` | R42 | not run | `573e9f9163b73cb56aaba856e6ea37220c5069fe040a7d1e16ea130ad641beaf` |
| <a id="r50"></a>R50 | `diagnostic-v1-cogact_a-st_merge_keep192` | C09 | [st_merge_keep192](../configs/experiments/cogact/st_merge_keep192.yaml) | `606f6350bf6c5b04521cee707004ef641ce24280` | R42 | not run | `4ee4f1e72768bf7c1b6eb64de4d48d39f863d48aa65e7b91bf99dc9a2fe430be` |
| <a id="r51"></a>R51 | `diagnostic-v1-cogact_a-st_merge_layer6` | C09 | [st_merge_layer6](../configs/experiments/cogact/st_merge_layer6.yaml) | `606f6350bf6c5b04521cee707004ef641ce24280` | R42 | not run | `a5bf48183517c596ed85db179ab8e17dfd68ef1fd6fb6bde9a1f11e1665148e9` |
| <a id="r52"></a>R52 | `diagnostic-v1-cogact_b_clean_b-baseline` | C09 | [baseline](../configs/experiments/cogact/baseline.yaml) | `606f6350bf6c5b04521cee707004ef641ce24280` | R52 | not run | `e8320dca8db52431db9ba26b53cde4fc5fcc532aeae27574dff547ae87180527` |
| <a id="r53"></a>R53 | `diagnostic-v1-cogact_b_clean_b-baseline-recheck` | C09 | [baseline](../configs/experiments/cogact/baseline.yaml) | `606f6350bf6c5b04521cee707004ef641ce24280` | R52 | not run | `f7aa6a3902251d96ea5150bdbb3d929463b0d98262b9cb18d2904f551baf9f4b` |
| <a id="r54"></a>R54 | `diagnostic-v1-cogact_b_clean_b-efficient_vla_cache2` | C09 | [efficient_vla_cache2](../configs/experiments/cogact/efficient_vla_cache2.yaml) | `606f6350bf6c5b04521cee707004ef641ce24280` | R52 | not run | `217ef6dfb48bbaef79db3fa100509f259975a82ae854ed055ed8c3301a08cf81` |
| <a id="r55"></a>R55 | `diagnostic-v1-cogact_b_clean_b-efficient_vla_cache3` | C09 | [efficient_vla_cache3](../configs/experiments/cogact/efficient_vla_cache3.yaml) | `606f6350bf6c5b04521cee707004ef641ce24280` | R52 | not run | `379bd3a21150e1df706f25ec9bc64f7bb91faf9101799d70aa521d9590506086` |
| <a id="r56"></a>R56 | `diagnostic-v1-cogact_b_clean_b-efficient_vla_diffusion_cache` | C09 | [efficient_vla_diffusion_cache](../configs/experiments/cogact/efficient_vla_diffusion_cache.yaml) | `606f6350bf6c5b04521cee707004ef641ce24280` | R52 | not run | `be95d6bf37bab85b653dab69a7a1f00a219d265fb7e577b557bb61d7ef3894b7` |
| <a id="r57"></a>R57 | `diagnostic-v1-cogact_b_stable-actuation_slack_refresh_deletion` | C10 | [actuation_slack_refresh_deletion](../configs/experiments/cogact/actuation_slack_refresh_deletion.yaml) | `ea3669a23b058e588d91b5f5be38ffb64aa762e6` | R60 | not run | `86dc87657fbc64dc053db2b932ded971fe4942edcc896276ff577b49b186d874` |
| <a id="r58"></a>R58 | `diagnostic-v1-cogact_b_stable-actuation_slack_refresh_keep25` | C10 | [actuation_slack_refresh_keep25](../configs/experiments/cogact/actuation_slack_refresh_keep25.yaml) | `ea3669a23b058e588d91b5f5be38ffb64aa762e6` | R60 | not run | `4a7323c40c1e649465ce12c3efa03a5b7b83521cde78fb59909faf94f4340998` |
| <a id="r59"></a>R59 | `diagnostic-v1-cogact_b_stable-actuation_slack_refresh_keep50` | C10 | [actuation_slack_refresh_keep50](../configs/experiments/cogact/actuation_slack_refresh_keep50.yaml) | `ea3669a23b058e588d91b5f5be38ffb64aa762e6` | R60 | not run | `999fb8cca9c9f522633f933770b7454c833ff53946f7cd5a78df68579c24005d` |
| <a id="r60"></a>R60 | `diagnostic-v1-cogact_b_stable-baseline` | C10 | [baseline](../configs/experiments/cogact/baseline.yaml) | `ea3669a23b058e588d91b5f5be38ffb64aa762e6` | R60 | not run | `3e821c7afcce5fb83b27ec0ee3b1dd4c8de15c0869c66c95f2b1a6106c8ece9a` |
| <a id="r61"></a>R61 | `diagnostic-v1-cogact_b_stable-baseline-recheck` | C10 | [baseline](../configs/experiments/cogact/baseline.yaml) | `ea3669a23b058e588d91b5f5be38ffb64aa762e6` | R60 | not run | `a5db179c964a080c025ca9455f7185ce572f09df1cf79259fe3e7e14f12256b1` |
| <a id="r62"></a>R62 | `diagnostic-v1-cogact_b_stable-flash_vla_layer4` | C10 | [flash_vla_layer4](../configs/experiments/cogact/flash_vla_layer4.yaml) | `ea3669a23b058e588d91b5f5be38ffb64aa762e6` | R60 | not run | `38bb80b94ac036f9ef6ac91c259cea8be999bf338d0d481e7e964384b9ab99e7` |
| <a id="r63"></a>R63 | `diagnostic-v1-openvla_oft_a_stable2-baseline` | C11 | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `ea3669a23b058e588d91b5f5be38ffb64aa762e6` | R63 | not run | `d8a812b0cf7cc94ec6cf731c2ca150be4c02da80fcce28dc2c049bea5e4034a3` |
| <a id="r64"></a>R64 | `diagnostic-v1-openvla_oft_a_stable2-baseline-recheck` | C11 | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `ea3669a23b058e588d91b5f5be38ffb64aa762e6` | R63 | not run | `0d591de326ef96a0d85bdf961398062244cbd9378464783d7deeb9b483df3638` |
| <a id="r65"></a>R65 | `diagnostic-v1-openvla_oft_a_stable2-st_merge` | C11 | [st_merge](../configs/experiments/openvla_oft/st_merge.yaml) | `ea3669a23b058e588d91b5f5be38ffb64aa762e6` | R63 | not run | `60c5bf49415e7874e6290fbbc3144854d258b8c5fb9c2db53f336d3fba11a429` |
| <a id="r66"></a>R66 | `diagnostic-v1-openvla_oft_a_stable2-st_merge_keep192` | C11 | [st_merge_keep192](../configs/experiments/openvla_oft/st_merge_keep192.yaml) | `ea3669a23b058e588d91b5f5be38ffb64aa762e6` | R63 | not run | `a71f9472bf441a8f834aef1b5aa5791e12a760ccd8b5f4c9820eb5b2b6964999` |
| <a id="r67"></a>R67 | `diagnostic-v1-openvla_oft_a_stable2-st_merge_layer6` | C11 | [st_merge_layer6](../configs/experiments/openvla_oft/st_merge_layer6.yaml) | `ea3669a23b058e588d91b5f5be38ffb64aa762e6` | R63 | not run | `eb7054ec9504e448cb966fae5bb76cf3710b145bddc01ae53565d7382016c9a7` |
| <a id="r68"></a>R68 | `diagnostic-v1-openvla_oft_a_stable2-team_vla` | C11 | [team_vla](../configs/experiments/openvla_oft/team_vla.yaml) | `ea3669a23b058e588d91b5f5be38ffb64aa762e6` | R63 | not run | `61b9a8c025adbaa405336bebe1ea21fdd052151b34efb2e2e35f3746d0e6f20b` |
| <a id="r69"></a>R69 | `diagnostic-v1-openvla_oft_a_stable2-team_vla_context50` | C11 | [team_vla_context50](../configs/experiments/openvla_oft/team_vla_context50.yaml) | `ea3669a23b058e588d91b5f5be38ffb64aa762e6` | R63 | not run | `b26b42a128d3567d5ff383fd2943956fc1f8912e00da473442e96e9d525c6e4c` |
| <a id="r70"></a>R70 | `diagnostic-v1-openvla_oft_a_stable2-team_vla_merge160` | C11 | [team_vla_merge160](../configs/experiments/openvla_oft/team_vla_merge160.yaml) | `ea3669a23b058e588d91b5f5be38ffb64aa762e6` | R63 | not run | `49379b2cd5148446bf0f8e14ecb8365a68b281c9748a3ff76ced7c3ac94a3490` |
| <a id="r71"></a>R71 | `diagnostic-v1-openvla_oft_b_stable-baseline` | C11 | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `ea3669a23b058e588d91b5f5be38ffb64aa762e6` | R71 | not run | `40b86c8f80d41427cf06753e622415626c1f83399354bf3bfdecd1d4d82ecdc5` |
| <a id="r72"></a>R72 | `diagnostic-v1-openvla_oft_b_stable-baseline-recheck` | C11 | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `ea3669a23b058e588d91b5f5be38ffb64aa762e6` | R71 | not run | `524e8bbcc2ef23124fda84ec3ae9db4c8fe0370f35766b2c505634e5a82e30f5` |
| <a id="r73"></a>R73 | `diagnostic-v1-openvla_oft_b_stable-dart` | C11 | [dart](../configs/experiments/openvla_oft/dart.yaml) | `ea3669a23b058e588d91b5f5be38ffb64aa762e6` | R71 | not run | `938e049169d88f61939bb24d8dc008d31ae0238cd5d928ed21b9b6acc83a0ed0` |
| <a id="r74"></a>R74 | `diagnostic-v1-openvla_oft_b_stable-dart_keep75` | C11 | [dart_keep75](../configs/experiments/openvla_oft/dart_keep75.yaml) | `ea3669a23b058e588d91b5f5be38ffb64aa762e6` | R71 | not run | `6b5391ef80eafa384b3c0dac8380478916edf8b7584a33da8c8ea9bed3cd1c41` |
| <a id="r75"></a>R75 | `diagnostic-v1-openvla_oft_b_stable-dart_layer4` | C11 | [dart_layer4](../configs/experiments/openvla_oft/dart_layer4.yaml) | `ea3669a23b058e588d91b5f5be38ffb64aa762e6` | R71 | not run | `d854b5405b8b44d26abced02f587836c8868d58b2a91374bc9f0f80c4bed7398` |
| <a id="r76"></a>R76 | `diagnostic-v1-openvla_oft_b_stable-specprune_vla` | C11 | [specprune_vla](../configs/experiments/openvla_oft/specprune_vla.yaml) | `ea3669a23b058e588d91b5f5be38ffb64aa762e6` | R71 | not run | `b396744afe10e3ebca21dc99d890fe8b410759a59dfcb0ae6c36e55e94f59cf0` |
| <a id="r77"></a>R77 | `diagnostic-v1-openvla_oft_b_stable-specprune_vla_score_first` | C11 | [specprune_vla_score_first](../configs/experiments/openvla_oft/specprune_vla_score_first.yaml) | `ea3669a23b058e588d91b5f5be38ffb64aa762e6` | R71 | not run | `a480f4d55a76eeb630a8e222060977fdb856dd3e1230877db9561897acb5fa9a` |
| <a id="r78"></a>R78 | `diagnostic-v1-openvla_oft_b_stable-specprune_vla_static128` | C11 | [specprune_vla_static128](../configs/experiments/openvla_oft/specprune_vla_static128.yaml) | `ea3669a23b058e588d91b5f5be38ffb64aa762e6` | R71 | not run | `0b4b4d7a8b4bcdc6a98019180a15c4c9c16aa6acd4bffccb10d648cd2705db46` |
| <a id="r79"></a>R79 | `fidelity-final-benchmark-adp-001` | C12 | [adp](../configs/experiments/openvla_oft/adp.yaml) | `b2d91319e440d6c3aeaf1ddc2c04a2c1f10c5362` → `d4eeadf8d3fbd1364d47ab9702280cab8769c0d9` | R80 | different | `fc1d8ece0b442050e715565e1f932e9d4ae170b43b2b010cf72c981db5d6b313` |
| <a id="r80"></a>R80 | `fidelity-final-benchmark-baseline-001` | C12 | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `b2d91319e440d6c3aeaf1ddc2c04a2c1f10c5362` → `d4eeadf8d3fbd1364d47ab9702280cab8769c0d9` | R80 | not run | `9e8d1cad9518377fd4b21b84b03ccbb70ebd8ecd7e0addea5ae2c700a5f402a3` |
| <a id="r81"></a>R81 | `fidelity-final-benchmark-baseline_recheck-001` | C12 | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `b2d91319e440d6c3aeaf1ddc2c04a2c1f10c5362` → `d4eeadf8d3fbd1364d47ab9702280cab8769c0d9` | R80 | not run | `0e78732459c98afe86a499171b2e162874b15b54533e8d260df8be4450c71f23` |
| <a id="r82"></a>R82 | `fidelity-final-benchmark-efficient_vla-001` | C12 | [efficient_vla](../configs/experiments/openvla_oft/efficient_vla.yaml) | `b2d91319e440d6c3aeaf1ddc2c04a2c1f10c5362` → `d4eeadf8d3fbd1364d47ab9702280cab8769c0d9` | R80 | different | `49639a395fd6bec2352dbb1608df5fb6accc9f9d003f86f878ea3f68544d33c4` |
| <a id="r83"></a>R83 | `fidelity-final-benchmark-flash_vla-001` | C12 | [flash_vla](../configs/experiments/openvla_oft/flash_vla.yaml) | `b2d91319e440d6c3aeaf1ddc2c04a2c1f10c5362` → `d4eeadf8d3fbd1364d47ab9702280cab8769c0d9` | R80 | different | `321ef6b360eb7fce75f5da6d783d9fea4b2081c808db185d0df4f2d115a92776` |
| <a id="r84"></a>R84 | `fidelity-final-benchmark-flash_vla_gesvd-001` | C12 | [flash_vla_gesvd](../configs/experiments/openvla_oft/flash_vla_gesvd.yaml) | `b2d91319e440d6c3aeaf1ddc2c04a2c1f10c5362` → `d4eeadf8d3fbd1364d47ab9702280cab8769c0d9` | R80 | not run | `c6817118ab0cc9e2d5f0fe7fb016f44ca9c8120026c506e816f73703622b9a15` |
| <a id="r85"></a>R85 | `fidelity-final-benchmark-safe_pruner-001` | C12 | [safe_pruner](../configs/experiments/openvla_oft/safe_pruner.yaml) | `b2d91319e440d6c3aeaf1ddc2c04a2c1f10c5362` → `d4eeadf8d3fbd1364d47ab9702280cab8769c0d9` | R80 | different | `ee1d25419d4ed8ef486fc26b10adf0b7bdf11530ea65b38d189311d17f7baf44` |
| <a id="r86"></a>R86 | `fidelity-final-benchmark-specprune_vla-001` | C12 | [specprune_vla](../configs/experiments/openvla_oft/specprune_vla.yaml) | `b2d91319e440d6c3aeaf1ddc2c04a2c1f10c5362` → `d4eeadf8d3fbd1364d47ab9702280cab8769c0d9` | R80 | different | `7b6b8ba2d827fa403a1bcbee9108cb9a8d14a4e50e6ff2ae80499a0d247bfc05` |
| <a id="r87"></a>R87 | `fidelity-final-benchmark-st_merge-001` | C12 | [st_merge](../configs/experiments/openvla_oft/st_merge.yaml) | `b2d91319e440d6c3aeaf1ddc2c04a2c1f10c5362` → `d4eeadf8d3fbd1364d47ab9702280cab8769c0d9` | R80 | different | `da7829e5a8c2121e36035a1b9e29067c62507861d180a2a3eae635c580ac0009` |
| <a id="r88"></a>R88 | `fidelity-final-benchmark-team_vla-001` | C12 | [team_vla](../configs/experiments/openvla_oft/team_vla.yaml) | `b2d91319e440d6c3aeaf1ddc2c04a2c1f10c5362` → `d4eeadf8d3fbd1364d47ab9702280cab8769c0d9` | R80 | different | `f4dbc822013399631c0353d0a007604afe9101bacbf2f2560968e220d8014882` |
| <a id="r89"></a>R89 | `fidelity-final-benchmark-vla_cache-001` | C12 | [vla_cache](../configs/experiments/openvla_oft/vla_cache.yaml) | `b2d91319e440d6c3aeaf1ddc2c04a2c1f10c5362` → `d4eeadf8d3fbd1364d47ab9702280cab8769c0d9` | R80 | different | `7e24802f0945e179c6b8fc5965c6d5ed7e961c50e1399961a31fa5066a49c943` |
| <a id="r90"></a>R90 | `fidelity-final-benchmark-vla_iap-001` | C12 | [vla_iap](../configs/experiments/openvla_oft/vla_iap.yaml) | `b2d91319e440d6c3aeaf1ddc2c04a2c1f10c5362` → `d4eeadf8d3fbd1364d47ab9702280cab8769c0d9` | R80 | different | `666948372c4174d51e555e819f4e1ef23a658088e44d4b7ef2f3522535a14e40` |
| <a id="r91"></a>R91 | `fidelity-final-benchmark-vla_infoentropy-001` | C12 | [vla_infoentropy](../configs/experiments/openvla_oft/vla_infoentropy.yaml) | `b2d91319e440d6c3aeaf1ddc2c04a2c1f10c5362` → `d4eeadf8d3fbd1364d47ab9702280cab8769c0d9` | R80 | different | `8d7db1528cb12540d7f9bbbe30851bdf49699a4815a1621c938dbfa43fac1536` |
| <a id="r92"></a>R92 | `fidelity-final-benchmark-vla_pruner-001` | C12 | [vla_pruner](../configs/experiments/openvla_oft/vla_pruner.yaml) | `b2d91319e440d6c3aeaf1ddc2c04a2c1f10c5362` → `d4eeadf8d3fbd1364d47ab9702280cab8769c0d9` | R80 | different | `b219ae2d9a9917df9e242566c12d711efa0e15ff8027e57ec9e5547956d37b08` |
| <a id="r93"></a>R93 | `next-batch-final-cog-benchmark-actuation_slack_refresh_deletion-001` | C13 | [actuation_slack_refresh_deletion](../configs/experiments/cogact/actuation_slack_refresh_deletion.yaml) | `fa01bb7dab94af5a79e616e9c4dd0d7e987d4ff7` | R94 | same | `57810e23b550d6929ac94552dccdd62a82774ecfacfdd2980899bacfedd7d777` |
| <a id="r94"></a>R94 | `next-batch-final-cog-benchmark-baseline-001` | C13 | [baseline](../configs/experiments/cogact/baseline.yaml) | `fa01bb7dab94af5a79e616e9c4dd0d7e987d4ff7` | R94 | different | `5b4753b567de37b2a16735517fc2fa522b815da2e07f240cf888bf201be6c1dd` |
| <a id="r95"></a>R95 | `next-batch-final-cog-benchmark-baseline_recheck-001` | C13 | [baseline](../configs/experiments/cogact/baseline.yaml) | `fa01bb7dab94af5a79e616e9c4dd0d7e987d4ff7` | R94 | different | `483898df8f78d16355a6ede6ce982ce82cac71820bcfc753014f52860ec33405` |
| <a id="r96"></a>R96 | `next-batch-final-cog-benchmark-dart-001` | C13 | [dart](../configs/experiments/cogact/dart.yaml) | `fa01bb7dab94af5a79e616e9c4dd0d7e987d4ff7` | R94 | same | `f23588aa76a2addede77e1daaeedc9fe1e00b4c913978daa1afc3450fc8202cd` |
| <a id="r97"></a>R97 | `next-batch-final-cog-benchmark-depth_cache_primary-001` | C13 | [depth_cache_primary](../configs/experiments/cogact/depth_cache_primary.yaml) | `fa01bb7dab94af5a79e616e9c4dd0d7e987d4ff7` | R94 | same | `49b4be997ca1f1222364ac56d4fca8b6ca57dcf05568c6984e1d61644ab3eb6a` |
| <a id="r98"></a>R98 | `next-batch-final-cog-benchmark-divprune-001` | C13 | [divprune](../configs/experiments/cogact/divprune.yaml) | `fa01bb7dab94af5a79e616e9c4dd0d7e987d4ff7` | R94 | same | `77f588c93c604d2aa51bd1c428200c545889f560b44bc1e9dd0ce685fc4ae535` |
| <a id="r99"></a>R99 | `next-batch-final-cog-benchmark-fastv-001` | C13 | [fastv](../configs/experiments/cogact/fastv.yaml) | `fa01bb7dab94af5a79e616e9c4dd0d7e987d4ff7` | R94 | same | `3c704a2589f1487b6da2bd5f42d7854f4b2eb11731fddf477c33715e76109977` |
| <a id="r100"></a>R100 | `next-batch-final-cog-benchmark-sparsevlm-001` | C13 | [sparsevlm](../configs/experiments/cogact/sparsevlm.yaml) | `fa01bb7dab94af5a79e616e9c4dd0d7e987d4ff7` | R94 | same | `75b8133123aef770b4b6afc9c07f7ca07ef0c9075f971aff09aae3bfb9563a1d` |
| <a id="r101"></a>R101 | `next-batch-final-oft-benchmark-actuation_slack_refresh_deletion-001` | C14 | [actuation_slack_refresh_deletion](../configs/experiments/openvla_oft/actuation_slack_refresh_deletion.yaml) | `fa01bb7dab94af5a79e616e9c4dd0d7e987d4ff7` | R102 | different | `20b65ec2a335ac2a80775e870249c66f0e34fd0a78a6dbcfc83e12016ef6ebe4` |
| <a id="r102"></a>R102 | `next-batch-final-oft-benchmark-baseline-001` | C14 | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `fa01bb7dab94af5a79e616e9c4dd0d7e987d4ff7` | R102 | different | `b6a9f67a25440b1384780af8667f72bbb6e7d01c44e4a7fccb72abaf4300faae` |
| <a id="r103"></a>R103 | `next-batch-final-oft-benchmark-baseline_recheck-001` | C14 | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `fa01bb7dab94af5a79e616e9c4dd0d7e987d4ff7` | R102 | different | `f86045e57cce601b2b694db52377534a551e937faf88b63196abd75da283def0` |
| <a id="r104"></a>R104 | `next-batch-final-oft-benchmark-dart-001` | C14 | [dart](../configs/experiments/openvla_oft/dart.yaml) | `fa01bb7dab94af5a79e616e9c4dd0d7e987d4ff7` | R102 | same | `28caabcee0d080d7c67f18d76dfc9b02c9974445b6edc45f6f2dbc6108028ee3` |
| <a id="r105"></a>R105 | `next-batch-final-oft-benchmark-depth_cache_primary-001` | C14 | [depth_cache_primary](../configs/experiments/openvla_oft/depth_cache_primary.yaml) | `fa01bb7dab94af5a79e616e9c4dd0d7e987d4ff7` | R102 | different | `c54b4d4f7407053cb53d8c647ce5dc9ba459fb90f38f5135dea8ffb01efc13fa` |
| <a id="r106"></a>R106 | `next-batch-final-oft-benchmark-divprune-001` | C14 | [divprune](../configs/experiments/openvla_oft/divprune.yaml) | `fa01bb7dab94af5a79e616e9c4dd0d7e987d4ff7` | R102 | same | `2715a88fda0a949bb9ce07040ef33b99221eaf6422e8cf31c46b0c1c63b23293` |
| <a id="r107"></a>R107 | `next-batch-final-oft-benchmark-fastv-001` | C14 | [fastv](../configs/experiments/openvla_oft/fastv.yaml) | `fa01bb7dab94af5a79e616e9c4dd0d7e987d4ff7` | R102 | different | `afcb0c6eeb1aae862deeb45907d97c5c226fdec4e33d293fa12a55eaf246bdcd` |
| <a id="r108"></a>R108 | `next-batch-final-oft-benchmark-sparsevlm-001` | C14 | [sparsevlm](../configs/experiments/openvla_oft/sparsevlm.yaml) | `fa01bb7dab94af5a79e616e9c4dd0d7e987d4ff7` | R102 | same | `a3e61db434b93e0305ea03159b2b1b5b4b24ac88816d778527efe42232d3612c` |
| <a id="r109"></a>R109 | `token-batch-final-benchmark-adp-001` | C15 | [adp](../configs/experiments/openvla_oft/adp.yaml) | `ff880edb449d3840d2a7420e1363d155ac9dfe95` → `7f851f4dbc57f40299d8f0acc3326b41dac20a1d` | R110 | different | `3692e25e645d7b0a7a511f97cbcc5e618124737c2cbc1841d88a8b5560cd166a` |
| <a id="r110"></a>R110 | `token-batch-final-benchmark-baseline-001` | C15 | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `ff880edb449d3840d2a7420e1363d155ac9dfe95` → `7f851f4dbc57f40299d8f0acc3326b41dac20a1d` | R110 | not run | `64a8c6d3d1239176e992d8baab97bb17e54512746d3ecaf22a069f1cb4c37b40` |
| <a id="r111"></a>R111 | `token-batch-final-benchmark-baseline_recheck-001` | C15 | [baseline](../configs/experiments/openvla_oft/baseline.yaml) | `ff880edb449d3840d2a7420e1363d155ac9dfe95` → `7f851f4dbc57f40299d8f0acc3326b41dac20a1d` | R110 | not run | `94e5197bd45df492e23c8692ca77d74cc67b3d53795bc905a554c081c84264f9` |
| <a id="r112"></a>R112 | `token-batch-final-benchmark-safe_pruner-001` | C15 | [safe_pruner](../configs/experiments/openvla_oft/safe_pruner.yaml) | `ff880edb449d3840d2a7420e1363d155ac9dfe95` → `7f851f4dbc57f40299d8f0acc3326b41dac20a1d` | R110 | different | `6057ae01a3230a08bd6bc39d984771bc016c678321296eab9061ad1e5acc2cd7` |
| <a id="r113"></a>R113 | `token-batch-final-benchmark-specprune_vla-001` | C15 | [specprune_vla](../configs/experiments/openvla_oft/specprune_vla.yaml) | `ff880edb449d3840d2a7420e1363d155ac9dfe95` → `7f851f4dbc57f40299d8f0acc3326b41dac20a1d` | R110 | different | `c98ebecd8039fcd1ffd691677410e4bb02708f0cf5e0473962eab52c80bcbaf2` |
| <a id="r114"></a>R114 | `token-batch-final-benchmark-team_vla-001` | C15 | [team_vla](../configs/experiments/openvla_oft/team_vla.yaml) | `ff880edb449d3840d2a7420e1363d155ac9dfe95` → `7f851f4dbc57f40299d8f0acc3326b41dac20a1d` | R110 | different | `abfd5e1af5ccbbedce776eddc3825ca0bdeec7bbea54338cc4bf1597f36dc7b5` |
| <a id="r115"></a>R115 | `token-batch-final-benchmark-vla_iap-001` | C15 | [vla_iap](../configs/experiments/openvla_oft/vla_iap.yaml) | `ff880edb449d3840d2a7420e1363d155ac9dfe95` → `7f851f4dbc57f40299d8f0acc3326b41dac20a1d` | R110 | different | `293184fa17e39c25574f6d91ec5a7b4118623d07239ca2f1703fe7a43fd1b0df` |
| <a id="r116"></a>R116 | `token-batch-final-benchmark-vla_pruner-001` | C15 | [vla_pruner](../configs/experiments/openvla_oft/vla_pruner.yaml) | `ff880edb449d3840d2a7420e1363d155ac9dfe95` → `7f851f4dbc57f40299d8f0acc3326b41dac20a1d` | R110 | different | `df68768c2b672d1b1bbfefded760cb6f4c42d1d9aa67961fddb5062dce0972ce` |

</details>

## Interpretation boundaries

- DepthCache rows implement the primary-view progressive merging core only;
  carrying-dependent scene reinitialization and anticipatory wrist gating are excluded.
  Measured metric depth is required; RGB-only traces cannot stand in for this input.
- ASR rows implement the deletion instrument with unconditional dense refresh,
  not KV reuse. CogACT uses native cognition attention as its disclosed gate.
  Speedup includes all auxiliary computation, not an assumed actuation overlap.
- FastV uses the released HF last-query variant; DART is token reduction, not
  speculative decoding. SparseVLM is original v1 rank-adaptive pruning/recycling,
  not SparseVLM+ or the released fixed-budget variant.
- SQAP-VLA is deferred pending clarification of contradictory activation
  quantization granularity; it has no runnable proxy and is not an implemented method.
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
