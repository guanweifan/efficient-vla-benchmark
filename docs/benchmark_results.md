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

Selection digest: `31183e6370350827709d4cf351e35a72730bad6dceb22e5fa857af7afddcc8b8`.

## cogact

| Section | Preset | H | Dev SR (n/200) | Δ pp | ms/control | Speedup | Peak GiB | Final SR (n/2000) | Selection / scope |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Reference | `baseline` | 8 | 91.5% (183) | +0.0 | 23.399 | 1.000× | 14.73 | — | Native H8 reference. Native model. |
| 5.1 | `actuation_slack_refresh_keep25` | 8 | 89.0% (178) | -2.5 | 41.122 | 0.569× | 14.74 | — | 178/200 meets the 3-pp target; keep25 costs less than eligible keep50 in matched compact replay. Deletion core with full auxiliary refresh cost. |
| 5.1 | `adp` | 8 | 91.0% (182) | -0.5 | 21.351 | 1.081× | 14.73 | — | Existing candidate meets Dev loss target. Adaptive pruning. |
| 5.1 | `dart` | 8 | 92.0% (184) | +0.5 | 16.466 | 1.404× | 14.64 | — | Existing candidate meets Dev loss target. Pivot-based token selection. |
| 5.1 | `depth_cache_primary` | 8 | 90.0% (180) | -1.5 | 23.377 | 1.001× | 14.73 | — | Meets Dev target; protected tokens limit observed compression. Primary-view depth merging only; unknown wrist/carrying rules excluded. |
| 5.1 | `divprune` | 8 | 91.0% (182) | -0.5 | 17.320 | 1.335× | 14.63 | — | Existing candidate meets Dev loss target. Diversity-based token selection. |
| 5.1 | `fastv` | 8 | 90.5% (181) | -1.0 | 16.464 | 1.404× | 14.64 | — | Existing candidate meets Dev loss target. Attention-based visual token pruning. |
| 5.1 | `flash_vla_keep224` | 8 | 88.0% (176) | -3.5 | 21.269 | 1.085× | 14.71 | — | 176/200 versus 154/200 at keep192; one episode below Dev target. Active ICS core only; unconfirmed action reuse excluded. |
| 5.1 | `safe_pruner` | 8 | 88.5% (177) | -3.0 | 15.805 | 1.460× | 14.74 | — | Existing candidate meets Dev loss target. Task-aware token pruning. |
| 5.1 | `sparsevlm` | 8 | 92.5% (185) | +1.0 | 18.493 | 1.250× | 14.64 | — | Existing candidate meets Dev loss target. Task-related selection and token recycling. |
| 5.1 | `specprune_vla_static96` | 8 | 91.0% (182) | -0.5 | 20.772 | 1.115× | 14.64 | — | 182/200 versus 174/200 at static128; cap denotes deletions. Static/dynamic pruning with bounded static deletion. |
| 5.2 | `action_cache` | 8 | 91.5% (183) | +0.0 | 22.506 | 1.033× | 14.75 | — | Threshold 0.90 from independent prefill; two native tail steps; frozen bank. DDIM refinement adaptation; 1412 pairs from 77/80 independent successful episodes. |
| 5.2 | `c3ache_conditioned_residual_prefix6` | 8 | 91.5% (183) | +0.0 | 22.971 | 1.012× | 14.73 | — | Six live blocks selected after shorter-prefix residuals caused severe Dev loss. Cross-chunk residual core; two cached steps, refresh interval two; not full Fast-WAM. |
| 5.2 | `dp_cache` | 8 | 90.0% (180) | -1.5 | 21.784 | 1.067× | 14.73 | — | Middle-window output reuse meets the Dev loss target. Native denoiser output reuse only; no VAE/compiler pipeline. |
| 5.2 | `efficient_vla_cache3` | 8 | 88.5% (177) | -3.0 | 21.386 | 1.079× | 14.73 | — | 177/200 after bounded cache-interval tuning. Within-sampling DiT feature cache; native sampler unchanged. |
| 5.2 | `temporal_redundancy_visual_update75` | 8 | 90.0% (180) | -1.5 | 23.954 | 0.970× | 14.85 | — | Update75 selected after update50 scored 168/200; retain measured full-cost overhead. Visual attention core; trained action branch excluded. |
| 5.2 | `vla_cache` | 8 | 89.0% (178) | -2.5 | 19.925 | 1.162× | 14.75 | — | Existing candidate meets Dev loss target. Query-row reuse with retained visual KV. |
| 5.2 | `vla_infoentropy` | 8 | 87.5% (175) | -4.0 | 20.307 | 1.140× | 14.75 | — | 175/200; tested static100 yielded 174/200, so retain original active setting. Entropy-guided query-row reuse; below Dev loss target. |
| 5.3 | `sp_vla` | 8 | 90.5% (181) | -1.0 | 19.096 | 1.208× | 14.73 | — | Existing candidate meets Dev loss target. Pruning plus executed-action scheduling. |
| 6.1 | `awq_fp16_kernel` | 8 | 91.0% (182) | -0.5 | 17.005 | 1.356× | 5.90 | — | Selected FP16-kernel W4A16; BF16 retained as a labeled precision auxiliary. Explicit FP16 cast boundary; training-only calibration. |

## openvla_oft

| Section | Preset | H | Dev SR (n/200) | Δ pp | ms/control | Speedup | Peak GiB | Final SR (n/2000) | Selection / scope |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Reference | `baseline` | 8 | 99.0% (198) | +0.0 | 24.793 | 1.000× | 14.98 | — | Native H8 reference. Native model. |
| 5.1 | `actuation_slack_refresh_deletion` | 8 | 97.5% (195) | -1.5 | 44.000 | 0.563× | 14.69 | — | Retain active configuration; full auxiliary cost is included. Deletion core; not real-robot overlap. |
| 5.1 | `adp` | 8 | 98.5% (197) | -0.5 | 24.037 | 1.022× | 14.68 | — | Existing candidate meets Dev loss target. Adaptive pruning. |
| 5.1 | `dart_layer4` | 8 | 98.5% (197) | -0.5 | 21.511 | 1.141× | 14.63 | — | 197/200 versus 184/200 at the original layer. Pivot-based token selection, later pruning layer. |
| 5.1 | `depth_cache_primary` | 8 | 98.0% (196) | -1.0 | 25.041 | 0.990× | 14.68 | — | Meets Dev target; protected tokens limit observed compression. Primary-view depth merging only; unknown wrist/carrying rules excluded. |
| 5.1 | `divprune` | 8 | 97.0% (194) | -2.0 | 23.605 | 1.039× | 14.62 | — | Existing candidate meets Dev loss target. Diversity-based token selection. |
| 5.1 | `efficient_vla` | 8 | 98.0% (196) | -1.0 | 21.301 | 1.154× | 14.63 | — | Existing candidate meets Dev loss target. Declared OFT token-reduction core. |
| 5.1 | `fastv` | 8 | 99.0% (198) | +0.0 | 20.979 | 1.169× | 14.62 | — | Existing candidate preserves native Dev successes. Attention-based visual token pruning. |
| 5.1 | `flash_vla_keep192` | 8 | 98.0% (196) | -1.0 | 26.203 | 0.937× | 14.66 | — | 196/200 versus 190/200 before retention adjustment. ICS core only; action reuse is unconfirmed and excluded. |
| 5.1 | `safe_pruner` | 8 | 97.5% (195) | -1.5 | 18.227 | 1.347× | 14.68 | — | Existing candidate meets Dev loss target. Task-aware token pruning. |
| 5.1 | `sparsevlm` | 8 | 96.5% (193) | -2.5 | 22.640 | 1.084× | 14.63 | — | Existing candidate meets Dev loss target. Task-related selection and token recycling. |
| 5.1 | `specprune_vla_score_first` | 8 | 96.5% (193) | -2.5 | 19.259 | 1.277× | 14.59 | — | 193/200 versus 158/200 for original timing; intentional ablation. Score-before-first-prune variant, not author-default scheduling. |
| 5.1 | `vla_pruner` | 8 | 97.5% (195) | -1.5 | 20.630 | 1.192× | 14.68 | — | Existing candidate meets Dev loss target. Declared progressive visual pruning core. |
| 5.2 | `temporal_redundancy_visual` | 8 | 96.0% (192) | -3.0 | 25.941 | 0.947× | 15.19 | — | Retain active half-row reuse; report measured full-cost slowdown. Visual attention core; trained action branch excluded. |
| 5.2 | `vla_cache` | 8 | 96.5% (193) | -2.5 | 25.887 | 0.950× | 15.03 | — | Existing candidate meets Dev loss target; retain any measured overhead honestly. Query-row reuse with retained visual KV. |
| 5.2 | `vla_infoentropy` | 8 | 97.5% (195) | -1.5 | 26.595 | 0.925× | 15.03 | — | Existing candidate meets Dev loss target; retain full reuse overhead. Entropy-guided query-row reuse with retained KV. |
| 5.3 | `sp_vla` | 8 | 97.5% (195) | -1.5 | 21.830 | 1.125× | 14.68 | — | Existing candidate meets Dev loss target. Pruning plus executed-action scheduling. |
| 6.1 | `awq_fp16_kernel` | 8 | 97.5% (195) | -1.5 | 21.359 | 1.161× | 6.12 | — | Measured FP16-kernel W4A16 candidate; compare retained BF16 auxiliary. Explicit FP16 cast boundary; training-only calibration. |

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
| cogact / `actuation_slack_refresh_keep25` | `1745acc6a866125188a9997d38d3a193bf3f2238f71a72e8522ed6a2933d2115` | `compact-41-4-cogact-1--actuation_slack_refresh_keep25` | `1dc47be1a1e7d1cd8623a50e329d3a72e5daa8ef` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `ba45d59001f53064891ce78afc87457c6fcfb42d74aae8f49bd7b9ed91127717` |
| cogact / `adp` | `42449ae56acdde4b772b59d1e06aae9fd967c8b2218fc23decbb18768ba3af66` | `compact-43-0-cogact-token-retry--adp` | `e86cbe2edc52d3bc69122a66dc727a8aa1109d93` / `2a49223fe7b558d8ca59b6ceab6557f06fc354b9` | `cb242ba60b5490a89b285e5839708c75c9b176a1bc1c40d22159649e6aa5f744` |
| cogact / `dart` | `703d93d8a87981631d66e7a6466a8793d795eecd9b2805b7525f54bf65efa389` | `compact-43-0-cogact-1--dart` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `4d19aee96c571e6f7d440537615e53cc842fbd6edda5f86777b48bf2e4ccc81b` |
| cogact / `depth_cache_primary` | `8f0ed815f5f86fb2088cb19f63a81b6e9c0637fab2ddf178cd6f1bcf9fa202d6` | `compact-41-4-cogact-1--depth_cache_primary` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `5890e49b51fb07496ee293205d0f3591e58bb0e2dbb6e1560c7f3f2948a2a41b` |
| cogact / `divprune` | `0450741919be84698dcad7bdb2ac98811edad1574c09cabcf4871983a05ff1ae` | `compact-43-0-cogact-1--divprune` | `e86cbe2edc52d3bc69122a66dc727a8aa1109d93` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `c84de5049fcbd29c87c3db70556f4a9e853eaa09581ba0f0e8746c71c36a5bee` |
| cogact / `fastv` | `27489ccc7f64a43601970748ec169d8a89d2c6547eaa078682debf490749bc32` | `compact-43-0-cogact-1--fastv` | `e86cbe2edc52d3bc69122a66dc727a8aa1109d93` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `796648e4a02c030771dee886ef9b8a4d13e098b9cddc20466e725504269077e3` |
| cogact / `flash_vla_keep224` | `6e80dea7bbd83095f95c6bd8dd8a0c5b1584398e9ae1259930ba1c8915e4cdac` | `compact-43-0-cogact-token-retry--flash_vla_keep224` | `1d954ec8f661f0f700810985e2e3d8b8ae5205aa` / `2a49223fe7b558d8ca59b6ceab6557f06fc354b9` | `337b507f4b183dd9fdaff569775a87f2020f365fa1ecfd7d2346697f8616c9a6` |
| cogact / `safe_pruner` | `bdc7820db41f431415a887ed411fead64ed546d22e056708ae5ef56344353f26` | `compact-43-0-cogact-token-retry--safe_pruner` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `2a49223fe7b558d8ca59b6ceab6557f06fc354b9` | `3ba866bc10cd8608a9e9300225f35f6f3a43c3d9c1397d8f0897837bde572a1d` |
| cogact / `sparsevlm` | `19e55b9daed5d94fe5e3c51402cfe296a0a1cb52a519fa5e24cc0025772342e2` | `compact-43-0-cogact-1--sparsevlm` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `7542a802e91644c423d35140814389dcb40ac6784df40029b02c378b0f15b03a` |
| cogact / `specprune_vla_static96` | `f11bd3ab2edf722f6cafdee4b9a5ded3164b79889d30a90afebf71887d62dc96` | `compact-43-2-cogact-inputs-retry--specprune_vla_static96` | `1d954ec8f661f0f700810985e2e3d8b8ae5205aa` / `2a49223fe7b558d8ca59b6ceab6557f06fc354b9` | `7a670001dac8a2193c74cc11ac6a43f0fb9c3474d5b76243d82ac341551611d3` |
| cogact / `action_cache` | `4b3d636daaf8c5a63af7431e8272b01f48762c2bb6a0d52b81a0ffc211f184b4` | `cache-v2-45-6-cogact--action_cache` | `d3c46d34696eb47f753dba3306526bad5ad461c7` / `d3c46d34696eb47f753dba3306526bad5ad461c7` | `6de89ce6f349258258e00f755594484f759dfd558b6dffdd0efa80d5d9b777ca` |
| cogact / `c3ache_conditioned_residual_prefix6` | `c692772b868aa42a4d774070b8028db7bf93ab031712cb7cb5d7b65d9cd03d6e` | `cache-v2-45-6-cogact--c3ache_conditioned_residual_prefix6` | `d3c46d34696eb47f753dba3306526bad5ad461c7` / `d3c46d34696eb47f753dba3306526bad5ad461c7` | `539f67019c803f6333f2f804c123a83bd0f31e373402b7c5fc321eab70d1662a` |
| cogact / `dp_cache` | `6458d39a74fadb3f25082501ae8d88486a7a1520c726bf6c23a4f9e9f328e5fb` | `cache-v1-45-6-cogact--dp_cache` | `92b64fc3044235db11f39fe1e0f9c26df4ac1781` / `92b64fc3044235db11f39fe1e0f9c26df4ac1781` | `9330efb92ea6d7f238e56f783e95bc0a07a19ca3e9568d57175ebf9068a0c2fb` |
| cogact / `efficient_vla_cache3` | `e265084237e3a715d4af23bf9fd4aa704cc121ee7b3e70d5addd79efc2792590` | `compact-42-0-cogact-quant--efficient_vla_cache3` | `1dc47be1a1e7d1cd8623a50e329d3a72e5daa8ef` / `465838d6af15428eace0f168b48392ce6886137b` | `6e8e73244e16c403800290b5f4bd0860e985cb68c5b0f8d4b813dd03df6369d1` |
| cogact / `temporal_redundancy_visual_update75` | `fd8de3e4c0776b169a8a488af5f09a840a22c597869235a171387ed9ab2adca5` | `cache-v3-45-6-cogact--temporal_redundancy_visual_update75` | `d3c46d34696eb47f753dba3306526bad5ad461c7` / `d3c46d34696eb47f753dba3306526bad5ad461c7` | `35d15909a0b0e2be29c00576968e94183efe13ea35c330a6d70b828780b89793` |
| cogact / `vla_cache` | `f9612931a47706abef36d9251e82bcf9c0ceeb249fce9e68412d8214f5a29fcb` | `compact-43-2-cogact-inputs-retry--vla_cache` | `3d6c5ab11dd9880b13536285a7ea478fe6ee7303` / `2a49223fe7b558d8ca59b6ceab6557f06fc354b9` | `f38f970dfc5297f06b904976332fc957ce4fc1fd798761e196d602f54de2ab8b` |
| cogact / `vla_infoentropy` | `3e7f5140c644c221a9a02a45731cf1e949650a3564053c9d662f85e37b11cabb` | `compact-43-2-cogact-inputs-retry--vla_infoentropy` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `2a49223fe7b558d8ca59b6ceab6557f06fc354b9` | `3330159196088131057ec15ea17cdc1667f6f9668632ed2266ff747cae62d186` |
| cogact / `sp_vla` | `27d4793d914e2be7c1eb929c254b8661581eaefb1a68530fc1c0746d0cd764d9` | `compact-43-0-cogact-token-retry--sp_vla` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `2a49223fe7b558d8ca59b6ceab6557f06fc354b9` | `e99743927fcec363c2c3d93fe4529ebf5caed70086bf11c3842164c097267b8c` |
| cogact / `awq_fp16_kernel` | `be03b017c6a40862b3ebc5afa78a0ce2cf31111d49e8763c1fe5d0650391c117` | `compact-42-0-cogact-quant--awq_fp16_kernel` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `465838d6af15428eace0f168b48392ce6886137b` | `cb173d2eb42c17ac8ae3398af5f884071e082d7e3ed7db8ac15db2f40bfa4a61` |
| openvla_oft / `baseline` | `83a2952de24bf599cafcbe504ce1b8e0c5f91824411122370deec9846dbaa71e` | `compact-124-7-openvla_oft-quant--native-pre` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `2a49223fe7b558d8ca59b6ceab6557f06fc354b9` | `77575dfbcb36ed385377a886f994a3d21e6cb2eb1de5d620f896aea2f9f493cd` |
| openvla_oft / `actuation_slack_refresh_deletion` | `2e2b75988b34636d5fa39e24e573aa4a159c179d48954e47203a23897921b39c` | `compact-45-5-openvla_oft-recheck--actuation_slack_refresh_deletion` | `f63c95a6f22a8fd2c1fcdc9fed2441ecb376379d` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `515f46e72adb7e0bd549ce44c0e3716db4d7b3573c3fdf3ad56c826d9df12c2d` |
| openvla_oft / `adp` | `83f9aeebdff8a240e92f929ab699eaabc470de89ecc663c04837580dce3f412a` | `compact-45-6-openvla_oft-2--adp` | `e86cbe2edc52d3bc69122a66dc727a8aa1109d93` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `6f76f82646ba660a22f3e3d9071fd906f2d2dcf3c6e4c9513037391e9fff795b` |
| openvla_oft / `dart_layer4` | `03a1390651f10da98428fda879b3bb34131f032a02463218beb1c31648b493b9` | `compact-45-6-openvla_oft-1--dart_layer4` | `1dc47be1a1e7d1cd8623a50e329d3a72e5daa8ef` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `aa6d0b7a820d6271646a6534dd6119bcf5fa7ee1924b3b299881dcaee26a05e0` |
| openvla_oft / `depth_cache_primary` | `23ee03e39b8e019f7c9674baabb00b045f8f303a25ea442dd81e7a27516cb331` | `compact-45-5-openvla_oft-recheck--depth_cache_primary` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `498b1401c4d447a11878f85934fb8fac87172fb74647f47716cf0afae77c7fc8` |
| openvla_oft / `divprune` | `4482ae63177ce1099a9ba5360b83ab01317cb8d5519527dbd8aecd75cb083aea` | `compact-45-6-openvla_oft-1--divprune` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `2985f18b349a71a33698d80f742a62ab0fe8ff8250dc1e1ae0b1008dae27f93c` |
| openvla_oft / `efficient_vla` | `fa758c551bfb39b4f8a0a122e5dd9b278d42ceb1728352c67697012322438fb3` | `compact-45-7-openvla_oft-2--efficient_vla` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `fcde2712458d3ac1f53224c59b56e040a4f7ad80f529ea1f01b3972e11eea793` |
| openvla_oft / `fastv` | `66828e8ed4865b10d7301a447b0fa90beb10e83827a02f0aac1c8551701e69d1` | `compact-45-6-openvla_oft-1--fastv` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `020dbfb6c0f14a90fe92417b4181f37248ac58e84ec37148d6147dd3a61c646a` |
| openvla_oft / `flash_vla_keep192` | `143755ed1aee192a41d545ed818b591dcfb00fc7d0f0c93802b56b060fb7f84e` | `compact-45-6-openvla_oft-2--flash_vla_keep192` | `1d954ec8f661f0f700810985e2e3d8b8ae5205aa` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `50392782613813f5ba30917361231127bbaf4cda0de1347d6322856fbd2853a6` |
| openvla_oft / `safe_pruner` | `21e11d22bc36089d4007bafe5b33d9b790947dbef1e904ab92e73336bf26ffec` | `compact-45-6-openvla_oft-2--safe_pruner` | `e86cbe2edc52d3bc69122a66dc727a8aa1109d93` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `ebcb296bd120d7f95d1fed85c36145056d78af44f8c542401a546628abface79` |
| openvla_oft / `sparsevlm` | `071e8857510d6e9ed29997e514942f2950f8af42284c5bba9ff50ba0b2c7f6ef` | `compact-45-6-openvla_oft-1--sparsevlm` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `a3b59ca6a58f2f2a7b4d348e25b1940cf367ccc10be67a83fe0c648e751ae3d9` |
| openvla_oft / `specprune_vla_score_first` | `4ad4e29d8c6340d9596de67535f4e4319935c722593a0e5be25a3e5974120685` | `compact-45-7-openvla_oft-1--specprune_vla_score_first` | `1dc47be1a1e7d1cd8623a50e329d3a72e5daa8ef` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `51144f8d8c2003e79eed54676630b81edfda7f349412b9acd5e7885ba6ec8cc1` |
| openvla_oft / `vla_pruner` | `bdc953860094a1e5dc0a255f8c0bc53142570fbd610242920303d1b1a1216310` | `compact-45-7-openvla_oft-1--vla_pruner` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `813e27c492f55ed87778cce03ec5406d3714d604863031d773944feecd385501` |
| openvla_oft / `temporal_redundancy_visual` | `a0480729b2ee658d04b2ce8dc01bd888cadca71e070a130a2cb0143cf2efd768` | `cache-v1-45-7-oft-recheck--temporal_redundancy_visual` | `92b64fc3044235db11f39fe1e0f9c26df4ac1781` / `92b64fc3044235db11f39fe1e0f9c26df4ac1781` | `2ccdeeaaf24de6c4d950289190a72706b8eadb5bdeaea93e0df62f9ec9c71e15` |
| openvla_oft / `vla_cache` | `05bde38fdbf83dfb4f0928768c7d460b51ef106a49e29eeb6aaf7783da1da17d` | `compact-45-7-openvla_oft-1--vla_cache` | `e86cbe2edc52d3bc69122a66dc727a8aa1109d93` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `34fe4a649f26a7631e6e0622091082808d643c0555b03cc201844e3fe12bf750` |
| openvla_oft / `vla_infoentropy` | `fc132144e01cbf7c6a681a8b9f59e0010c7cb55a4742809365070e9632b55733` | `compact-45-7-openvla_oft-1--vla_infoentropy` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `4a6499beeec7a1677b998c573e499ae9891d141426efad983fbf81feeafa1bb1` |
| openvla_oft / `sp_vla` | `c321f89069f8c73c936a1388ee57c00bab318a500a5f81d0ec5b9074a94a23a8` | `compact-45-6-openvla_oft-2--sp_vla` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `70ef6b76fdff34a2ca60ad5c3029782aecf46b4d4cc1b7aaf7ec3ecebc60756e` |
| openvla_oft / `awq_fp16_kernel` | `566ed69c9ce2db872ba40344f9123dac0ba1db2f05ef3c240c7f156406fe6d66` | `compact-124-7-openvla_oft-quant--awq_fp16_kernel` | `e86cbe2edc52d3bc69122a66dc727a8aa1109d93` / `2a49223fe7b558d8ca59b6ceab6557f06fc354b9` | `b1a7536157e29af90a22cc43bb47a37850b6d3ea9e3d8b499df6089f0311cab1` |
| openvla_oft / `awq` | `5d64703c2acfcfe4a33ed22f76215b16457c1d61cc0be948e134f97124f509cf` | `compact-124-7-openvla_oft-quant--awq` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `2a49223fe7b558d8ca59b6ceab6557f06fc354b9` | `8720ab41aa47c710bc8c10a98479089045df80598caa44b3cf79b9645c6b8a8d` |
| cogact / `awq` | `a223cf526c859a8a6071ba21da16d2be890c981c5048f52ab863fda22eace8dc` | `compact-42-0-cogact-quant--awq` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `465838d6af15428eace0f168b48392ce6886137b` | `2b53ecaeb6d71307c5630e29c56920594ad7c3f6663d15eb0660d780aebc8f2f` |
| cogact / `actuation_slack_refresh_keep50` | `3e9ab07826e30785e609207a39c4c964aadc37185ff424e01b3643cfb527bd3c` | `compact-41-4-cogact-1--actuation_slack_refresh_keep50` | `1dc47be1a1e7d1cd8623a50e329d3a72e5daa8ef` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `c688748fc315d380b58c73462d81222c806f6ed88f9cf027c16ad63c3ca73762` |
| cogact / `st_merge_keep192` | `2715131409b114d95dc8dc1349d3ea4da8b4baa5ab6f0e78127aff0d3dc431eb` | `compact-43-2-cogact-3--st_merge_keep192` | `1dc47be1a1e7d1cd8623a50e329d3a72e5daa8ef` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `c67baa5ef382b3aa4fbdce5a312500f136981ba4179eb5163cb4f29398bb594f` |
| openvla_oft / `st_merge_keep192` | `d6e8965949d65ac92a7079a13bbe478bc63fd38ef9d9657f513d8051b75f9a46` | `compact-45-5-openvla_oft-recheck--st_merge_keep192` | `1dc47be1a1e7d1cd8623a50e329d3a72e5daa8ef` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `945d70c1affb3d3c243e7eed3a688112f005fbf591f8e87be257572bd16dc62a` |
| openvla_oft / `team_vla_merge160` | `02fc7c9f038ee76b07c85196460993479de227eba74c4343cca09310552f327c` | `compact-45-7-openvla_oft-3--team_vla_merge160` | `1dc47be1a1e7d1cd8623a50e329d3a72e5daa8ef` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `5f45fec789ab8a8b09723be9fa5fcb66ebf42d78ee443fa28c56b7be88d7a7bd` |
| openvla_oft / `vla_iap` | `0a6a2c9c71aefa83b1f9ce71ac8d7813d3bb4ed9c4494769d977fdf123756ae0` | `compact-45-7-openvla_oft-3--vla_iap` | `bf60e9ff70b38b989df57982bf1f92ea51fd08c9` / `e6a91178f1a1f6f744878977c34856770ee9f1d9` | `c51e783e5187bfc02f9c44dd8c91c8309d4a26383cb18a190dfa3dd0378a862b` |
