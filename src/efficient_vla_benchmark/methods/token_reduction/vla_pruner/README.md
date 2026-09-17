# VLA-Pruner on OpenVLA-OFT

[Paper](https://arxiv.org/abs/2511.16449v3),
[official OFT source](https://github.com/MINT-SJTU/VLA-Pruner/tree/84d4b7192c77abf1585610e2f12393319b7ebff9).
Reference submodule: `third_party/methods/vla_pruner` (MIT).

The port implements the full dual-score token mechanism: combine per-camera
semantic and action top-k sets, then filter candidate redundancy by greedy max-min
cosine distance. Semantic importance averages all pre-action query rows, matching
the released OFT code, not only language queries. Action guidance comes from the
three preceding query frames, weighted newest-first by 0.8 decay. Their scores are
sampled at layer 15, scattered to original patch IDs, and missing positions are
zero-filled. The first three queries are full-token warm start. Pruning follows
layer 3 (zero-based), retaining 25% per camera in the supplied preset.

## Integration and explicit differences

- Native joint-suite checkpoint, original positions, bilateral mask and native
  one-token-offset readout are unchanged. No training or additional model is used.
- Collect only necessary query rows at the pruning/history layers. The original
  fork requests all attention tensors at all layers; our native-SDPA probes avoid
  retaining those unrelated tensors. Head statistics use FP32 accumulation.
- Diversity uses original projected visual embeddings, not intermediate hidden
  states. Incremental nearest-distance updates compute the same max-min rule as
  repeatedly reducing all chosen rows. Explicitly exclude already selected IDs
  to avoid duplicate selection for tied/degenerate features in the reference.
- History is bounded and GPU-resident. A query snapshots only PREVIOUS history;
  changing history/pruning layer order cannot leak current scores into that prior.
- Original patch IDs—not compacted indices—link history across frames. Episode or
  instruction reset clears all history. No independent method is composed here.
- `enabled: false` and `keep_ratio: 1` use the untouched native execution path.
  `temporal: false` is the documented current-action ablation. All budgets and
  layer/history settings are configurable and validated before model loading.

Tests check pinned-source selection parity, decay order, degenerate-feature uniqueness,
invalid configs and real OFT warm-start/disabled parity, active pruning and reset.
Use `configs/experiments/openvla_oft/vla_pruner.yaml`; scoring/filtering/history
overhead stays inside common timing. The central measurement ledger records the
adaptation's outcomes, not the paper's exact success rate or published speedup.
