# SAFE-Pruner on OpenVLA-OFT

[Paper](https://arxiv.org/abs/2605.29662v1),
[official source](https://github.com/msssl/SAFE-Pruner/tree/0094232dfdb332a087b246d56316a7c6c16c8598).
Reference submodule: `third_party/methods/safe_pruner` (MIT).

## Core mechanism

Match current projected visual features to the latest complete keyframe by cosine
nearest neighbors; transfer its layer-wise saliency to current patch IDs. At layer
zero, compare forecast and current saliency to detect a reference refresh. A refresh
query runs full-token inference, storing new features and saliency; otherwise fuse
current shallow saliency with forecast saliency and prune before layer 3. A keyframe
is an observed past frame, not access to future observations or an extra model pass.
No policy training, predictor training or KV-cache acceleration is involved.

## Released preset and adaptations

- Follow `run_eval.sh`: reference layers 0/31, forecast weight 1, gate cosine
  threshold 0.92, action queries for deep/shallow pruning scores. Layer-zero
  reference and gate explicitly use the final token because zero action-slot
  embeddings give uniform first-layer action attention. This source detail is
  not silently replaced with a different query slice.
- Fusion is `current + weight * mean(forecast)`, as released. Matching is global
  across both cameras, also as released. Normalize in native dtype before FP32
  correspondence matmul. Preserve native positions, bilateral mask and readout.
- Store exactly 512 VISUAL reference features. The source updates its feature
  cache after appending proprio, potentially giving 513 features for 512 saliency
  entries; the port prevents that feature/score indexing mismatch.
- Use one 25% retention preset across all four suites, rather than the author's
  task-specific 10–30% settings and separate weights. Actual full-refresh frequency
  and retained-token counts are recorded; nominal sparse ratio is not average cost.
- Only required attention rows/layers are probed, with FP32 head reduction.
  No replacement Transformers installation, full-layer attention retention, hidden
  CUDA synchronization or plotting. Necessary gate synchronization and ALL feature
  matching, probing, refresh and selection costs remain in policy timing.
- Cache state belongs to this instance, clears on reset, and is never shared with
  another method. Disabled/100%-retention configurations use native model execution.

The independent scorer tests compare feature correspondences with pinned code and
check fusion, shape guards and refresh/reset behavior. Real-model tests cover native
cold-frame parity, sparse subsequent frames and disabled parity. Use
`configs/experiments/openvla_oft/safe_pruner.yaml`; outcomes belong in the central
measurement ledger and do not constitute exact paper-number reproduction.
