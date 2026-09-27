# V-AEFusion

[Paper, section 5.3.1](https://arxiv.org/html/2604.24447v1).
Independent implementation of the VLM/action-expert pipeline contribution, not
the same paper's separately registered DP-Cache or compilation optimizations.

After one native warm query, a private worker and CUDA stream perform the first
`stale_steps` denoising iterations using the previous query's cognition. The native
VLM concurrently encodes the current observation. The remaining iterations use
its fresh cognition. At least one fresh step is mandatory; no denoising step,
native CFG branch or H8 control is omitted. First query/reset are entirely fresh.
CogACT uses a cognition vector instead of pi0's conditioning KV tensors; this is
an explicit conditioning-boundary adaptation, not a claim of pi0 code parity.

The native prediction/preprocessing/action conversion path stays intact. Only
instance-local sampler and VLM inspection wrappers are installed. A private RNG
clones the next native action draw without racing/restoring the global RNG; the
main thread retains identical eta-zero DDIM random-number advancement. Every
active query verifies the initial native and private noises match. Worker errors
propagate, and reset/close join pending work before removing hooks and state.

CUDA event intervals verify that both streams are dispatched during a query.
Their interval overlap is not a claim that individual kernels execute
simultaneously; resource contention may erase any end-to-end benefit. The common
full-cost timing includes thread dispatch, event recording, RNG verification,
joins and all stale/fresh inference. No background cost is discarded. A fixed
stale budget follows the paper's ablation; this port does not dynamically change
the budget based on host scheduling or the eventual task outcome.

Initial candidates 1/3/5/7/9 are not final report selections. Zero is a native
parity diagnostic, never an active primary result. OFT's deterministic L1 action
head has no iterative refinement and is not registered.

```bash
evb evaluate --config configs/experiments/cogact/v_aefusion.yaml \
  --model-path "$MODEL_DIR" --libero-root "$LIBERO_ROOT" --output-dir "$RUN_DIR"
```
