# DVAC

[Denoising Tells When to Replan](https://arxiv.org/html/2606.03847v1),
equations 4 and 6--9 and Algorithm 1. The core measures variance across the final
clean-action estimates, chooses the prefix before the first high-variance action,
then updates a causal rolling threshold buffer.

This independent CogACT adaptation reads native DDIM `pred_xstart`, not the noisy
latent and not a flow velocity. It keeps the original denoising steps, CFG,
precision, full sixteen-action prediction, native normalization and environment
conversion. Only the executed prefix changes. Conditional row zero contains
guided predictions; do not average conditional and unconditional CFG rows.
There is one VLM encoding and one native denoising chain, with no extra sampling.

The paper gives L=5, m=5, alpha=2 and minimum execution one. These defaults are
retained; maximum execution is bounded by the real CogACT prediction length.
Population variance is used in equation 4. The rolling standard deviation also
uses population normalization, an explicit convention where no official code
was available. Decide using prior queries, then append current variance, as
Algorithm 1 specifies. The paper does not define empty-buffer initialization:
this port executes the native reference prefix on the first query only. It does
not calibrate with evaluation outcomes or substitute a trained uncertainty model.

Reset/instruction changes clear the rolling buffer. Close restores the sampler
instance without modifying its class. Disabled mode has no observation wrapper.
Core tests check causal ordering, exact first-crossing semantics, scale behavior,
native action/RNG parity and detach/reset. Real-model tests precede smoke and Dev.

H8 remains the native comparator, not a claim of fixed method cadence. Record
selected horizons, actual query counts and complete policy/control cost. Tail
collection, variance computation and host synchronization are included in timing.
Performance on this diffusion checkpoint does not reproduce the paper's flow-model
numbers; low success or overhead is reported honestly after fidelity checks.

```bash
python -m pytest src/efficient_vla_benchmark/methods/action_chunking/dvac/tests -m 'not gpu'
python -m pytest src/efficient_vla_benchmark/methods/action_chunking/dvac/tests -m gpu \
  --model-path "$MODEL_DIR" --trace-path "$TRACE"
```
