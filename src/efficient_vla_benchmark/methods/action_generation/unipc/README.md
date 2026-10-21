# UniPC: native CogACT port

[Paper](https://arxiv.org/abs/2302.04867). This training-free port retains the paper's numerical
integration mechanism through the pinned `diffusers==0.30.3` scheduler.
[Runtime source](https://github.com/huggingface/diffusers/tree/v0.30.3/src/diffusers/schedulers).
It is not a VLA-specific method or a reproduction of the paper's image metrics.

The corrector remains enabled at every eligible step and reuses existing evaluations. No external predictor or corrector-disable heuristic is substituted. Both predictor and corrector work are included in timing.

Only the action sampler changes: VLM conditioning, FP32 DiT, CFG 1.5, all sixteen
predicted actions, native clipping/unnormalization and H8 execution remain unchanged.
The finite-noise start stays at the native DDIM10 training index 90; changing the
budget never moves it to index 99 or assumes a zero-SNR start. The selected integer
grid ends with an evaluation at index zero and then the analytical clean endpoint.
The final preset uses a rounded log-SNR grid; repeated indices fail.
The library's generic initial-noise scaling is NOT applied to native CogACT noise.

The shared sampler executor only owns geometry, lifecycle and accounting. This
directory owns the scheduler choice and its method-specific equations. No method
imports another method, and the native backend does not select a paper algorithm.
History resets for every query. Detach restores the exact sampler instance.
Preserving the native sampler's discarded eta-zero draws keeps subsequent initial
action noises aligned across budgets. All overhead remains in full-cost timing.

Order two requires at least three evaluations; order three requires five.
Smaller budgets would disable the defining high-order update under endpoint
safeguards and are not primary method configurations. The final preset uses 5 evaluations. Its Dev-selected
workpoint and measured outcomes are in the central ledger; this is not a claim
of global optimality or a guarantee of acceleration.

OFT's deterministic L1 head has no diffusion ODE and is unsupported.
No training, model replacement, action-cache combination or horizon modification.

```bash
python -m pytest src/efficient_vla_benchmark/methods/action_generation/unipc/tests -m 'not gpu'
python -m pytest src/efficient_vla_benchmark/methods/action_generation/unipc/tests -m gpu \
  --model-path "$MODEL_DIR" --trace-path "$TRACE"
evb evaluate --config configs/experiments/cogact/unipc.yaml \
  --model-path "$MODEL_DIR" --libero-root "$LIBERO_ROOT" --output-dir "$RUN_DIR"
```

Measurement status and selected configurations belong to the canonical result
ledger; registration alone is not evidence of tested performance.
