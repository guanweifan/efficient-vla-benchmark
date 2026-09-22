# CogACT backend

Optional distribution: `efficient-vla-backend-cogact`.
Backend ID: `cogact`; model variant: `libero_joint`.

This adapter assembles a complete joint-four-suite LIBERO export using pinned native
CogACT and Prismatic classes. Model loading is strict and offline; checkpoints are
read-only. The upstream `predict_action` implementation is retained unchanged.

## Native baseline contract

- One agentview camera; no wrist or proprioception input to this trained architecture.
- DINOv2/SigLIP 224px features, Llama-2 causal SDPA, DiT-B, sixteen predicted 7D actions.
- BF16 VLM and native FP32 diffusion head. DDIM 10 steps, CFG 1.5 by default.
- Shared controller executes eight actions before replanning; no adaptive ensemble.
- Four suite-specific normalization entries from one checkpoint, not four models.
- Native image resize/normalization without an additional OFT center crop.
- Convert binary open-gripper output once to LIBERO's signed gripper convention;
  Cartesian/axis-angle deltas are not transformed using SIMPLER-specific Euler logic.

Training diffusion repeats are not inference denoising steps. Each is recorded
separately; do not infer the sampler settings from the training repeat count.

## Integration boundary

The plugin owns model assembly, preprocessing and action conversion. The common
LIBERO environment loop, success checks and sequential timing are unchanged.
Existing OFT methods are not automatically compatible: each requires an explicit
CogACT binding and its own tests before registration.

The loader replaces only a scoped vision-constructor factory to suppress downloads
of redundant pretrained weights, restores it before inference, loads every component
strictly, and resolves the tokenizer from the caller-selected export. It does not
modify upstream files, global model classes, or the supplied export.

See [INSTALL.md](INSTALL.md) for setup and [validation](../../docs/cogact_validation.md)
for the executed checks, results and limitations. Smoke outcomes must not be reported
as formal success rates.

## Causal visual-token selection

`model_access.attach_token_strategy()` exposes before/after-layer selection hooks
to the selected method. A strategy returns a boolean mask of current rows; only
visual rows can be removed. Original patch/RoPE positions and all nonvisual tokens,
including the final cognition EOS, remain protected. One camera has 256 fused
DINO/SigLIP patch tokens, not two camera streams.

This adapter keeps each native LlamaSdpaAttention module. Explicit masks retain
each query's native causal and padding constraints; an absent mask uses the native
causal SDPA path. Native generation, fresh-prefill KV construction, final
normalization and diffusion conditioning are retained. It does not reconstruct
OFT action slots, broadcast the final mask row or change the DDIM sampler.

Only fresh batch-one native prefill is supported. External past KV, nested
forwards and full attention-probability output requests fail explicitly.
Cross-frame KV caching and arbitrary autoregressive cached decoding are not
advertised by this capability. Selection/probing policies belong to method
bindings, not this backend. Closing hooks restores the original instance.

Validated bindings are ADP (including observed-EEF gating) and FlashVLA's ICS core.
See [token-port validation](../../docs/cogact_token_batch1_validation.md) for core
mechanism checks, native-parity tests, paired timing and closed-loop smoke evidence.
Other methods require their own CogACT bindings and fidelity checks.
