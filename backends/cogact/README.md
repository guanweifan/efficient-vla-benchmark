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
