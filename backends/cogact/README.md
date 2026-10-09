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

### Fixed native controls

`native_h16` executes all sixteen already-predicted actions, retaining DDIM10.
`native_ddim4` and `native_ddim5` retain H8 and use upstream `ddim4`/`ddim5`
respacing without solver replacement. With the native 100-step training schedule,
their descending prediction grids are [75, 50, 25, 0] and [80, 60, 40, 20, 0],
versus [90, 80, ..., 0] for DDIM10. Thus these are native-budget controls, not
endpoint-matched solver ablations. CFG remains 1.5; one paired CFG batch counts
as one denoiser evaluation. No methods, training or cached history are added.

These three owner-specified supplementary controls are not new method scopes
or candidates for primary selection. Dev-200 validates each fixed configuration;
it does not choose among them or enforce the 3 pp selection target. All three
receive one Final-2000 run regardless of outcome after correctness checks.
Speed references stay explicitly native DDIM10/H8. Report actual execution
horizons, denoiser grids, query counts and full-control costs separately.

## Integration boundary

The plugin owns model assembly, preprocessing and action conversion. The common
LIBERO environment loop, success checks and sequential timing are unchanged.
Existing OFT methods are not automatically compatible: each requires an explicit
CogACT binding and its own tests before registration.

The loader replaces only a scoped vision-constructor factory to suppress downloads
of redundant pretrained weights, restores it before inference, loads every component
strictly, and resolves the tokenizer from the caller-selected export. It does not
modify upstream files, global model classes, or the supplied export.

See [INSTALL.md](INSTALL.md) for setup and [results](../../docs/benchmark_results.md)
for measurements and limits. Native output/reset parity passed exactly on the tested
export; four-episode smoke is not a formal success rate. The export names training
revision `01869b7348b41cb315049f32abc7221fc01897db`, but its actual training-source
location remains unverified. Native parity does not validate the training fork's
evaluation conventions; a failed guessed URL is not proof that its source is private.

## Causal visual-token selection

`model_access.attach_token_strategy()` exposes before/after-layer selection hooks
to the selected method. A strategy returns a boolean mask of current rows; only
visual rows can be removed. A shared `TokenReduction` result may also replace
selected visual features; merge anchors must preserve causal information order.
Original patch/RoPE positions and all nonvisual tokens,
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

Bindings include ADP (including observed-EEF gating), FlashVLA's ICS core and
SAFE-Pruner's forecast/refresh mechanism with explicit native-cognition queries.
Their tests check original positions, protected cognition, causal/padding masks,
full-retention/reset/detach parity and unchanged diffusion RNG consumption. Paired
measurements and smoke are in [results](../../docs/benchmark_results.md). VLA-IAP is
deferred until its defining text/vision semantic-feature construction is verifiable;
matching dimensions or implementing only edges is insufficient. Additional bindings
must preserve their own core mechanisms and pass native-boundary checks.

## Additional model-access capabilities

`model_access.policy_image(image)` reproduces the native spatial transform in RGB
for patch inspection (including 256px simulator images resized to the 224px grid).
`attach_reuse_strategy()` is a separate causal indexed-KV adapter; it retains full
key context while skipping only selected visual query/MLP rows. Causal masks use
original query positions, not the last mask row or a triangle over sparse indices.
Native cognition EOS is always computed, and the DiT sampler is unchanged.
Attention inspection has an explicit eager/SDPA mode reported in runtime metadata.
Fresh-prefill visual selection alone does not imply this caching capability.

`attach_vision_strategy()` provides one-view paired DINO/SigLIP block transforms.
It shares group/mass bookkeeping with the OFT adapter but consumes native CogACT
image dictionaries. SigLIP supplies a method's grouping before DINO runs; both
branches must retain identical original patch memberships and protected prefixes.
Sparse output is restored to native patch slots for the unmodified projector;
the selected method then removes empty slots before LLM layer zero. The dense
projector and bridge overhead remain part of timing.

`model_access.action_denoiser` exposes the native DiT instance and its blocks.
Its attention keys are one cognition condition plus sixteen action positions,
not 256 image patches. CFG calls `forward_with_cfg()`, which directly invokes
the instance's `forward()`; method wrappers must cover that path and restore
instance attributes on close. Sampler/CFG/noise ownership remains native.

## Audited ports and unresolved scopes

The 2026-09-22 batch adds SAFE-Pruner, VLA-InfoEntropy, ST-Merge, SpecPrune-VLA
and the explicitly separate EfficientVLA DiT cache core. SAFE/SpecPrune disclose
native-cognition query adaptations; neither claims measured action-to-patch
attention inside DiT. Read method READMEs for reference choices and tested scope.

| Scope not registered on CogACT | Missing core condition |
|---|---|
| VLA-Pruner | Independent action-decoder-to-visual-patch attention for its dual scores |
| TEAM-VLA | Verifiable joint text/action-token guidance for its mid-layer merging |
| EfficientVLA token core | Eq. 2 visual-to-following-text attention conflicts with native causality |
| VLA-IAP | Specified aligned semantic visual/text feature construction |

These are scoped fidelity deferrals, not blanket claims of impossibility. Kernel,
directory or hook work alone is not a reason to defer; inventing a different core
score is. The existing OFT bindings and historical measurements remain unchanged.

FastV, DivPrune, DART, SparseVLM+ v2, DepthCache primary and ASR deletion have
separate native bindings. FastV uses final cognition attention; DART reads native
post-RoPE keys; SparseVLM+ v2 preserves positions, while DepthCache anchors merges.
DepthCache consumes measured primary-camera depth without adding a model input.
ASR uses a disclosed cognition-attention gate, not nonexistent DiT patch attention;
its discarded dense pass retains the native sampler and restores RNG afterwards.
The common runtime joins auxiliary work for full-cost timing. None of these ports
changes DDIM steps, CFG, normalization, native action head or executed horizon.

## Optional sampler replacement

`sampling.EpsilonSamplerContract` exposes the original epsilon-prediction noise
schedule, native DDIM start and discarded-draw count to a selected method.
The original CFG model callback and action conversion remain native. Install
`backends/cogact[samplers]` for the pinned optional scheduler dependency; baseline
loading does not import it. Solver equations and budget decisions belong to the
selected method, not this backend. Model weights remain read-only.
