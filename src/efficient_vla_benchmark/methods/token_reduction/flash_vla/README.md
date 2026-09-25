# FlashVLA (2025): ICS token-pruning core

This is **Think Twice, Act Once**, [arXiv:2505.21200v1](https://arxiv.org/html/2505.21200v1),
not the later streaming-action-decoding work with the same name.
Independent paper-based section 3.2 implementation; NOT full FlashVLA reproduction.

Read the actual attention output after its output projection, BEFORE the residual
and MLP. Per camera, perform FP32 economy SVD and score each visual token by
sum(abs(U * singular_values)), Eq. 2. Retain the top configured count in original
order. No Gram approximation, low-rank truncation or squared-energy replacement.
Full-retention/disabled mode is native and installs no hooks. SVD runs ONLINE and
its complete cost is included in timing, even if the method is slower.

`svd_driver: auto` uses PyTorch's complete Jacobi SVD with `gesvd` fallback on
nonconvergence. The earlier implementation forced the slower QR-based `gesvd`
driver; `flash_vla_gesvd.yaml` retains it as a numerical/performance reference.
Both keep every singular direction and use the same FP32 L1 scoring equation.
`gesvda`, randomized SVD, rank truncation and Gram-matrix approximations are not
supported. Floating-point scores need not be bitwise equal across solvers;
nearly repeated singular values can rotate singular vectors and affect L1 ranks.
Driver choice and any numerical discrepancies must remain visible in evidence.
An explicit QR pre-reduction was investigated but not adopted: it added complexity
without a meaningful improvement over direct `gesvd` on the measured shapes.

OFT conventions: layer 1 (zero-based), 160/256 tokens per camera, all singular
directions, native original RoPE/action positions. Per-camera scoring extends the
paper's single-camera setup. The theoretical L1-score argument does not establish
the claimed L2-energy dominance; we reproduce the stated score, not that proof.

FlashTrigger action reuse is deliberately unresolved: Eq. 7 defines an angle,
Eq. 10 triggers when the angle is GREATER than epsilon, while the narrative argues
for reuse of stable actions. The default epsilon=2 does not disambiguate units.
No verified author implementation was found. We do not silently reverse the
inequality or use a guessed chunk-reuse policy. Reuse remains pending an explicit
interpretation decision; runtime metadata and tables label this as ICS core only.

Use `configs/experiments/openvla_oft/flash_vla.yaml` for the same native H=8 replay
and four-episode smoke. Published full-FlashVLA speedups include action reuse and
cannot be compared as if that component were implemented here.

## CogACT binding

`implementations/cogact/` uses the same complete-SVD ICS equation on the native
Llama attention `o_proj` output. One 256-patch view retains 160 patches after
zero-based layer 1, with unchanged original positions, causal visibility and final
cognition readout. Native SDPA modules are not replaced. All online SVD, ranking
and selection overhead remains timed; the diffusion head, DDIM steps, CFG, noise
generation and action prefix stay at the native baseline settings.

Unlike OFT's bidirectional execution, the CogACT visual prefix cannot attend to
later language tokens. This is the native model context, not a defect to fix by
unmasking text or substituting cognition-token attention. ICS remains a visual
information score; do not relabel it task-attention scoring. The paper's original
OpenVLA path is also causal. We preserve Eq. 2, not its disputed L2-energy proof.

Use `configs/experiments/cogact/flash_vla.yaml`. No action reuse, low-rank
approximation or alternate scoring equation is introduced. These results test
the independent ICS contribution on this checkpoint, not full FlashVLA scores.

## Bounded development diagnostics

CogACT Dev-200 presets `flash_vla_keep192` and `flash_vla_layer4` vary only
retention (160 to 192) or pruning layer (1 to 4). Complete FP32 SVD and its L1
score remain unchanged. Action reuse remains excluded, including in all tables.

Fixed CogACT Dev-200 scored 126/200 by default, 154/200 with keep192 and
113/200 with layer4, versus native 183/200. Budget relief helped, whereas this
later pruning layer did not. The residual 14.5-point gap at keep192 is retained;
complete SVD checks do not establish that all lost accuracy is intrinsic to ICS.
