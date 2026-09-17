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
