# VQ-VLA tokenizer

[Model](https://huggingface.co/VQ-VLA/vq-vla-weight) revision
`cbf998d66004f32010e9d070539d8f6e7eed510f`; [source](https://github.com/xiaoxiao0406/VQ-VLA)
revision `7fb78e63f5c0f7baf6d340166ffea527c6514610`.
Load only the released all-data VQ-VAE, not its policy. All tensor keys must match.
The checkpoint's training flags enable both action-type and temporal positional
encoding; preserve both and verify the flags rather than dropping unmatched keys.
The tokenizer source subset retains original MIT/Apache notices; the only source
adaptation changes absolute `prismatic.action_vqvae` imports into relative imports,
so it cannot import or collide with a policy's Prismatic package.

The native decoder produces five actions. A forty-frame sample uses eight successive
native blocks, each with four residual-quantizer codes from 256-entry codebooks.
Keep stage-local IDs with known stage order; LLM-vocabulary offsets are a separate
policy serialization and are not extra codebook bits here. No encoder latents reach
the decoder. FP32 is explicit for both encoder and decoder.

Preserve the native BOUNDS_Q99 formula for the six motion dimensions and the native
0/1 open-gripper convention. Quantile bounds are fitted on the fixed calibration
split, not reconstruction samples. This is an adaptation of the normalization
recipe, not a claim to recover unpublished original training-set statistics.
Clipping and continuous gripper reconstruction remain in raw-command error.
Released all-data pretraining may include LIBERO; no unseen-data/policy-success claim.
