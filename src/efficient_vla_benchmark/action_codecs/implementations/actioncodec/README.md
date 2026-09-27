# ActionCodec-Base

[Released model and runtime](https://huggingface.co/ZibinDong/ActionCodec-Base),
revision `65b162110b48303fd409fa08ddb088648993b702`. Strictly load the Base VQ
checkpoint, sixteen 2048-vocabulary tokens per native twenty-frame LIBERO block.
The forty-frame corpus is encoded as two successive native blocks, without padding,
resampling, replacing the codebook or using an RVQ-finetuned checkpoint.

Use the released `a_franka_libero_20hz` embodiment and its model-card normalizer:
divide each motion coordinate by `max(abs(q01), abs(q99))`, convert open-gripper
to signed values, then clip. Reverse conversion is included in timing and error.
Do not substitute affine per-suite normalization. Validate every decode padding mask.

The embodiment's 20 Hz label is preserved; no physical control-rate equivalence is
claimed for the demonstration corpus. Report frames/tokens, not seconds of motion.
These released weights explicitly include LIBERO, so reconstruction is in-domain,
not unseen-data generalization and not the success rate of a new VLA policy.
