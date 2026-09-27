# FAST+

Use the released [universal processor](https://huggingface.co/physical-intelligence/fast)
at `ec4d7aa71691cac0b8bed6942be45684db2110f4`: orthonormal DCT, scale 10,
minimum symbol -354 and the pretrained 2048-entry BPE. Never fit a replacement BPE.
Input is normalized with calibration-only per-suite action quantiles and signed
open-gripper. Inverse normalization remains inside reconstruction timing/error.

The native processor catches malformed decode sequences and emits zeros. This
binding verifies the recovered coefficient count first and records malformed
payloads as errors instead. Horizon is explicit, never inherited from the previous
sample. The released DCT/BPE implementation is CPU-based even on a GPU test host.
No policy-head replacement or policy-success claim is made.
