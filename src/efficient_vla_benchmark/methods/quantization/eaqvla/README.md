# EaqVLA — modular weight-GPTQ core

[Paper](https://arxiv.org/abs/2505.21567v1), section 4.3, specifies vision W4,
language W8 and an unquantized projector. Both bindings preserve that independent
module policy. Vision Linear/patch-Conv2d and Llama decoder projections are packed;
embedding, vocabulary readout, projector and native action head remain unchanged.
This transfers the OpenVLA proposal to OFT and CogACT, not their original numbers.

The paper provides no released implementation or reproducible activation quantizer,
GPTQ grouping or calibration recipe. This explicitly scoped implementation uses
weight-only GPTQ, native activations, symmetric per-output ungrouped scales,
full-Hessian global column compensation, block size 128 and 1% dampening.
It must not be described as a full weight-and-activation implementation.
These numerical choices follow the standard GPTQ equations, not a substitute RTN.
The shared solver is independently written against [GPTQ](https://github.com/IST-DASLab/gptq).

Calibration uses the same balanced RLDS training observations, collecting native
inputs independently for each module. Uniform token sampling bounds calibration
memory. Modules are quantized separately, not a jointly tuned end-to-end model.
No gradients, policy updates or evaluation observations are used.
The core mixed precision is the primary preset; any uniform-bit diagnostic must be
labeled as an ablation, not another EaqVLA result.

W4/W8 remain physically packed; tile dequantization feeds floating GEMM/FP32
accumulation. All patch unfolding and packing-runtime overhead is measured.
There is no full reconstructed weight cache and no integer-MMA speed claim.
Fresh native loading is required to undo conversion; artifact coverage, native
biases, geometry, checkpoint and checksums are validated before conversion.
