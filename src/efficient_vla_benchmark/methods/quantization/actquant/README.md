# ActQuant — author GGUF runtime on OFT

The implementation uses the unmodified author revision
`b64791125070652fe6b554e244fe809c79ef5246`, supplied through `--runtime-path`.
It supports OFT LIBERO joint H8 only; see [results](../../../../../docs/benchmark_results.md).

CogACT is unsupported: the floating GGUF bridge did not pass action parity.
This is an engineering boundary, not mathematical incompatibility; action-only
Fisher and native DDIM differentiation are available.

## Mechanism and scope

HSIC-IB scores allocate GGUF types to 224 LLM projection matrices using the author's
layer-balanced greedy allocator. AMF combines action L1 and token NLL gradients
before squaring, then supplies the importance matrix to the author's quantizer.
Both contributions are required. Vision, projectors, action head, embeddings and
output projection are not quantized in this port. The budget is projection
bits/weight, not the full model's average precision.

Calibration is 200 balanced distinct **training** episodes, one complete H8 chunk
per episode, fixed seed, batch one, alpha 0.5, no optimizer updates. Sampling uses
one chunk per episode, not every frame of the author's trajectories.
This is a disclosed bounded calibration adaptation, not the paper's data-budget reproduction.
Raw RLDS gripper actions use the native LIBERO transform before quantile normalization.
HSIC uses normalized demonstration actions, all-token pooling, CCA, median RBF
bandwidth and lx=ly=1. Its signed F_out follows author allocation semantics;
negative scores are not silently replaced by absolute values. Score distributions
must be inspected before interpreting the allocation.

The native joint model's action readout is `hidden[:, -58:-2]`; calibration uses
that verified readout and corresponding preceding NLL positions, not the author's
different trailing slice. The calibration action head is FP32 as in the author
tool. Complete parameter gradients are squared and accumulated on CPU after each
backward pass; this saves memory without a training update or partial-gradient
approximation. The bounded training pool and native readout adaptation are disclosed.

## Runtime and controls

The external engine is installed through a generic backend contract; evaluation,
chunk control and timing have no ActQuant-specific branches. The pinned engine
clears KV memory on every query. Suite changes rebuild normalization state.
Preprocessing remains native center crop, raw proprio for author normalization,
and exactly one gripper conversion. Native PyTorch modules move to CPU and do not
participate in inference. Conversion is irreversible; native needs a fresh backend.

`actquant_floating` is an auxiliary same-engine control, **not another algorithm**.
It retains BF16 matrix weights, F32 vectors/patch convolution and FP16 KV with GGML
arithmetic. Compare the quantized method to native BF16 primarily, and to this
floating control to separate runtime effects. Do not label the control full-BF16
arithmetic or replace the native reference. Torch cannot measure GGML allocations;
its peak-memory fields are missing, never zero or purported full-model memory.
The verified quantized CUDA path dispatches packed MMQ with internal Q8_1
activation blocks. It is not native BF16-activation or W4A16 execution. Kernel
evidence belongs to the actual artifact and runtime build; floating-runtime
results cannot substitute for calibrated quantization measurements.

The isolated runtime layout is `source/` at the pinned revision and `build/bin/`
containing `openvla_oft.so` and `llama-quantize`. The extension's shared-library
dependencies must be available in the launching process. Do not alter global
CUDA/library settings or install author tooling into an unrelated backend env.

`evb calibrate` creates HSIC/AMF evidence. The module
`efficient_vla_benchmark.methods.quantization.actquant.quantize` accepts an
experiment config, `--calibration-path`, validated `--floating-path`,
`--runtime-path` and a new `--output`. It invokes the pinned allocator and quantizer,
checks actual GGUF tensor types, and produces an immutable runtime artifact.
All paths are CLI-only. Each runtime artifact binds checkpoint, calibration,
allocation, weight files and compiled binary checksums. Uncalibrated Q4 diagnostic
exports cannot satisfy this contract and are never formal ActQuant results.
Launch offline GGUF packing with `CUDA_VISIBLE_DEVICES=""`: the author CPU
quantizer otherwise initializes idle CUDA contexts on all visible devices.
This setting is for packing only, not CUDA policy evaluation.
