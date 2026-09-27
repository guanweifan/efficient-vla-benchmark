# AutoHorizon

[VLA Knows Its Limits](https://arxiv.org/html/2602.21445v1) estimates execution
horizons from action self-attention. This port follows the released
[bidirectional selector](https://github.com/hatchetProject/AutoHorizon/blob/c7504f1756109103f2cfcc2e23f1b1a23841c885/src/openpi/models_pytorch/pi0_pytorch.py),
not an arbitrary attention threshold. The Apache-2.0 attribution/terms are retained
in root NOTICE; selection code is reorganized and its changes are labeled.

Capture the third denoising call by default, average across all heads/layers,
compute monotone soft pointers and entropy reliability, and join forward/backward
coverage using the author's rule. The released backward index mapping is retained.
Plateau threshold, reliable entropy quantile and consecutive hold length remain
configurable. A maximum prefix cannot exceed the actually predicted sixteen actions.

CogACT's DiT attention includes one cognition token and sixteen action positions.
This binding observes native QKV without replacing fused attention: normalize over
all seventeen keys, then crop action rows/keys. Cropping before softmax or averaging
CFG branches would change the signal. Only the conditional row is inspected.
Native DDIM, CFG, full chunk, normalization, action conversion and RNG are unchanged;
no extra VLM or denoiser forward is introduced. A diffusion-model adaptation is not
a claim of the paper's flow-checkpoint performance.

The original H8 policy is the reference; this method uses an adaptive executed
prefix. Probe, selector and all synchronization cost are included in common
control replay. Reset and close clear captures, remove handles and restore the
original instance forward. Disabled mode has no attention hooks.

CPU tests check reference-equivalent selection, full-key normalization and native
attention output parity. Real-model tests check exact actions/RNG, reset, repeated
construction and cleanup before shared smoke/Dev/timing. Registration alone is
not a successful measurement or a performance claim.

```bash
python -m pytest src/efficient_vla_benchmark/methods/action_chunking/autohorizon/tests -m 'not gpu'
python -m pytest src/efficient_vla_benchmark/methods/action_chunking/autohorizon/tests -m gpu \
  --model-path "$MODEL_DIR" --trace-path "$TRACE"
```
