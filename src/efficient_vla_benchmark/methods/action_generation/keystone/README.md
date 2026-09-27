# KeyStone

[Geometry Guided Self-Consistency for Physical AI](https://arxiv.org/html/2605.08638v1)
equations 3--6: encode context once, draw independent action noises, batch the
action expert, then select a real sampled chunk with a unimodality-guarded
cluster medoid. No action averaging, auxiliary judge, training, or additional
vision/language forward is introduced.

This independent CogACT port expands only the native DDIM sampler input.
Conditional/unconditional CFG halves preserve candidate pairing. Native ten-step
sampling, precision, normalization, clipping, gripper conversion and H8 remain
unchanged. L2 selection uses the executed eight-step prefix of normalized,
pre-clipping action predictions; the whole selected sixteen-step chunk is returned.
This is the author's optional executed-prefix selection, not a changed cadence.

[Author selector](https://github.com/dywsjtu/keystone/blob/29a15a606f7a32251ef48ac66c281e10c6785eb0/src/keystone/selection.py)
uses deterministic first-C initialization, up to ten Lloyd updates, allclose
stopping, and clustering only when spread > tau. We follow these executable
conventions; the paper prose instead mentions random centroid initialization.
Default C=2, tau=0.3. K=3,4,8,16 are profiling candidates, not certified selections.
K=1 disables consensus; with K=C=2 both singleton clusters tie and select the first
sample. Neither is admitted as an active speed-oriented primary configuration.

Candidate generation, geometry, CPU/GPU synchronizations and postprocessing are
inside common full-cost timing. This method trades additional expert computation
for consistency; matching single-sample latency on another device is not a promise
of acceleration here. Never reduce native sampler steps merely to manufacture a gain.
The OFT deterministic L1 head has no independent stochastic action candidates,
so this method is not registered there.

Disabled mode attaches no sampler wrapper. Reset clears query state; close restores
the sampler instance and is idempotent. No global model class is patched. RNG is
seeded by the common episode harness; additional candidates genuinely consume
additional noise and subsequent policy queries use the resulting deterministic
stream.

Run CPU mechanism tests and the explicit real-model test before measurements:

```bash
python -m pytest src/efficient_vla_benchmark/methods/action_generation/keystone/tests -m 'not gpu'
python -m pytest src/efficient_vla_benchmark/methods/action_generation/keystone/tests -m gpu \
  --model-path "$MODEL_DIR" --trace-path "$TRACE"
```

Implementation and measurement status are separate. Refer to the canonical result
ledger for measured outcomes; no paper score or speedup is imputed by registration.
