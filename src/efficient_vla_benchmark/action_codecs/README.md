# Independent action codec evaluation

This component evaluates discrete action representations, not policy acceleration.
It never replaces an OFT/CogACT action head or borrows their success-rate claims.
Each codec owns one adapter directory; common corpus, packet and measurement code
knows no codec/model names. Tensor runtimes are loaded only by selected adapters.

## Contract

- Input/output: CPU float32 raw LIBERO OSC commands, `[frames, 7]`, signed gripper.
- Encode returns integer tokens only. Decode receives those tokens, horizon, suite
  identity and one initial observed state; no future actions/observations or latent
  encoder features. Required initial-state side information must be disclosed.
- Preserve released codec weights/quantizers; no training or codebook/BPE refitting.
  Closed-form calibration uses only the frozen calibration split and is not updated
  during reconstruction. Pretrained codecs may already have seen LIBERO.
- The main group uses forty-frame windows; native five-/eight-/twenty-frame
  codecs tile them exactly. A separately frozen sixteen-frame prefix group
  supports native sixteen-frame codecs, with references rerun on those SAME prefixes.
  The original episode set cannot all support eighty frames: do not replace short
  episodes to manufacture a least-common-multiple workload. Flexible codecs
  receive the complete group window. Do not pad, drop or resample inside adapters.
  This compares representations on one workload, not identical model receptive fields.
- A seeded demonstration corpus has disjoint calibration/reconstruction episode IDs,
  one window per episode and equal counts across four suites. It is separate from
  simulator Dev/final episodes, but is NOT claimed as unseen pretraining data.
- Report raw-command error per dimension, aggregate MSE/MAE, gripper agreement,
  tokens/frame and nominal fixed-width bits/frame. Nominal bits exclude shared model,
  statistics and known context; they are not an actual entropy-coded file size.
- Batch one, ten warmups, one measured pass. Include normalization, conversion,
  host/device transfers and token extraction; exclude loading and disk I/O. Encode
  and decode are separately synchronized. Never turn codec latency into VLA speedup.
- A failed decode remains an error; no zero fallback or successful-only averages.
  Record exact corpus, weights, runtime, source, calibration and adapter identities.

Prepare in the existing RLDS data environment. For codec evaluation, use a separate
Python 3.10 environment; do not install over either policy environment:

```bash
python3.10 -m venv .venv-codecs
.venv-codecs/bin/python -m pip install \
  -r src/efficient_vla_benchmark/action_codecs/requirements.txt
.venv-codecs/bin/python -m pip install -e .
```

The validated GPU build is PyTorch 2.5.1+cu124. Choose the matching official CUDA
wheel for the host; do not change policy packages. Assets are supplied separately
and verified against each adapter's pinned manifest. VQ-VLA and X-Tokenizer need
their small pinned source subsets, obtained without cloning whole policy projects:

```bash
python -m efficient_vla_benchmark.action_codecs.fetch_sources \
  --output "$CODEC_SOURCES"
```

Run preparation/evaluation in the corresponding environment:

```bash
python -m efficient_vla_benchmark.action_codecs.prepare \
  --dataset-root "$RLDS_ROOT" --output "$CORPUS_DIR"
python -m efficient_vla_benchmark.action_codecs.run \
  --config configs/codecs/fast_plus.yaml --assets "$FAST_PLUS_DIR" \
  --corpus "$CORPUS_DIR" --output "$RESULT_DIR"
```

Choose `configs/codecs/{fast_plus,beast,vq_vla,actioncodec,x_tokenizer}.yaml`.
VQ-VLA/X-Tokenizer additionally take `--sources` pointing to their fetched subset;
GPU adapters take `--device cuda:0`. FAST+ uses native CPU DCT/BPE. BEAST's basis5,
basis10 and basis20 presets expose rate–distortion choices, not policy tuning.
Only released quantizers are used; each adapter README states representation,
calibration, tiling and decoder-side-information limits.

Additional presets: uniform_binning, minivla_rvq, spatialvla, mint_sdat,
actioncodec_rvq and its q1/q2 prefixes. RVQ prefixes are one released method's
rate-distortion points, not separately trained methods. New released assets are
available through the portable, explicit downloader:

```bash
python -m efficient_vla_benchmark.action_codecs.fetch_assets \
  --output "$CODEC_ASSETS" --codec minivla_rvq --codec mint_sdat \
  --codec spatialvla --codec actioncodec_rvq
python -m efficient_vla_benchmark.action_codecs.prepare \
  --source-corpus "$CORPUS_DIR" --horizon 16 --output "$SHORT_CORPUS_DIR"
```

Before measurement, serialize only integer IDs and reconstruct with a separately
loaded decoder using `verification.independent_decode`. Opt-in released-asset
tests also check native equations and malformed packets; see tests/test_codec_released.py.
Do not use reconstruction outcomes to select the checkpoint component or corpus.
MINT's published runtime uses multiscale residual quantization; DCT belongs to its
training loss and is not an inference transform.

Each result records per-window timestamps plus a measured-pass interval for
external GPU-quality checks. Use one clean frozen source and an idle GPU, record
thermal flags throughout that interval, and retain failed quality checks separately
from accepted latency. This sampler cannot certify absence of sub-poll transients.
The timing protocol is still one pass, not a best-of-repeat. Codec dependencies
and measurement records stay isolated from policy results.

The dataset's nominal frame rate is not assumed. Native embodiment labels may name
another rate; without a verified conversion, report frames rather than physical Hz.

## Survey coverage and admission audit

Paper cutoff: 2026-08-17; resource audit: 2026-10-06. This is an audited selection,
not a claim that every action representation has a transferable released codec.
Public availability below means the checked official entries, not proof that no
other checkpoint exists. Literature inclusion does not require a local result.

| Work / source | Local evidence or concrete remaining condition |
|---|---|
| [Uniform binning](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/vla/action_tokenizer.py) | Classical scalar reference; exact released digitize/midpoint convention |
| [FAST / FAST+](https://arxiv.org/abs/2501.09747) | Released universal DCT/BPE codec; no target BPE fitting |
| [BEAST](https://huggingface.co/zhouhongyi/beast) | B-spline codec, three retained basis counts; one family |
| [VQ-VLA](https://github.com/xiaoxiao0406/VQ-VLA) | Released five-frame residual-VQ codec |
| [MiniVLA RVQ](https://github.com/Stanford-ILIAD/openvla-mini) | Released eight-frame RVQ component; includes parity against its VQ-BeT dependency, not full VQ-BeT policy |
| [ActionCodec](https://arxiv.org/abs/2602.15397) | Base plus released RVQ-finetuned weights and three complete-layer prefixes; one family, no local training |
| [MINT / SDAT](https://arxiv.org/abs/2602.08602) | Released sixteen-frame multiscale residual tokenizer; separate matched H16 group |
| [SpatialVLA](https://arxiv.org/abs/2501.15830) | Released spatial grids; normalized-coordinate transfer diagnostic, not a physical Euler-angle or policy comparison |
| [X-Tokenizer](https://github.com/X-Square-Robot/X-Tokenizer) | Separate initial-state-conditioned, command-integrated pose diagnostic; not native pose reconstruction |
| [MoEActok](https://github.com/cpaaax/MoEActok/tree/b0db8e0cbd1aec020c872de1883738ca17b6d327) | Inspected released Bridge weights. Encoder needs action expert labels; no verified frozen clustering/routing map accompanying them. Do not use an oracle expert or retrain clusters with arbitrary expert ordering |
| [CARP / MSAT](https://github.com/ZhefeiGong/carp/tree/3ca00f7d1810b6cb0590d0cc8eec828d25843363) | Released MimicGen weights AND normalizers exist; native 10D absolute-pose target is not the frozen 7D OSC-command corpus. Requires a separate native-pose protocol, not a claim of fundamental incompatibility |
| [RDT2 / RVQ-AT](https://huggingface.co/robotics-diffusion-transformer/RVQActionTokenizer/tree/a6a614ae049cc3b9fa6bd4f5fadaeee3375d4dec) | Released 24-frame, dual-arm 20D relative-EE/rotation6D codec. Missing native dual-arm target data in this corpus; do not silently pad an arm or resample |
| [LipVQ-VAE](https://github.com/andvg3/LipVQ-VAE/tree/7bed406f13d8b53f60ec2a2ba657832ebfba193c) | Downloaded and inspected official checkpoint: discrete LLFQVAE_V4, 1024 entries, 12D mobile RoboCasa actions and identity normalization. No verified matching 7D release; arm-only projection would be a new cross-embodiment diagnostic, not an equivalent native result |
| [OAT](https://ordered-action-tokenization.github.io/) | Original and full-paper versions are one family. Official OAT/Praxis code documents training; no matching released tokenizer checkpoint verified in this audit |
| [NAC](https://github.com/ahadjawaid/nac) | Multiscale adversarial tokenizer training code available; no usable released tokenizer checkpoint verified |
| [QueST](https://github.com/pairlab/QueST) | Related robot skill-quantization reference; official instructions require tokenizer training, no matching released checkpoint verified |
| [FASTer / FASTerVQ](https://arxiv.org/abs/2512.04952) | Literature-only: no executable official codec plus checkpoint verified |
| [OmniSAT](https://arxiv.org/abs/2510.09667) | Official RoboBrain-X0 preview uses FAST; do not count that checkpoint as the paper's GRVQ/B-spline tokenizer |
| [FACT / GenieReasoner](https://github.com/GenieReasoner/GenieReasoner/tree/7acebf96f8c8e79fe112d3b548587b7b68ff7178) | Official entry is a project website/paper; ERIQ is a reasoning benchmark, not released FACT codec weights |
| [SA-VLA state-aware tokenizer](https://arxiv.org/abs/2606.30113) | State-conditioned literature coverage; no matching implementation/checkpoint verified. Not the same-named spatial-RL project |

Stable-FAST's variable-length/training-stability claim cannot be validated by a
fixed-window reconstruction test alone. CATok's pre-cutoff publication identity and
usable release remain unverified. MolmoAct2's FAST implementation is not another
tokenizer family. Post-cutoff works such as ActionPiece, BAAT and TOAST are excluded.
These dispositions must not be described as methods being inherently unsuitable.

Writing use: distinguish token count from alphabet-aware payload, expose
rate–distortion within one codec, and separate released-model transfer limitations
from quantizer design. Do not infer closed-loop success, autoregressive query count,
unseen-domain generalization, or VLA speedup from reconstruction alone.
