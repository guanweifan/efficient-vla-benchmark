# OpenVLA-OFT backend

Optional distribution: `efficient-vla-backend-openvla-oft`.
Python package: `efficient_vla_backend_openvla_oft`.
Backend ID: `openvla_oft`; supported model variant: `libero_joint`.

The pinned native code in `upstream/` owns model classes. This plugin assembles the
joint-four-suite LIBERO checkpoint read-only: two cameras, 8D proprioception,
eight 7D actions, L1 action head, native bidirectional SDPA and original normalization.
Its processor converts native gripper outputs exactly once to the LIBERO convention.
The shared benchmark neither imposes these dimensions nor imports OFT modules.

## Environment

The tested recipe is Linux, Python 3.10, PyTorch 2.2/CUDA 12.1, pinned OFT Transformers,
FlashAttention 2.5.5, and compatible AWQ kernels when quantization is selected.
`requirements.txt` records the validated dependency closure. Native source packages
are installed separately with `--no-deps`; stock Transformers is not interchangeable.
No CogACT or starVLA package is needed.

After installing the pinned native dependencies and main package:

```bash
python -m pip install --no-deps --no-build-isolation -e backends/openvla_oft
```

The environment installer and complete commands are documented in docs/installation.md.
Model and dataset locations are CLI arguments, never defaults. The plugin supports
API version 1. A new model variant requires explicit configuration, parity tests and
method-compatibility metadata; do not silently treat an arbitrary OFT checkpoint as
the validated joint-suite checkpoint.
