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
No CogACT package is needed.

After installing the pinned native dependencies and main package:

```bash
python -m pip install --no-deps --no-build-isolation -e backends/openvla_oft
```

The environment installer and complete commands are documented in [INSTALL.md](INSTALL.md).
Model and dataset locations are CLI arguments, never defaults. The plugin supports
API version 1. A new model variant requires explicit configuration, parity tests and
method-compatibility metadata; do not silently treat an arbitrary OFT checkpoint as
the validated joint-suite checkpoint.

## Model asset

Use the author's [joint-four-suite checkpoint](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10)
at revision `638918f3d1c2e43a39a8a20772bdb8b91835e4b7`, not four independently trained
suite checkpoints. The four weight shards, action head and proprio projector in
the experimental asset were SHA-256 checked against that release's LFS metadata.
Download explicitly with the installed Hugging Face CLI:

```bash
hf download moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
  --revision 638918f3d1c2e43a39a8a20772bdb8b91835e4b7 \
  --exclude 'lora_adapter/*' --local-dir "$MODEL_DIR"
```

Keep the tokenizer/processor/configuration/statistics files, all four safetensors
shards and their index, and exactly one `action_head--*_checkpoint.pt` and
`proprio_projector--*_checkpoint.pt`. The backend loads pinned native model code,
not arbitrary Python from the downloaded model. It does not download missing
weights automatically. `evb check-model` validates loading and records file hashes.
Model-card terms apply separately from this repository's code license.
