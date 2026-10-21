# CogACT runtime installation

Use a separate Python 3.10 environment. Stock Transformers 4.40.1 is required;
the OFT fork with the same version string is not an interchangeable dependency.
Never replace packages in the OFT environment or a global site-packages directory.

Run from the repository root. Initialize only the CogACT runtime sources at the
parent gitlinks; no OFT fork is needed:

```bash
git submodule update --init --recursive -- \
  backends/cogact/upstream third_party/runtimes/prismatic_cogact \
  third_party/runtimes/dlimp third_party/benchmarks/libero
python3.10 scripts/create_environment.py --backend cogact \
  --venv "$ENV_DIR" --flash-attn-wheel "$FLASH_ATTN_WHEEL"
source "$ENV_DIR/bin/activate"
python -m pip check
```

The installer uses `requirements.txt`, stock Transformers 4.40.1 and pinned native
sources with `--no-deps`; it never inherits another environment's site-packages.
The native packages import FlashAttention even with SDPA. Supply the Linux x86-64,
Python 3.10/Torch 2.2 FlashAttention 2.5.5 wheel from its
[official release](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.5.5).
An existing destination is rejected. `--wheelhouse "$WHEELHOUSE_DIR"` enables
offline installation from a complete wheel collection; see the
[wheel preparation example](../openvla_oft/INSTALL.md), using this backend's
requirements file instead. Do not install packages into a global environment.
An offline wheelhouse including `pip-26.0.1-py3-none-any.whl` can bootstrap the
isolated environment even when the system Python lacks `ensurepip`.

A model export must contain one complete checkpoint under `checkpoints/`,
`config.json`, four-suite `dataset_statistics.json`, and an offline `tokenizer/`
with tokenizer files and the Llama configuration. Supply its root (or the explicit
checkpoint file) via `--model-path`; training-time resource paths are never reused.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m efficient_vla_benchmark.cli evaluate \
  --config configs/experiments/cogact/baseline.yaml \
  --model-path "$MODEL_DIR" --libero-root third_party/benchmarks/libero \
  --output-dir "$SMOKE_DIR" --record-traces

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m efficient_vla_benchmark.cli benchmark \
  --config configs/experiments/cogact/baseline.yaml \
  --model-path "$MODEL_DIR" --traces "$SMOKE_DIR"/traces/*.npz \
  --output-dir "$TIMING_DIR"
```

GPU selection and CUDA/EGL device mapping are caller-controlled. Development
evaluation is four episodes total, not a formal success-rate estimate. Do not
compare timings from different trajectories/horizons as paired speedups.
The shared evaluator redirects robosuite's hard-coded temporary log to the run's
`_private/robosuite.log` during simulator import. No administrator access, shared
log-file deletion or edits to installed packages are needed.

## Optional quantization runtimes

AWQ, GPTQ and DA-MPA use the pinned `third_party/runtimes/llm_awq` extension;
QuantVLA and HoloQ also require the current-stream derivative for their selected
DiT graph runtime. SmoothQuant and DyQ-VLA use different engines. Consult the
selected method's README before building optional components.

Initialize `third_party/runtimes/llm_awq`, then compile against this environment's
Torch/CUDA/Python ABI, or supply a separately verified matching wheel via the
installer's optional `--awq-wheel`. A wheel working in another environment does
not by itself validate this one. Do not install the author's conflicting
application dependency stack. Select
a compatible CUDA toolkit through `CUDA_HOME` and its `bin` on `PATH`, then run:

```bash
MAX_JOBS=2 TORCH_CUDA_ARCH_LIST="$GPU_ARCH" \
python -m pip install --no-deps --no-build-isolation third_party/runtimes/llm_awq/awq/kernels

python -m efficient_vla_benchmark.cli calibrate \
  --config configs/experiments/cogact/awq.yaml \
  --model-path "$MODEL_DIR" --dataset-path "$RLDS_DIR" --artifact-dir "$ARTIFACT_DIR"
```

Supply `--artifact-path "$ARTIFACT_DIR"` for AWQ evaluate/benchmark runs. The
completed artifact must match this backend, checkpoint, precision and calibration
configuration. Its manifest and every packed layer are checked before mutation.
It is separate from the original checkpoint and is not stored in Git. Calibration
uses 32 training observations across four suites and no gradient updates.

For rendering on a selected physical GPU, the installed robosuite version checks
that `MUJOCO_EGL_DEVICE_ID` appears in `CUDA_VISIBLE_DEVICES`. A remapped Torch
`cuda:0` is not necessarily physical EGL device zero; configure both variables
consistently at launch. This mapping does not belong in experiment YAML.

The trace replay above is a smoke diagnostic, not the frozen four-prefix timing
protocol. For comparable reported measurements, follow
[evaluation inputs](../../docs/development.md#rebuilding-evaluation-inputs).
The exact experimental model is described under [model asset](README.md#model-asset);
an official CogACT checkpoint trained for another benchmark is not a replacement.
