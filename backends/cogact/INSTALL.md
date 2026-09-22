# CogACT runtime installation

Use a separate Python 3.10 environment. Stock Transformers 4.40.1 is required;
the OFT fork with the same version string is not an interchangeable dependency.
Never replace packages in the OFT environment or a global site-packages directory.

Initialize the existing project submodules for this backend, the Prismatic runtime,
dlimp and LIBERO. Their exact revisions are recorded by the parent gitlinks.
Install PyTorch 2.2.0 / torchvision 0.17.0 for the appropriate CUDA runtime and the
dependencies declared by the native packages. Then install native packages without
letting pip fetch unpinned Git dependencies:

```bash
python -m pip install transformers==4.40.1 timm==0.9.10
python -m pip install --no-deps --no-build-isolation \
  third_party/runtimes/dlimp third_party/runtimes/prismatic_cogact backends/cogact/upstream
python -m pip install --no-deps --no-build-isolation -e third_party/benchmarks/libero
python -m pip install --no-deps --no-build-isolation -e . -e backends/cogact
```

When using an explicitly selected existing environment as read-only system packages,
install stock Transformers and native packages into the new backend environment
with `--ignore-installed --no-deps` where necessary. Verify actual module locations;
a satisfied version string alone does not prove that the stock implementation won.

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

## Optional AWQ kernel

Only AWQ needs the pinned `third_party/runtimes/llm_awq` extension. Compile it
inside this CogACT environment against its Torch/CUDA ABI; do not copy the OFT
binary or install the author's conflicting application dependency stack. Select
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
