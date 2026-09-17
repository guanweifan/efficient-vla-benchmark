# Rebuild the tested OFT environment

The first release targets Linux x86-64, Python 3.10, PyTorch 2.2/CUDA 12.1 and
Ampere GPUs. The recorded functional and timing checks use RTX 3090. Other GPUs
need compatible extension builds and their own validation; no cross-device
performance portability is implied.

Initialize Git submodules at the parent repository's recorded commits. Do not
replace OFT's pinned Transformers with stock Transformers or another algorithm's
fork. CogACT and StarVLA sources are present but are not installed in this process.

`backends/openvla_oft/requirements.txt` pins the dependency closure of the tested runtime
and CPU development tools. Source projects are installed separately with
`--no-deps` so their floating Git dependencies cannot override pinned versions.
The dependency lock is also installed with `--no-deps`, avoiding a transient
installation of a different Transformers release through transitive requirements.
The final `pip check` validates the combined environment after source installation.
The lock additionally includes LIBERO/BDDL runtime imports such as `future` and
`easydict`, which their package metadata does not fully declare. `pip check` alone
is insufficient: the independent installation was also exercised through real
LIBERO episodes.
OFT and LIBERO use editable compatibility mode to expose their namespace packages.

Supply compatible FlashAttention and AWQ extension wheels and a new environment
location explicitly:

```bash
python3.10 backends/openvla_oft/scripts/create_environment.py \
  --venv "$ENV_DIR" \
  --flash-attn-wheel "$FLASH_ATTN_WHEEL" \
  --awq-wheel "$AWQ_WHEEL"
```

For restricted or slow networks, build/download the pinned wheels on a
compatible Linux/Python-3.10 host with the build tools installed, transfer that directory, and add
`--wheelhouse "$WHEELHOUSE_DIR"` to the installer. This uses `--no-index`:

```bash
python3.10 -m pip wheel --wheel-dir "$WHEELHOUSE_DIR" \
  -r backends/openvla_oft/requirements.txt pip==26.0.1 setuptools==79.0.1 wheel==0.46.3
```

The script refuses an existing environment and does not modify system CUDA, global
Python packages, another project environment or Conda configuration. Generated
environment files are private machine-local assets, not repository configuration.
An interrupted installation is left for inspection; use another new destination
or repair that isolated environment explicitly rather than deleting user data.

The FlashAttention 2.5.5 Python-3.10/Torch-2.2 CUDA wheel is available from the
[official release](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.5.5).
Build AWQ's engine from the pinned `third_party/runtimes/llm_awq/awq/kernels` source using
a matching CUDA toolkit. For the tested Ampere build, `TORCH_CUDA_ARCH_LIST=8.6`
was used. Supply the toolkit through the build command's `CUDA_HOME`, and keep
wheel outputs outside tracked source. A compatible prebuilt wheel avoids needing
that toolkit on every inference worker; the NVIDIA driver is still required.

In an isolated build environment already containing the tested Torch version:

```bash
CUDA_HOME="$CUDA_TOOLKIT_DIR" TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4 \
  python -m pip wheel --no-deps --no-build-isolation \
  third_party/runtimes/llm_awq/awq/kernels --wheel-dir "$WHEELHOUSE_DIR"
```

Verified transferred wheel identities (SHA-256, not universal build outputs):

| Wheel | SHA-256 |
|---|---|
| `flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl` | `3fc5d8813904d32cfc77aa4b88d40169dbc12053edb6ee0f1d94159527d05ab0` |
| `awq_inference_engine-0.0.0-cp310-cp310-linux_x86_64.whl` | `b04dc34c425bef26b11372771a236c05c3306eac183ce8dcfb47d71917160618` |

An independently rebuilt wheel can have a different digest; validate its numerical
behavior and record the installed extension digest rather than claiming identity.

AWQ's top-level application package is intentionally not installed: its pinned
metadata requires different Torch/Transformers versions. Selected source
primitives are loaded under a private namespace; their actual packed kernels
receive numerical tests on the installed runtime.

The same AWQ extension includes BF16 and FP16 execution branches with different
accumulation precision. The optional `awq_fp16_kernel.yaml` preset exposes FP16
execution explicitly; it needs no replacement Torch/Transformers package or new
calibration. See the method README for numeric and measurement caveats.

After installation, run `python -m pip check`, CPU tests, the privacy scanner and
explicit GPU checks. Configure LIBERO through the CLI's `--libero-root` and output
directory; do not overwrite a user's global LIBERO configuration. The dataset is
needed for AWQ calibration, not for ordinary closed-loop evaluation.
