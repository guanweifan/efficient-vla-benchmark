"""Resolve one selected GPU by UUID and co-locate CUDA and EGL before evaluation.

EGL enumeration order is not nvidia-smi order. Query EGL_NV_device_cuda with
exactly one CUDA-visible device and select its logical CUDA ordinal zero.
No model, simulator, global environment or installed package is modified.
"""

import argparse
import ctypes
import json
import logging
import os
import pyexpat  # noqa: F401 -- bind Python's matching XML runtime before EGL's system libraries
import runpy
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


@contextmanager
def private_simulator_log(directory):
    """Redirect only the legacy simulator's import-time shared log, then restore."""
    handler = logging.FileHandler

    def create(filename, *args, **kwargs):
        if Path(filename).name == "robosuite.log":
            directory.mkdir(parents=True, exist_ok=True)
            filename = directory / "robosuite.log"
        return handler(filename, *args, **kwargs)

    with patch.object(logging, "FileHandler", create):
        yield


def select_gpu(mask):
    if not mask or "," in mask:
        raise ValueError("Select exactly one GPU in CUDA_VISIBLE_DEVICES")
    rows = (
        subprocess.check_output(
            ["nvidia-smi", f"--id={mask}", "--query-gpu=uuid", "--format=csv,noheader"],
            text=True,
            timeout=10,
        )
        .strip()
        .splitlines()
    )
    if len(rows) != 1 or not rows[0].startswith("GPU-"):
        raise ValueError("GPU selection did not resolve to exactly one UUID")
    return rows[0]


def egl_index():
    egl = ctypes.CDLL("libEGL.so.1")
    egl.eglGetProcAddress.restype = ctypes.c_void_p
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    query_pointer = egl.eglGetProcAddress(b"eglQueryDevicesEXT")
    attribute_pointer = egl.eglGetProcAddress(b"eglQueryDeviceAttribEXT")
    if not query_pointer or not attribute_pointer:
        raise RuntimeError("EGL device-query extensions are unavailable")
    query = ctypes.CFUNCTYPE(
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
    )(query_pointer)
    attribute = ctypes.CFUNCTYPE(
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_ssize_t),
    )(attribute_pointer)
    count = ctypes.c_int()
    if not query(0, None, ctypes.byref(count)) or count.value <= 0:
        raise RuntimeError("EGL device enumeration failed")
    devices = (ctypes.c_void_p * count.value)()
    if not query(count.value, devices, ctypes.byref(count)):
        raise RuntimeError("EGL device enumeration failed")
    matches = []
    for index, device in enumerate(devices):
        ordinal = ctypes.c_ssize_t(-1)
        # EGL_CUDA_DEVICE_NV returns a CUDA-visible ordinal, not a PCI/SMI index.
        if attribute(device, 0x323A, ctypes.byref(ordinal)) and ordinal.value == 0:
            matches.append(index)
    if len(matches) != 1:
        raise RuntimeError("Cannot uniquely map the visible CUDA device to EGL")
    return matches[0]


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = select_gpu(os.environ.get("CUDA_VISIBLE_DEVICES"))
    selected_egl = str(egl_index())
    if sys.argv[1:] == ["--probe"]:
        print(json.dumps({"egl_index": int(selected_egl)}))
        return
    if sys.argv[1:3] != ["-m", "efficient_vla_benchmark.cli"]:
        raise ValueError("This launcher supports only the benchmark CLI module")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    args, _ = parser.parse_known_args(sys.argv[3:])
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    # Pinned robosuite 1.4.x incorrectly compares the EGL index to the CUDA mask
    # at import time. Import its binding with no explicit EGL override, then set
    # the verified index before any simulation/context is constructed. CUDA
    # visibility stays fixed to ONE UUID throughout; no source/func patch occurs.
    os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
    try:
        with private_simulator_log(args.output_dir / "_private"):
            import robosuite.utils.binding_utils  # noqa: F401
    finally:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = selected_egl
    sys.argv = sys.argv[2:]
    runpy.run_module("efficient_vla_benchmark.cli", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
