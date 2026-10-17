from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--model-path", type=Path, default=None)
    parser.addoption("--trace-path", type=Path, default=None)
    parser.addoption("--artifact-path", type=Path, default=None)
    parser.addoption("--runtime-path", type=Path, default=None)


@pytest.fixture(scope="session")
def model_path(request):
    path = request.config.getoption("--model-path")
    if path is None:
        pytest.skip("Pass an explicit --model-path for GPU integration tests")
    return path


@pytest.fixture(scope="session")
def trace_path(request):
    path = request.config.getoption("--trace-path")
    if path is None:
        pytest.skip("Pass an explicit --trace-path for GPU integration tests")
    return path


@pytest.fixture(scope="session")
def artifact_path(request):
    path = request.config.getoption("--artifact-path")
    if path is None:
        pytest.skip("Pass an explicit --artifact-path for quantized model integration tests")
    return path


@pytest.fixture(scope="session")
def runtime_path(request):
    path = request.config.getoption("--runtime-path")
    if path is None:
        pytest.skip("Pass an explicit --runtime-path for optional execution-engine tests")
    return path
