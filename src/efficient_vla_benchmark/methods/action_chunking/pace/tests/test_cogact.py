import pytest

from ..config import PACEConfig


@pytest.mark.gpu
def test_native_parity_and_shared_control(model_path, trace_path, artifact_path, request):
    from efficient_vla_backend_cogact.backend import CogACTBackend

    from .gpu_checks import check_native

    backend = CogACTBackend(model_path)
    request.addfinalizer(backend.close)
    check_native(backend, PACEConfig(), artifact_path, trace_path, 16)
