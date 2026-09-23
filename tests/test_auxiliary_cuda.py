import pytest
import torch

from efficient_vla_benchmark.model_ops.auxiliary import AuxiliaryCUDARunner


@pytest.mark.gpu
def test_auxiliary_runner_restores_rng_counts_work_and_closes():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda:0")
    torch.manual_seed(431)
    torch.cuda.manual_seed_all(431)
    before_cpu, before_gpu = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    results = []
    runner = AuxiliaryCUDARunner(device)

    def work():
        assert not torch.is_grad_enabled()
        torch.randn(3)
        results.append(torch.randn(16, 16, device=device) @ torch.randn(16, 16, device=device))

    try:
        runner.submit(work)
        with pytest.raises(RuntimeError, match="complete"):
            runner.submit(work)
        metrics = runner.drain()
        assert metrics["auxiliary_forward_ms"] > 0
        assert len(results) == 1 and torch.isfinite(results[0]).all()
        assert torch.equal(torch.get_rng_state(), before_cpu)
        assert torch.equal(torch.cuda.get_rng_state(), before_gpu)
        assert runner.drain() is None
    finally:
        runner.close()
        runner.close()
    with pytest.raises(RuntimeError, match="complete"):
        runner.submit(work)


@pytest.mark.gpu
def test_auxiliary_failure_reaches_join_and_does_not_leak_rng():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    before = torch.cuda.get_rng_state().clone()
    runner = AuxiliaryCUDARunner("cuda:0")

    def work():
        torch.randn(100, device="cuda:0")
        raise ValueError("failed refresh")

    try:
        runner.submit(work)
        with pytest.raises(ValueError, match="failed refresh"):
            runner.drain()
        assert torch.equal(torch.cuda.get_rng_state(), before)
    finally:
        runner.close()
