"""Native sampler parity and actual NFE, without replacing the denoiser equations."""

from unittest.mock import patch

import numpy as np
import pytest


@pytest.mark.parametrize(
    "steps,grid", [(4, [0, 25, 50, 75]), (5, [0, 20, 40, 60, 80]), (10, list(range(0, 100, 10)))]
)
def test_upstream_native_respacing(steps, grid):
    from action_model import create_diffusion

    sampler = create_diffusion(
        f"ddim{steps}",
        noise_schedule="squaredcos_cap_v2",
        diffusion_steps=100,
        sigma_small=True,
        learn_sigma=False,
    )
    assert sampler.timestep_map == grid
    assert sampler.num_timesteps == steps


@pytest.mark.gpu
@pytest.mark.parametrize("steps", [4, 5, 10])
def test_real_native_control_parity_nfe_rng_and_reset(steps, model_path, trace_path, request):
    import torch
    from efficient_vla_backend_cogact.backend import CogACTBackend
    from efficient_vla_backend_cogact.config import CogACTParameters
    from efficient_vla_backend_cogact.processor import process_actions

    from efficient_vla_benchmark.evaluation.traces import load_trace

    host = CogACTBackend(model_path, parameters=CogACTParameters(num_ddim_steps=steps))
    request.addfinalizer(host.close)
    context, frames, _ = load_trace(trace_path, require_control_steps=True)
    host.reset(context)
    prepared = host.prepare(frames[0])
    net = host.model.action_model.net
    original = net.forward_with_cfg
    calls = []

    def observed(x, t, **kwargs):
        calls.append((x.shape[0], t.detach().cpu().tolist(), kwargs["cfg_scale"]))
        return original(x, t, **kwargs)

    with patch.object(net, "forward_with_cfg", observed):
        actual = host.predict_chunk(frames[0])
    rng = torch.cuda.get_rng_state().clone()
    assert host.model.action_model.diffusion_steps == 100
    grid = list(reversed(host.model.action_model.ddim_diffusion.timestep_map))
    assert len(calls) == steps
    assert [row[1] for row in calls] == [[t, t] for t in grid]
    assert all(batch == 2 and scale == 1.5 for batch, _, scale in calls)
    assert actual.shape == (16, 7) and np.isfinite(actual).all()
    host.reset(context)
    with torch.inference_mode():
        expected, _ = host.model.predict_action(
            image=prepared["image"],
            instruction=prepared["instruction"],
            unnorm_key=host.unnorm_key,
            cfg_scale=1.5,
            use_ddim=True,
            num_ddim_steps=steps,
            do_sample=False,
        )
    np.testing.assert_array_equal(actual, process_actions(expected))
    assert torch.equal(torch.cuda.get_rng_state(), rng)
    host.reset(context)
    np.testing.assert_array_equal(actual, host.predict_chunk(frames[0]))
    assert torch.equal(torch.cuda.get_rng_state(), rng)
    assert host.method is None
