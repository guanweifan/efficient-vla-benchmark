from types import SimpleNamespace

import pytest
from efficient_vla_backend_cogact.config import CogACTParameters, validate
from efficient_vla_backend_cogact.registration import register


def config(**overrides):
    return SimpleNamespace(
        **{
            "model_variant": "libero_joint",
            "precision": "bfloat16",
            "execution_horizon": 8,
            "backend_parameters": {},
            **overrides,
        }
    )


def test_registration_and_defaults():
    assert register().name == "cogact"
    assert CogACTParameters().num_ddim_steps == 10
    validate(config())
    validate(config(execution_horizon=16))


@pytest.mark.parametrize(
    "overrides",
    [
        {"precision": "float16"},
        {"model_variant": "other"},
        {"execution_horizon": 17},
        {"backend_parameters": {"unknown": 1}},
        {"backend_parameters": {"num_ddim_steps": True}},
        {"backend_parameters": {"num_ddim_steps": 0}},
        {"backend_parameters": {"cfg_scale": float("nan")}},
        {"backend_parameters": {"cfg_scale": 0.5}},
    ],
)
def test_invalid_configuration(overrides):
    with pytest.raises(ValueError):
        validate(config(**overrides))
